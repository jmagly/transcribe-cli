"""Batch processing for multiple files.

Implements Sprint 4: Batch Processing
- Directory scanning for audio/video files
- Concurrent transcription with semaphore control
- Progress tracking and error handling
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

from .extractor import SUPPORTED_EXTENSIONS, is_supported_file
from .transcriber import (
    TranscriptionResult,
    transcribe_file,
)


@dataclass
class BatchResult:
    """Result of a single file in batch processing."""

    input_path: Path
    output_path: Optional[Path]
    success: bool
    error: Optional[str] = None
    result: Optional[TranscriptionResult] = None


@dataclass
class BatchSummary:
    """Summary of batch processing results."""

    total_files: int
    successful: int
    failed: int
    skipped: int
    results: list[BatchResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful / self.total_files) * 100


def scan_directory(
    directory: Path,
    recursive: bool = False,
) -> list[Path]:
    """Scan directory for supported audio/video files.

    Args:
        directory: Directory to scan.
        recursive: Whether to scan subdirectories.

    Returns:
        List of paths to supported media files.

    Raises:
        FileNotFoundError: If directory doesn't exist.
        ValueError: If path is not a directory.
    """
    directory = Path(directory).resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    files = []
    pattern = "**/*" if recursive else "*"

    for path in directory.glob(pattern):
        if path.is_file() and is_supported_file(path):
            files.append(path)

    # Sort for consistent ordering
    files.sort()
    return files


async def _process_file_async(
    input_path: Path,
    output_dir: Optional[Path],
    output_format: Literal["txt", "srt"],
    language: str,
    api_key: Optional[str],
    semaphore: asyncio.Semaphore,
    progress_callback: Optional[Callable[[Path, str], None]] = None,
) -> BatchResult:
    """Process a single file asynchronously.

    Args:
        input_path: Path to input file.
        output_dir: Output directory (None = same as input).
        output_format: Output format.
        language: Language code or "auto".
        api_key: OpenAI API key.
        semaphore: Semaphore for concurrency control.
        progress_callback: Optional callback for progress updates.

    Returns:
        BatchResult with success/failure status.
    """
    async with semaphore:
        if progress_callback:
            progress_callback(input_path, "started")

        try:
            # Determine output path
            if output_dir:
                output_path = output_dir / f"{input_path.stem}.{output_format}"
            else:
                output_path = input_path.with_suffix(f".{output_format}")

            # Run transcription in thread pool (blocking I/O)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: transcribe_file(
                    input_path=input_path,
                    output_path=output_path,
                    language=language,
                    api_key=api_key,
                ),
            )

            # Save with formatter
            from transcribe_cli.output import save_formatted_transcript

            saved_path = await loop.run_in_executor(
                None,
                lambda: save_formatted_transcript(result, output_path, output_format),
            )

            if progress_callback:
                progress_callback(input_path, "completed")

            return BatchResult(
                input_path=input_path,
                output_path=saved_path,
                success=True,
                result=result,
            )

        except Exception as e:
            if progress_callback:
                progress_callback(input_path, "failed")

            return BatchResult(
                input_path=input_path,
                output_path=None,
                success=False,
                error=str(e),
            )


async def process_batch_async(
    files: list[Path],
    output_dir: Optional[Path] = None,
    output_format: Literal["txt", "srt"] = "txt",
    language: str = "auto",
    concurrency: int = 5,
    api_key: Optional[str] = None,
    progress_callback: Optional[Callable[[Path, str], None]] = None,
) -> BatchSummary:
    """Process multiple files concurrently.

    Args:
        files: List of files to process.
        output_dir: Output directory (None = same as input).
        output_format: Output format for all files.
        language: Language code or "auto".
        concurrency: Maximum concurrent transcriptions.
        api_key: OpenAI API key.
        progress_callback: Optional callback(path, status) for progress.

    Returns:
        BatchSummary with results for all files.
    """
    if not files:
        return BatchSummary(
            total_files=0,
            successful=0,
            failed=0,
            skipped=0,
            results=[],
        )

    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    # Create tasks for all files
    tasks = [
        _process_file_async(
            input_path=f,
            output_dir=output_dir,
            output_format=output_format,
            language=language,
            api_key=api_key,
            semaphore=semaphore,
            progress_callback=progress_callback,
        )
        for f in files
    ]

    # Run all tasks
    results = await asyncio.gather(*tasks)

    # Calculate summary
    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    return BatchSummary(
        total_files=len(files),
        successful=successful,
        failed=failed,
        skipped=0,
        results=list(results),
    )


def process_batch(
    files: list[Path],
    output_dir: Optional[Path] = None,
    output_format: Literal["txt", "srt"] = "txt",
    language: str = "auto",
    concurrency: int = 5,
    api_key: Optional[str] = None,
    progress_callback: Optional[Callable[[Path, str], None]] = None,
) -> BatchSummary:
    """Process multiple files (synchronous wrapper).

    Args:
        files: List of files to process.
        output_dir: Output directory (None = same as input).
        output_format: Output format for all files.
        language: Language code or "auto".
        concurrency: Maximum concurrent transcriptions.
        api_key: OpenAI API key.
        progress_callback: Optional callback(path, status) for progress.

    Returns:
        BatchSummary with results for all files.
    """
    return asyncio.run(
        process_batch_async(
            files=files,
            output_dir=output_dir,
            output_format=output_format,
            language=language,
            concurrency=concurrency,
            api_key=api_key,
            progress_callback=progress_callback,
        )
    )


def process_directory(
    directory: Path,
    output_dir: Optional[Path] = None,
    output_format: Literal["txt", "srt"] = "txt",
    language: str = "auto",
    concurrency: int = 5,
    recursive: bool = False,
    api_key: Optional[str] = None,
    progress_callback: Optional[Callable[[Path, str], None]] = None,
) -> BatchSummary:
    """Scan directory and process all supported files.

    Args:
        directory: Directory to scan.
        output_dir: Output directory (None = same as input).
        output_format: Output format for all files.
        language: Language code or "auto".
        concurrency: Maximum concurrent transcriptions.
        recursive: Whether to scan subdirectories.
        api_key: OpenAI API key.
        progress_callback: Optional callback(path, status) for progress.

    Returns:
        BatchSummary with results for all files.
    """
    files = scan_directory(directory, recursive=recursive)

    return process_batch(
        files=files,
        output_dir=output_dir,
        output_format=output_format,
        language=language,
        concurrency=concurrency,
        api_key=api_key,
        progress_callback=progress_callback,
    )
