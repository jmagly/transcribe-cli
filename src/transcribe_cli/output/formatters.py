"""Output formatters for transcription results.

Implements Sprint 4: Output Formats
- TXT plain text format
- SRT subtitle format with timestamps
"""

from pathlib import Path
from typing import Literal

import srt
from datetime import timedelta

from transcribe_cli.core.transcriber import TranscriptionResult, TranscriptionSegment


def _seconds_to_timedelta(seconds: float) -> timedelta:
    """Convert seconds to timedelta for SRT timestamps.

    Args:
        seconds: Time in seconds (can include milliseconds as decimals).

    Returns:
        timedelta object for use with srt library.
    """
    return timedelta(seconds=seconds)


def format_as_txt(result: TranscriptionResult) -> str:
    """Format transcription result as plain text.

    Args:
        result: TranscriptionResult to format.

    Returns:
        Plain text transcript.
    """
    return result.text.strip()


def format_as_srt(result: TranscriptionResult) -> str:
    """Format transcription result as SRT subtitles.

    Args:
        result: TranscriptionResult with segments.

    Returns:
        SRT formatted string with timestamps.

    Raises:
        ValueError: If no segments are available.
    """
    if not result.segments:
        # If no segments, create a single subtitle from full text
        # This can happen if API returned text without detailed segments
        if result.text and result.duration:
            subtitle = srt.Subtitle(
                index=1,
                start=timedelta(seconds=0),
                end=timedelta(seconds=result.duration),
                content=result.text.strip(),
            )
            return srt.compose([subtitle])
        raise ValueError(
            "Cannot create SRT: no segments available. "
            "The transcription may not have timestamp information."
        )

    subtitles = []
    for i, segment in enumerate(result.segments, start=1):
        subtitle = srt.Subtitle(
            index=i,
            start=_seconds_to_timedelta(segment.start),
            end=_seconds_to_timedelta(segment.end),
            content=segment.text.strip(),
        )
        subtitles.append(subtitle)

    return srt.compose(subtitles)


def format_transcript(
    result: TranscriptionResult,
    output_format: Literal["txt", "srt"] = "txt",
) -> str:
    """Format transcription result in specified format.

    Args:
        result: TranscriptionResult to format.
        output_format: Output format ("txt" or "srt").

    Returns:
        Formatted transcript string.

    Raises:
        ValueError: If format is not supported or SRT has no segments.
    """
    if output_format == "txt":
        return format_as_txt(result)
    elif output_format == "srt":
        return format_as_srt(result)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def save_formatted_transcript(
    result: TranscriptionResult,
    output_path: Path,
    output_format: Literal["txt", "srt"] = "txt",
) -> Path:
    """Format and save transcription result to file.

    Args:
        result: TranscriptionResult to save.
        output_path: Path for output file.
        output_format: Output format ("txt" or "srt").

    Returns:
        Path to saved file.
    """
    content = format_transcript(result, output_format)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


def get_output_extension(output_format: Literal["txt", "srt"]) -> str:
    """Get file extension for output format.

    Args:
        output_format: Output format.

    Returns:
        File extension including dot (e.g., ".txt", ".srt").
    """
    return f".{output_format}"
