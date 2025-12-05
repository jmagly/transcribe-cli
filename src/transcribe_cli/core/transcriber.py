"""Transcription client for OpenAI Whisper API.

Implements Sprint 3: Transcription Client
- OpenAI Whisper API integration
- Retry logic with exponential backoff
- Response parsing with timestamps
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .extractor import extract_audio, is_video_file
from .ffmpeg import FFmpegNotFoundError


class TranscriptionError(Exception):
    """Raised when transcription fails."""

    pass


class APIKeyMissingError(Exception):
    """Raised when OpenAI API key is not configured."""

    def __init__(self) -> None:
        message = (
            "OpenAI API key is not configured.\n\n"
            "Set the OPENAI_API_KEY environment variable:\n"
            "  export OPENAI_API_KEY=sk-your-api-key-here\n\n"
            "Or create a .env file with:\n"
            "  OPENAI_API_KEY=sk-your-api-key-here\n\n"
            "Get your API key at: https://platform.openai.com/api-keys"
        )
        super().__init__(message)


class FileTooLargeError(Exception):
    """Raised when file exceeds Whisper API size limit."""

    def __init__(self, path: Path, size_mb: float, max_mb: float = 25.0) -> None:
        message = (
            f"File is too large for Whisper API: {size_mb:.1f}MB\n"
            f"Maximum allowed size: {max_mb}MB\n"
            f"File: {path}\n\n"
            "For large files, use chunking (available in future release)."
        )
        super().__init__(message)
        self.path = path
        self.size_mb = size_mb
        self.max_mb = max_mb


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing."""

    id: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        """Duration of segment in seconds."""
        return self.end - self.start


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""

    input_path: Path
    output_path: Optional[Path]
    text: str
    segments: list[TranscriptionSegment]
    language: str
    duration: Optional[float]

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.text.split())


# Maximum file size for Whisper API (25MB)
MAX_FILE_SIZE_MB = 25.0
MAX_FILE_SIZE_BYTES = int(MAX_FILE_SIZE_MB * 1024 * 1024)


def _check_file_size(path: Path) -> None:
    """Check if file is within Whisper API limits.

    Args:
        path: Path to audio file.

    Raises:
        FileTooLargeError: If file exceeds 25MB limit.
    """
    size = path.stat().st_size
    size_mb = size / (1024 * 1024)
    if size > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(path, size_mb, MAX_FILE_SIZE_MB)


def _create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create OpenAI client.

    Args:
        api_key: Optional API key. If not provided, uses OPENAI_API_KEY env var.

    Returns:
        Configured OpenAI client.

    Raises:
        APIKeyMissingError: If no API key is available.
    """
    try:
        client = OpenAI(api_key=api_key)
        # Validate key is present (OpenAI client doesn't validate until first call)
        if not client.api_key:
            raise APIKeyMissingError()
        return client
    except Exception as e:
        if "api_key" in str(e).lower():
            raise APIKeyMissingError() from e
        raise


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _transcribe_audio_file(
    client: OpenAI,
    audio_path: Path,
    language: Optional[str] = None,
    response_format: Literal["json", "text", "verbose_json"] = "verbose_json",
) -> dict:
    """Call Whisper API to transcribe audio file.

    Args:
        client: OpenAI client.
        audio_path: Path to audio file.
        language: Optional language code (e.g., "en", "es").
        response_format: API response format.

    Returns:
        API response as dictionary.

    Raises:
        RateLimitError: On rate limit (will be retried).
        APIConnectionError: On connection issues (will be retried).
        APIStatusError: On other API errors.
    """
    with open(audio_path, "rb") as audio_file:
        kwargs = {
            "model": "whisper-1",
            "file": audio_file,
            "response_format": response_format,
        }
        if language and language != "auto":
            kwargs["language"] = language

        response = client.audio.transcriptions.create(**kwargs)

    # Handle different response formats
    if response_format == "text":
        return {"text": response}
    elif hasattr(response, "model_dump"):
        return response.model_dump()
    else:
        return dict(response)


def _parse_segments(response: dict) -> list[TranscriptionSegment]:
    """Parse segments from Whisper API response.

    Args:
        response: API response dictionary.

    Returns:
        List of TranscriptionSegment objects.
    """
    segments = []
    raw_segments = response.get("segments", [])

    for i, seg in enumerate(raw_segments):
        segments.append(
            TranscriptionSegment(
                id=seg.get("id", i),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
            )
        )

    return segments


def transcribe_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    language: str = "auto",
    api_key: Optional[str] = None,
) -> TranscriptionResult:
    """Transcribe an audio or video file.

    For video files, audio is automatically extracted first.

    Args:
        input_path: Path to audio or video file.
        output_path: Optional path for output text file.
        language: Language code or "auto" for detection.
        api_key: Optional OpenAI API key.

    Returns:
        TranscriptionResult with transcribed text and metadata.

    Raises:
        APIKeyMissingError: If API key not configured.
        FileTooLargeError: If file exceeds 25MB.
        FFmpegNotFoundError: If FFmpeg needed but not installed.
        TranscriptionError: If transcription fails.
    """
    input_path = Path(input_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    # Create client (validates API key)
    client = _create_client(api_key)

    # Handle video files - extract audio first
    audio_path = input_path
    temp_audio = None

    try:
        if is_video_file(input_path):
            # Extract audio to temporary file
            temp_dir = tempfile.mkdtemp(prefix="transcribe_")
            temp_audio = Path(temp_dir) / f"{input_path.stem}.mp3"
            extraction_result = extract_audio(
                input_path=input_path,
                output_path=temp_audio,
                output_format="mp3",
            )
            audio_path = extraction_result.output_path

        # Check file size
        _check_file_size(audio_path)

        # Call Whisper API
        try:
            response = _transcribe_audio_file(
                client=client,
                audio_path=audio_path,
                language=language if language != "auto" else None,
            )
        except RateLimitError as e:
            raise TranscriptionError(
                f"Rate limit exceeded after retries. Please wait and try again.\n{e}"
            ) from e
        except APIStatusError as e:
            raise TranscriptionError(f"API error: {e.message}") from e
        except APIConnectionError as e:
            raise TranscriptionError(
                f"Connection error after retries. Check your internet connection.\n{e}"
            ) from e

        # Parse response
        text = response.get("text", "")
        segments = _parse_segments(response)
        detected_language = response.get("language", "unknown")
        duration = response.get("duration")

        # Determine output path
        if output_path is None:
            output_path = input_path.with_suffix(".txt")

        return TranscriptionResult(
            input_path=input_path,
            output_path=output_path,
            text=text,
            segments=segments,
            language=detected_language,
            duration=duration,
        )

    finally:
        # Clean up temporary audio file
        if temp_audio and temp_audio.exists():
            temp_audio.unlink()
            temp_audio.parent.rmdir()


def save_transcript(result: TranscriptionResult, output_path: Optional[Path] = None) -> Path:
    """Save transcription result to a text file.

    Args:
        result: TranscriptionResult to save.
        output_path: Optional output path. Uses result.output_path if not provided.

    Returns:
        Path to saved file.
    """
    path = output_path or result.output_path
    if path is None:
        path = result.input_path.with_suffix(".txt")

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(result.text)

    return path
