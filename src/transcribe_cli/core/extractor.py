"""Audio extraction from video files.

Implements ADR-001: FFmpeg Integration Approach
- Uses ffmpeg-python library for audio extraction
- Supports MKV, MP4, AVI, MOV containers
- Extracts to MP3 format (optimal for Whisper API)
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import ffmpeg

from .ffmpeg import FFmpegNotFoundError, find_ffprobe, validate_ffmpeg

# Supported input formats
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".avi", ".mov", ".webm", ".wmv", ".flv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".wma"}
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


class ExtractionError(Exception):
    """Raised when audio extraction fails."""

    pass


class UnsupportedFormatError(Exception):
    """Raised when file format is not supported."""

    def __init__(self, path: Path, extension: str) -> None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        message = (
            f"Unsupported file format: {extension}\n"
            f"File: {path}\n"
            f"Supported formats: {supported}"
        )
        super().__init__(message)
        self.path = path
        self.extension = extension


class NoAudioStreamError(Exception):
    """Raised when file has no audio stream."""

    def __init__(self, path: Path) -> None:
        message = f"No audio stream found in file: {path}"
        super().__init__(message)
        self.path = path


@dataclass
class MediaInfo:
    """Information about a media file."""

    path: Path
    format_name: str
    duration: Optional[float]
    has_video: bool
    has_audio: bool
    audio_codec: Optional[str]
    audio_channels: Optional[int]
    audio_sample_rate: Optional[int]

    @property
    def is_video(self) -> bool:
        """Check if file is a video container."""
        return self.has_video

    @property
    def is_audio_only(self) -> bool:
        """Check if file is audio-only."""
        return self.has_audio and not self.has_video

    @property
    def duration_display(self) -> str:
        """Return duration as human-readable string."""
        if self.duration is None:
            return "unknown"
        hours, remainder = divmod(int(self.duration), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


@dataclass
class ExtractionResult:
    """Result of audio extraction."""

    input_path: Path
    output_path: Path
    duration: Optional[float]
    audio_codec: str
    file_size: int

    @property
    def file_size_display(self) -> str:
        """Return file size as human-readable string."""
        if self.file_size < 1024:
            return f"{self.file_size} B"
        elif self.file_size < 1024 * 1024:
            return f"{self.file_size / 1024:.1f} KB"
        elif self.file_size < 1024 * 1024 * 1024:
            return f"{self.file_size / (1024 * 1024):.1f} MB"
        return f"{self.file_size / (1024 * 1024 * 1024):.2f} GB"


def get_media_info(path: Path) -> MediaInfo:
    """Get information about a media file using ffprobe.

    Args:
        path: Path to media file.

    Returns:
        MediaInfo with file details.

    Raises:
        FFmpegNotFoundError: If ffprobe is not available.
        ExtractionError: If file cannot be probed.
    """
    ffprobe_path = find_ffprobe()
    if ffprobe_path is None:
        raise FFmpegNotFoundError("ffprobe is not installed or not in PATH.")

    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise ExtractionError(f"ffprobe failed: {result.stderr}")

        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Failed to parse ffprobe output: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise ExtractionError(f"ffprobe timed out for {path}") from e

    # Parse streams
    streams = data.get("streams", [])
    format_info = data.get("format", {})

    has_video = any(s.get("codec_type") == "video" for s in streams)
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    has_audio = len(audio_streams) > 0

    # Get first audio stream info
    audio_codec = None
    audio_channels = None
    audio_sample_rate = None
    if audio_streams:
        first_audio = audio_streams[0]
        audio_codec = first_audio.get("codec_name")
        audio_channels = first_audio.get("channels")
        sample_rate = first_audio.get("sample_rate")
        if sample_rate:
            audio_sample_rate = int(sample_rate)

    # Parse duration
    duration = None
    duration_str = format_info.get("duration")
    if duration_str:
        try:
            duration = float(duration_str)
        except ValueError:
            pass

    return MediaInfo(
        path=path,
        format_name=format_info.get("format_name", "unknown"),
        duration=duration,
        has_video=has_video,
        has_audio=has_audio,
        audio_codec=audio_codec,
        audio_channels=audio_channels,
        audio_sample_rate=audio_sample_rate,
    )


def validate_input_file(path: Path) -> None:
    """Validate that input file exists and is supported.

    Args:
        path: Path to input file.

    Raises:
        FileNotFoundError: If file does not exist.
        UnsupportedFormatError: If format is not supported.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(path, extension)


def extract_audio(
    input_path: Path,
    output_path: Optional[Path] = None,
    output_format: Literal["mp3", "wav"] = "mp3",
    audio_bitrate: str = "192k",
    overwrite: bool = True,
) -> ExtractionResult:
    """Extract audio from a video or audio file.

    Args:
        input_path: Path to input media file.
        output_path: Path for output audio file. If None, uses input name with new extension.
        output_format: Output audio format (mp3 or wav).
        audio_bitrate: Audio bitrate for MP3 (e.g., "192k", "320k").
        overwrite: Whether to overwrite existing output file.

    Returns:
        ExtractionResult with details about the extracted audio.

    Raises:
        FFmpegNotFoundError: If FFmpeg is not installed.
        FileNotFoundError: If input file does not exist.
        UnsupportedFormatError: If input format is not supported.
        NoAudioStreamError: If input has no audio stream.
        ExtractionError: If extraction fails.
    """
    # Validate FFmpeg first
    validate_ffmpeg()

    # Validate input
    input_path = Path(input_path).resolve()
    validate_input_file(input_path)

    # Get media info
    media_info = get_media_info(input_path)
    if not media_info.has_audio:
        raise NoAudioStreamError(input_path)

    # Determine output path
    if output_path is None:
        output_path = input_path.with_suffix(f".{output_format}")
    else:
        output_path = Path(output_path).resolve()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    try:
        stream = ffmpeg.input(str(input_path))

        # Configure output based on format
        if output_format == "mp3":
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec="libmp3lame",
                audio_bitrate=audio_bitrate,
                vn=None,  # No video
            )
        else:  # wav
            stream = ffmpeg.output(
                stream,
                str(output_path),
                acodec="pcm_s16le",
                ar=16000,  # 16kHz sample rate (good for speech)
                ac=1,  # Mono
                vn=None,
            )

        if overwrite:
            stream = ffmpeg.overwrite_output(stream)

        # Run extraction
        ffmpeg.run(stream, quiet=True, capture_stderr=True)

    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if e.stderr else "Unknown error"
        raise ExtractionError(f"FFmpeg extraction failed: {stderr}") from e

    # Verify output was created
    if not output_path.exists():
        raise ExtractionError(f"Output file was not created: {output_path}")

    return ExtractionResult(
        input_path=input_path,
        output_path=output_path,
        duration=media_info.duration,
        audio_codec=output_format,
        file_size=output_path.stat().st_size,
    )


def is_audio_file(path: Path) -> bool:
    """Check if file is an audio-only file (no video extraction needed).

    Args:
        path: Path to file.

    Returns:
        True if file is audio-only format.
    """
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_video_file(path: Path) -> bool:
    """Check if file is a video file (needs audio extraction).

    Args:
        path: Path to file.

    Returns:
        True if file is a video format.
    """
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_supported_file(path: Path) -> bool:
    """Check if file format is supported.

    Args:
        path: Path to file.

    Returns:
        True if file format is supported.
    """
    return path.suffix.lower() in SUPPORTED_EXTENSIONS
