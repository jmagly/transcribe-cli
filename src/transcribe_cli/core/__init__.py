"""Core processing modules for transcribe-cli."""

from .extractor import (
    AUDIO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ExtractionError,
    ExtractionResult,
    MediaInfo,
    NoAudioStreamError,
    UnsupportedFormatError,
    extract_audio,
    get_media_info,
    is_audio_file,
    is_supported_file,
    is_video_file,
)
from .ffmpeg import (
    FFmpegInfo,
    FFmpegNotFoundError,
    FFmpegVersionError,
    check_ffmpeg_available,
    validate_ffmpeg,
)

__all__ = [
    # FFmpeg
    "FFmpegInfo",
    "FFmpegNotFoundError",
    "FFmpegVersionError",
    "validate_ffmpeg",
    "check_ffmpeg_available",
    # Extractor
    "ExtractionError",
    "ExtractionResult",
    "MediaInfo",
    "NoAudioStreamError",
    "UnsupportedFormatError",
    "extract_audio",
    "get_media_info",
    "is_audio_file",
    "is_video_file",
    "is_supported_file",
    # Constants
    "VIDEO_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
]
