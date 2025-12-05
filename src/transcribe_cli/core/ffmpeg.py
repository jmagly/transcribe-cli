"""FFmpeg detection and validation.

Implements ADR-001: FFmpeg Integration Approach
- FFmpeg binary detection
- Version validation (4.0+)
- Platform-specific installation guidance
"""

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

MINIMUM_FFMPEG_VERSION = (4, 0)


class FFmpegNotFoundError(Exception):
    """Raised when FFmpeg is not installed or not in PATH."""

    def __init__(self, message: Optional[str] = None) -> None:
        if message is None:
            message = self._build_message()
        super().__init__(message)

    @staticmethod
    def _build_message() -> str:
        """Build platform-specific installation guidance."""
        base = "FFmpeg is not installed or not in PATH.\n\n"

        if sys.platform.startswith("linux"):
            instructions = (
                "Install FFmpeg on Linux:\n"
                "  Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg -y\n"
                "  Fedora: sudo dnf install ffmpeg -y\n"
                "  Arch: sudo pacman -S ffmpeg"
            )
        elif sys.platform == "darwin":
            instructions = (
                "Install FFmpeg on macOS:\n"
                "  Homebrew: brew install ffmpeg\n"
                "  MacPorts: sudo port install ffmpeg"
            )
        elif sys.platform == "win32":
            instructions = (
                "Install FFmpeg on Windows:\n"
                "  Chocolatey: choco install ffmpeg -y\n"
                "  Scoop: scoop install ffmpeg\n"
                "  Or download from: https://ffmpeg.org/download.html"
            )
        else:
            instructions = "Download FFmpeg from: https://ffmpeg.org/download.html"

        return f"{base}{instructions}"


class FFmpegVersionError(Exception):
    """Raised when FFmpeg version is too old."""

    def __init__(
        self, found_version: tuple[int, int], required: tuple[int, int]
    ) -> None:
        found_str = f"{found_version[0]}.{found_version[1]}"
        required_str = f"{required[0]}.{required[1]}"
        message = (
            f"FFmpeg version {found_str} is too old. "
            f"Minimum required version is {required_str}.\n\n"
            "Please update FFmpeg to the latest version."
        )
        super().__init__(message)
        self.found_version = found_version
        self.required_version = required


@dataclass
class FFmpegInfo:
    """Information about the FFmpeg installation."""

    path: str
    version: tuple[int, int]
    version_string: str

    @property
    def version_display(self) -> str:
        """Return version as display string."""
        return f"{self.version[0]}.{self.version[1]}"


def find_ffmpeg() -> Optional[str]:
    """Find FFmpeg binary in PATH.

    Returns:
        Path to FFmpeg binary, or None if not found.
    """
    return shutil.which("ffmpeg")


def find_ffprobe() -> Optional[str]:
    """Find ffprobe binary in PATH.

    Returns:
        Path to ffprobe binary, or None if not found.
    """
    return shutil.which("ffprobe")


def parse_version(version_output: str) -> tuple[int, int]:
    """Parse FFmpeg version from version output.

    Args:
        version_output: Output from `ffmpeg -version`.

    Returns:
        Tuple of (major, minor) version numbers.

    Raises:
        ValueError: If version cannot be parsed.
    """
    # Match patterns like "ffmpeg version 5.1.2" or "ffmpeg version n5.1"
    patterns = [
        r"ffmpeg version (\d+)\.(\d+)",  # Standard: 5.1.2
        r"ffmpeg version n(\d+)\.(\d+)",  # Git builds: n5.1
        r"ffmpeg version N-(\d+)",  # Development builds
    ]

    for pattern in patterns:
        match = re.search(pattern, version_output, re.IGNORECASE)
        if match:
            return (int(match.group(1)), int(match.group(2)))

    raise ValueError(f"Could not parse FFmpeg version from: {version_output[:100]}")


def get_ffmpeg_version(ffmpeg_path: str) -> tuple[tuple[int, int], str]:
    """Get FFmpeg version information.

    Args:
        ffmpeg_path: Path to FFmpeg binary.

    Returns:
        Tuple of ((major, minor), full_version_string).

    Raises:
        RuntimeError: If version cannot be determined.
    """
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout or result.stderr
        version = parse_version(output)
        # Extract first line for display
        version_string = output.split("\n")[0] if output else "unknown"
        return version, version_string
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("FFmpeg version check timed out") from e
    except Exception as e:
        raise RuntimeError(f"Failed to get FFmpeg version: {e}") from e


def validate_ffmpeg(
    min_version: tuple[int, int] = MINIMUM_FFMPEG_VERSION,
) -> FFmpegInfo:
    """Validate FFmpeg installation.

    Args:
        min_version: Minimum required version (major, minor).

    Returns:
        FFmpegInfo with details about the installation.

    Raises:
        FFmpegNotFoundError: If FFmpeg is not installed.
        FFmpegVersionError: If FFmpeg version is too old.
    """
    path = find_ffmpeg()
    if path is None:
        raise FFmpegNotFoundError()

    version, version_string = get_ffmpeg_version(path)

    if version < min_version:
        raise FFmpegVersionError(version, min_version)

    return FFmpegInfo(path=path, version=version, version_string=version_string)


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is available (non-throwing).

    Returns:
        True if FFmpeg is installed and meets version requirements.
    """
    try:
        validate_ffmpeg()
        return True
    except (FFmpegNotFoundError, FFmpegVersionError):
        return False
