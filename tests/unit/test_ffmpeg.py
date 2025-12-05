"""Unit tests for FFmpeg detection module."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from transcribe_cli.core.ffmpeg import (
    MINIMUM_FFMPEG_VERSION,
    FFmpegInfo,
    FFmpegNotFoundError,
    FFmpegVersionError,
    check_ffmpeg_available,
    find_ffmpeg,
    find_ffprobe,
    get_ffmpeg_version,
    parse_version,
    validate_ffmpeg,
)


class TestParseVersion:
    """Tests for version parsing."""

    def test_parse_standard_version(self) -> None:
        """Parse standard FFmpeg version string."""
        output = "ffmpeg version 5.1.2 Copyright (c) 2000-2023"
        assert parse_version(output) == (5, 1)

    def test_parse_git_version(self) -> None:
        """Parse git build version string."""
        output = "ffmpeg version n5.1 Copyright (c) 2000-2023"
        assert parse_version(output) == (5, 1)

    def test_parse_version_with_extra_info(self) -> None:
        """Parse version with Ubuntu suffix."""
        output = "ffmpeg version 4.4.2-0ubuntu0.22.04.1"
        assert parse_version(output) == (4, 4)

    def test_parse_version_6x(self) -> None:
        """Parse FFmpeg 6.x version."""
        output = "ffmpeg version 6.0 Copyright (c) 2000-2023"
        assert parse_version(output) == (6, 0)

    def test_parse_invalid_version_raises(self) -> None:
        """Invalid version string raises ValueError."""
        with pytest.raises(ValueError, match="Could not parse"):
            parse_version("invalid output")


class TestFindFFmpeg:
    """Tests for FFmpeg binary detection."""

    def test_find_ffmpeg_when_installed(self) -> None:
        """find_ffmpeg returns path when installed."""
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            assert find_ffmpeg() == "/usr/bin/ffmpeg"

    def test_find_ffmpeg_when_missing(self) -> None:
        """find_ffmpeg returns None when not installed."""
        with patch("shutil.which", return_value=None):
            assert find_ffmpeg() is None

    def test_find_ffprobe_when_installed(self) -> None:
        """find_ffprobe returns path when installed."""
        with patch("shutil.which", return_value="/usr/bin/ffprobe"):
            assert find_ffprobe() == "/usr/bin/ffprobe"

    def test_find_ffprobe_when_missing(self) -> None:
        """find_ffprobe returns None when not installed."""
        with patch("shutil.which", return_value=None):
            assert find_ffprobe() is None


class TestGetFFmpegVersion:
    """Tests for version retrieval."""

    def test_get_version_success(self) -> None:
        """Successfully get FFmpeg version."""
        mock_result = MagicMock()
        mock_result.stdout = "ffmpeg version 5.1.2 Copyright (c) 2000-2023\nbuilt with gcc"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            version, version_string = get_ffmpeg_version("/usr/bin/ffmpeg")
            assert version == (5, 1)
            assert "ffmpeg version 5.1.2" in version_string

    def test_get_version_timeout(self) -> None:
        """Version check timeout raises RuntimeError."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 10)):
            with pytest.raises(RuntimeError, match="timed out"):
                get_ffmpeg_version("/usr/bin/ffmpeg")


class TestValidateFFmpeg:
    """Tests for FFmpeg validation."""

    def test_validate_ffmpeg_success(self) -> None:
        """Successful validation returns FFmpegInfo."""
        with patch("transcribe_cli.core.ffmpeg.find_ffmpeg", return_value="/usr/bin/ffmpeg"):
            with patch(
                "transcribe_cli.core.ffmpeg.get_ffmpeg_version",
                return_value=((5, 1), "ffmpeg version 5.1.2"),
            ):
                info = validate_ffmpeg()
                assert isinstance(info, FFmpegInfo)
                assert info.path == "/usr/bin/ffmpeg"
                assert info.version == (5, 1)

    def test_validate_ffmpeg_not_found(self) -> None:
        """Missing FFmpeg raises FFmpegNotFoundError."""
        with patch("transcribe_cli.core.ffmpeg.find_ffmpeg", return_value=None):
            with pytest.raises(FFmpegNotFoundError):
                validate_ffmpeg()

    def test_validate_ffmpeg_version_too_old(self) -> None:
        """Old FFmpeg version raises FFmpegVersionError."""
        with patch("transcribe_cli.core.ffmpeg.find_ffmpeg", return_value="/usr/bin/ffmpeg"):
            with patch(
                "transcribe_cli.core.ffmpeg.get_ffmpeg_version",
                return_value=((3, 4), "ffmpeg version 3.4.1"),
            ):
                with pytest.raises(FFmpegVersionError) as exc_info:
                    validate_ffmpeg()
                assert exc_info.value.found_version == (3, 4)
                assert exc_info.value.required_version == MINIMUM_FFMPEG_VERSION


class TestCheckFFmpegAvailable:
    """Tests for non-throwing FFmpeg check."""

    def test_check_available_when_installed(self) -> None:
        """Returns True when FFmpeg is properly installed."""
        with patch("transcribe_cli.core.ffmpeg.validate_ffmpeg") as mock:
            mock.return_value = FFmpegInfo("/usr/bin/ffmpeg", (5, 1), "5.1.2")
            assert check_ffmpeg_available() is True

    def test_check_available_when_missing(self) -> None:
        """Returns False when FFmpeg is not installed."""
        with patch("transcribe_cli.core.ffmpeg.validate_ffmpeg") as mock:
            mock.side_effect = FFmpegNotFoundError()
            assert check_ffmpeg_available() is False

    def test_check_available_when_old_version(self) -> None:
        """Returns False when FFmpeg version is too old."""
        with patch("transcribe_cli.core.ffmpeg.validate_ffmpeg") as mock:
            mock.side_effect = FFmpegVersionError((3, 4), (4, 0))
            assert check_ffmpeg_available() is False


class TestFFmpegNotFoundError:
    """Tests for FFmpegNotFoundError message generation."""

    def test_error_message_linux(self) -> None:
        """Linux installation instructions included."""
        with patch("sys.platform", "linux"):
            error = FFmpegNotFoundError()
            message = str(error)
            assert "apt" in message or "dnf" in message

    def test_error_message_macos(self) -> None:
        """macOS installation instructions included."""
        with patch("sys.platform", "darwin"):
            error = FFmpegNotFoundError()
            message = str(error)
            assert "brew" in message.lower() or "homebrew" in message.lower()

    def test_error_message_windows(self) -> None:
        """Windows installation instructions included."""
        with patch("sys.platform", "win32"):
            error = FFmpegNotFoundError()
            message = str(error)
            assert "choco" in message.lower() or "chocolatey" in message.lower()


class TestFFmpegInfo:
    """Tests for FFmpegInfo dataclass."""

    def test_version_display(self) -> None:
        """version_display formats correctly."""
        info = FFmpegInfo("/usr/bin/ffmpeg", (5, 1), "ffmpeg version 5.1.2")
        assert info.version_display == "5.1"
