"""Unit tests for audio extractor module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_cli.core.extractor import (
    AUDIO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ExtractionError,
    ExtractionResult,
    MediaInfo,
    NoAudioStreamError,
    UnsupportedFormatError,
    get_media_info,
    is_audio_file,
    is_supported_file,
    is_video_file,
    validate_input_file,
)


class TestFileTypeChecks:
    """Tests for file type detection functions."""

    def test_is_video_file_mkv(self) -> None:
        """MKV detected as video."""
        assert is_video_file(Path("video.mkv")) is True

    def test_is_video_file_mp4(self) -> None:
        """MP4 detected as video."""
        assert is_video_file(Path("video.mp4")) is True

    def test_is_video_file_avi(self) -> None:
        """AVI detected as video."""
        assert is_video_file(Path("video.avi")) is True

    def test_is_video_file_mov(self) -> None:
        """MOV detected as video."""
        assert is_video_file(Path("video.mov")) is True

    def test_is_video_file_audio_returns_false(self) -> None:
        """Audio file not detected as video."""
        assert is_video_file(Path("audio.mp3")) is False

    def test_is_audio_file_mp3(self) -> None:
        """MP3 detected as audio."""
        assert is_audio_file(Path("audio.mp3")) is True

    def test_is_audio_file_wav(self) -> None:
        """WAV detected as audio."""
        assert is_audio_file(Path("audio.wav")) is True

    def test_is_audio_file_flac(self) -> None:
        """FLAC detected as audio."""
        assert is_audio_file(Path("audio.flac")) is True

    def test_is_audio_file_video_returns_false(self) -> None:
        """Video file not detected as audio."""
        assert is_audio_file(Path("video.mkv")) is False

    def test_is_supported_file_video(self) -> None:
        """Video files are supported."""
        assert is_supported_file(Path("video.mkv")) is True

    def test_is_supported_file_audio(self) -> None:
        """Audio files are supported."""
        assert is_supported_file(Path("audio.mp3")) is True

    def test_is_supported_file_unsupported(self) -> None:
        """Unsupported files return False."""
        assert is_supported_file(Path("document.pdf")) is False

    def test_case_insensitive_extension(self) -> None:
        """Extension check is case insensitive."""
        assert is_video_file(Path("video.MKV")) is True
        assert is_audio_file(Path("audio.MP3")) is True


class TestExtensionSets:
    """Tests for extension set constants."""

    def test_video_extensions_include_common_formats(self) -> None:
        """Video extensions include MKV, MP4, AVI, MOV."""
        for ext in [".mkv", ".mp4", ".avi", ".mov"]:
            assert ext in VIDEO_EXTENSIONS

    def test_audio_extensions_include_common_formats(self) -> None:
        """Audio extensions include MP3, WAV, FLAC, AAC."""
        for ext in [".mp3", ".wav", ".flac", ".aac"]:
            assert ext in AUDIO_EXTENSIONS

    def test_supported_is_union_of_video_and_audio(self) -> None:
        """SUPPORTED_EXTENSIONS is union of VIDEO and AUDIO."""
        assert SUPPORTED_EXTENSIONS == VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


class TestValidateInputFile:
    """Tests for input file validation."""

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.mp4"
        with pytest.raises(FileNotFoundError):
            validate_input_file(nonexistent)

    def test_directory_raises_value_error(self, tmp_path: Path) -> None:
        """Directory instead of file raises ValueError."""
        with pytest.raises(ValueError, match="not a file"):
            validate_input_file(tmp_path)

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        """Unsupported format raises UnsupportedFormatError."""
        pdf_file = tmp_path / "document.pdf"
        pdf_file.write_text("fake pdf")
        with pytest.raises(UnsupportedFormatError) as exc_info:
            validate_input_file(pdf_file)
        assert ".pdf" in str(exc_info.value)

    def test_valid_file_passes(self, tmp_path: Path) -> None:
        """Valid supported file passes validation."""
        mp4_file = tmp_path / "video.mp4"
        mp4_file.write_bytes(b"fake video")
        validate_input_file(mp4_file)  # Should not raise


class TestMediaInfo:
    """Tests for MediaInfo dataclass."""

    def test_is_video_with_video_stream(self) -> None:
        """is_video returns True when has_video is True."""
        info = MediaInfo(
            path=Path("video.mp4"),
            format_name="mp4",
            duration=120.0,
            has_video=True,
            has_audio=True,
            audio_codec="aac",
            audio_channels=2,
            audio_sample_rate=44100,
        )
        assert info.is_video is True

    def test_is_audio_only_without_video(self) -> None:
        """is_audio_only returns True for audio-only files."""
        info = MediaInfo(
            path=Path("audio.mp3"),
            format_name="mp3",
            duration=180.0,
            has_video=False,
            has_audio=True,
            audio_codec="mp3",
            audio_channels=2,
            audio_sample_rate=44100,
        )
        assert info.is_audio_only is True
        assert info.is_video is False

    def test_duration_display_hours(self) -> None:
        """duration_display formats hours correctly."""
        info = MediaInfo(
            path=Path("long.mp4"),
            format_name="mp4",
            duration=3723.0,  # 1:02:03
            has_video=True,
            has_audio=True,
            audio_codec="aac",
            audio_channels=2,
            audio_sample_rate=44100,
        )
        assert info.duration_display == "1:02:03"

    def test_duration_display_minutes(self) -> None:
        """duration_display formats minutes correctly."""
        info = MediaInfo(
            path=Path("short.mp4"),
            format_name="mp4",
            duration=125.0,  # 2:05
            has_video=True,
            has_audio=True,
            audio_codec="aac",
            audio_channels=2,
            audio_sample_rate=44100,
        )
        assert info.duration_display == "2:05"

    def test_duration_display_unknown(self) -> None:
        """duration_display handles None."""
        info = MediaInfo(
            path=Path("unknown.mp4"),
            format_name="mp4",
            duration=None,
            has_video=True,
            has_audio=True,
            audio_codec="aac",
            audio_channels=2,
            audio_sample_rate=44100,
        )
        assert info.duration_display == "unknown"


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_file_size_display_bytes(self) -> None:
        """file_size_display formats bytes correctly."""
        result = ExtractionResult(
            input_path=Path("input.mp4"),
            output_path=Path("output.mp3"),
            duration=60.0,
            audio_codec="mp3",
            file_size=500,
        )
        assert result.file_size_display == "500 B"

    def test_file_size_display_kilobytes(self) -> None:
        """file_size_display formats KB correctly."""
        result = ExtractionResult(
            input_path=Path("input.mp4"),
            output_path=Path("output.mp3"),
            duration=60.0,
            audio_codec="mp3",
            file_size=5 * 1024,
        )
        assert result.file_size_display == "5.0 KB"

    def test_file_size_display_megabytes(self) -> None:
        """file_size_display formats MB correctly."""
        result = ExtractionResult(
            input_path=Path("input.mp4"),
            output_path=Path("output.mp3"),
            duration=60.0,
            audio_codec="mp3",
            file_size=5 * 1024 * 1024,
        )
        assert result.file_size_display == "5.0 MB"

    def test_file_size_display_gigabytes(self) -> None:
        """file_size_display formats GB correctly."""
        result = ExtractionResult(
            input_path=Path("input.mp4"),
            output_path=Path("output.mp3"),
            duration=60.0,
            audio_codec="mp3",
            file_size=2 * 1024 * 1024 * 1024,
        )
        assert result.file_size_display == "2.00 GB"


class TestUnsupportedFormatError:
    """Tests for UnsupportedFormatError."""

    def test_error_message_includes_format(self) -> None:
        """Error message includes the unsupported format."""
        error = UnsupportedFormatError(Path("file.xyz"), ".xyz")
        assert ".xyz" in str(error)

    def test_error_message_includes_supported_formats(self) -> None:
        """Error message lists supported formats."""
        error = UnsupportedFormatError(Path("file.xyz"), ".xyz")
        message = str(error)
        assert ".mp3" in message
        assert ".mkv" in message


class TestNoAudioStreamError:
    """Tests for NoAudioStreamError."""

    def test_error_message_includes_path(self) -> None:
        """Error message includes file path."""
        path = Path("/some/video.mp4")
        error = NoAudioStreamError(path)
        assert str(path) in str(error)


class TestGetMediaInfo:
    """Tests for get_media_info function."""

    def test_get_media_info_success(self) -> None:
        """Successfully parse ffprobe output."""
        ffprobe_output = {
            "format": {
                "format_name": "matroska,webm",
                "duration": "120.500",
            },
            "streams": [
                {"codec_type": "video", "codec_name": "h264"},
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "sample_rate": "44100",
                },
            ],
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(ffprobe_output)

        with patch("transcribe_cli.core.extractor.find_ffprobe", return_value="/usr/bin/ffprobe"):
            with patch("subprocess.run", return_value=mock_result):
                info = get_media_info(Path("video.mkv"))
                assert info.has_video is True
                assert info.has_audio is True
                assert info.audio_codec == "aac"
                assert info.duration == 120.5

    def test_get_media_info_no_ffprobe(self) -> None:
        """Missing ffprobe raises FFmpegNotFoundError."""
        from transcribe_cli.core.ffmpeg import FFmpegNotFoundError

        with patch("transcribe_cli.core.extractor.find_ffprobe", return_value=None):
            with pytest.raises(FFmpegNotFoundError):
                get_media_info(Path("video.mkv"))

    def test_get_media_info_audio_only(self) -> None:
        """Audio-only file detected correctly."""
        ffprobe_output = {
            "format": {
                "format_name": "mp3",
                "duration": "180.0",
            },
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "mp3",
                    "channels": 2,
                    "sample_rate": "44100",
                },
            ],
        }

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(ffprobe_output)

        with patch("transcribe_cli.core.extractor.find_ffprobe", return_value="/usr/bin/ffprobe"):
            with patch("subprocess.run", return_value=mock_result):
                info = get_media_info(Path("audio.mp3"))
                assert info.has_video is False
                assert info.has_audio is True
                assert info.is_audio_only is True
