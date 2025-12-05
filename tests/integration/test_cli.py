"""Integration tests for CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from transcribe_cli.cli.main import app

runner = CliRunner()


class TestCLI:
    """Tests for CLI commands."""

    def test_version_flag(self) -> None:
        """--version should display version and exit."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "transcribe-cli version" in result.stdout

    def test_help_flag(self) -> None:
        """--help should display help text."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Transcribe audio and video files" in result.stdout

    def test_transcribe_help(self) -> None:
        """transcribe --help should show command help."""
        result = runner.invoke(app, ["transcribe", "--help"])
        assert result.exit_code == 0
        assert "Transcribe a single audio or video file" in result.stdout

    def test_extract_help(self) -> None:
        """extract --help should show command help."""
        result = runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0
        assert "Extract audio from a video file" in result.stdout

    def test_batch_help(self) -> None:
        """batch --help should show command help."""
        result = runner.invoke(app, ["batch", "--help"])
        assert result.exit_code == 0
        assert "Batch transcribe all audio/video files" in result.stdout

    def test_config_show(self) -> None:
        """config --show should display configuration."""
        result = runner.invoke(app, ["config", "--show"])
        assert result.exit_code == 0
        assert "Current Configuration" in result.stdout

    def test_transcribe_nonexistent_file(self) -> None:
        """transcribe should error on non-existent file."""
        result = runner.invoke(app, ["transcribe", "nonexistent.mp3"])
        assert result.exit_code != 0

    def test_batch_invalid_concurrency(self) -> None:
        """batch should reject invalid concurrency values."""
        result = runner.invoke(app, ["batch", ".", "--concurrency", "0"])
        assert result.exit_code != 0

    def test_batch_concurrency_max(self) -> None:
        """batch should reject concurrency > 20."""
        result = runner.invoke(app, ["batch", ".", "--concurrency", "25"])
        assert result.exit_code != 0


class TestExtractCommand:
    """Tests for extract command."""

    def test_extract_nonexistent_file(self) -> None:
        """extract should error on non-existent file."""
        result = runner.invoke(app, ["extract", "nonexistent.mp4"])
        assert result.exit_code != 0

    def test_extract_invalid_format(self, tmp_path: Path) -> None:
        """extract should reject invalid output formats."""
        # Create a fake video file
        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake video content")

        result = runner.invoke(app, ["extract", str(fake_video), "--format", "ogg"])
        assert result.exit_code == 1
        assert "Unsupported format" in result.stdout

    def test_extract_unsupported_input_format(self, tmp_path: Path) -> None:
        """extract should reject unsupported input formats."""
        # Create an unsupported file
        pdf_file = tmp_path / "document.pdf"
        pdf_file.write_bytes(b"fake pdf content")

        result = runner.invoke(app, ["extract", str(pdf_file)])
        assert result.exit_code == 1
        assert "Unsupported" in result.stdout or "Error" in result.stdout

    def test_extract_help_shows_format_option(self) -> None:
        """extract --help should show format option."""
        result = runner.invoke(app, ["extract", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.stdout
        assert "mp3" in result.stdout
        assert "wav" in result.stdout

    def test_extract_with_mock_ffmpeg_success(self, tmp_path: Path) -> None:
        """extract should succeed with mocked FFmpeg."""
        # Create fake video file
        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake video content")

        # Create expected output file
        output_file = tmp_path / "video.mp3"

        # Mock the extraction
        mock_result = MagicMock()
        mock_result.input_path = fake_video
        mock_result.output_path = output_file
        mock_result.duration = 60.0
        mock_result.audio_codec = "mp3"
        mock_result.file_size = 1024 * 1024
        mock_result.file_size_display = "1.0 MB"

        with patch("transcribe_cli.core.extract_audio", return_value=mock_result):
            result = runner.invoke(app, ["extract", str(fake_video)])
            assert result.exit_code == 0
            assert "Success" in result.stdout

    def test_extract_ffmpeg_not_found(self, tmp_path: Path) -> None:
        """extract should show helpful error when FFmpeg not found."""
        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake video content")

        from transcribe_cli.core.ffmpeg import FFmpegNotFoundError

        with patch(
            "transcribe_cli.core.extract_audio",
            side_effect=FFmpegNotFoundError("FFmpeg not found"),
        ):
            result = runner.invoke(app, ["extract", str(fake_video)])
            assert result.exit_code == 1
            assert "Error" in result.stdout

    def test_extract_no_audio_stream(self, tmp_path: Path) -> None:
        """extract should error when file has no audio."""
        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake video content")

        from transcribe_cli.core.extractor import NoAudioStreamError

        with patch(
            "transcribe_cli.core.extract_audio",
            side_effect=NoAudioStreamError(fake_video),
        ):
            result = runner.invoke(app, ["extract", str(fake_video)])
            assert result.exit_code == 1
            assert "No audio" in result.stdout or "Error" in result.stdout
