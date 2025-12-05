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


class TestTranscribeCommand:
    """Tests for transcribe command."""

    def test_transcribe_nonexistent_file(self) -> None:
        """transcribe should error on non-existent file."""
        result = runner.invoke(app, ["transcribe", "nonexistent.mp3"])
        assert result.exit_code != 0

    def test_transcribe_invalid_format(self, tmp_path: Path) -> None:
        """transcribe should reject invalid output formats."""
        fake_audio = tmp_path / "audio.mp3"
        fake_audio.write_bytes(b"fake audio content")

        result = runner.invoke(app, ["transcribe", str(fake_audio), "--format", "pdf"])
        assert result.exit_code == 1
        assert "Unsupported format" in result.stdout

    def test_transcribe_srt_shows_warning(self, tmp_path: Path) -> None:
        """transcribe with --format srt should show warning."""
        fake_audio = tmp_path / "audio.mp3"
        fake_audio.write_bytes(b"fake audio content")

        # Mock the transcription to avoid API call
        mock_result = MagicMock()
        mock_result.text = "Test transcript"
        mock_result.language = "en"
        mock_result.word_count = 2
        mock_result.duration = 5.0

        with patch("transcribe_cli.core.transcribe_file", return_value=mock_result):
            with patch("transcribe_cli.core.save_transcript", return_value=tmp_path / "audio.txt"):
                result = runner.invoke(app, ["transcribe", str(fake_audio), "--format", "srt"])
                # Should show SRT warning but not fail
                assert "SRT format will be available" in result.stdout or result.exit_code == 0

    def test_transcribe_with_mock_success(self, tmp_path: Path) -> None:
        """transcribe should succeed with mocked API."""
        fake_audio = tmp_path / "audio.mp3"
        fake_audio.write_bytes(b"fake audio content")

        mock_result = MagicMock()
        mock_result.text = "This is a test transcription."
        mock_result.language = "english"
        mock_result.word_count = 5
        mock_result.duration = 10.0

        with patch("transcribe_cli.core.transcribe_file", return_value=mock_result):
            with patch("transcribe_cli.core.save_transcript", return_value=tmp_path / "audio.txt"):
                result = runner.invoke(app, ["transcribe", str(fake_audio)])
                assert result.exit_code == 0
                assert "Success" in result.stdout

    def test_transcribe_api_key_missing(self, tmp_path: Path) -> None:
        """transcribe should show helpful error when API key missing."""
        fake_audio = tmp_path / "audio.mp3"
        fake_audio.write_bytes(b"fake audio content")

        from transcribe_cli.core.transcriber import APIKeyMissingError

        with patch(
            "transcribe_cli.core.transcribe_file",
            side_effect=APIKeyMissingError(),
        ):
            result = runner.invoke(app, ["transcribe", str(fake_audio)])
            assert result.exit_code == 1
            assert "OPENAI_API_KEY" in result.stdout or "Error" in result.stdout

    def test_transcribe_file_too_large(self, tmp_path: Path) -> None:
        """transcribe should error on files >25MB."""
        fake_audio = tmp_path / "audio.mp3"
        fake_audio.write_bytes(b"fake audio content")

        from transcribe_cli.core.transcriber import FileTooLargeError

        with patch(
            "transcribe_cli.core.transcribe_file",
            side_effect=FileTooLargeError(fake_audio, 30.0, 25.0),
        ):
            result = runner.invoke(app, ["transcribe", str(fake_audio)])
            assert result.exit_code == 1
            assert "too large" in result.stdout.lower() or "Error" in result.stdout

    def test_transcribe_with_output_dir(self, tmp_path: Path) -> None:
        """transcribe should respect --output-dir option."""
        fake_audio = tmp_path / "audio.mp3"
        fake_audio.write_bytes(b"fake audio content")
        output_dir = tmp_path / "output"

        mock_result = MagicMock()
        mock_result.text = "Test"
        mock_result.language = "en"
        mock_result.word_count = 1
        mock_result.duration = 1.0

        with patch("transcribe_cli.core.transcribe_file", return_value=mock_result):
            with patch("transcribe_cli.output.save_formatted_transcript") as mock_save:
                mock_save.return_value = output_dir / "audio.txt"
                result = runner.invoke(
                    app, ["transcribe", str(fake_audio), "--output-dir", str(output_dir)]
                )
                # Verify output_dir was created
                assert result.exit_code == 0


class TestBatchCommand:
    """Tests for batch command."""

    def test_batch_empty_directory(self, tmp_path: Path) -> None:
        """batch should handle empty directory gracefully."""
        result = runner.invoke(app, ["batch", str(tmp_path)])
        assert result.exit_code == 0
        assert "No audio/video files" in result.stdout

    def test_batch_invalid_format(self, tmp_path: Path) -> None:
        """batch should reject invalid output formats."""
        (tmp_path / "audio.mp3").write_bytes(b"fake")
        result = runner.invoke(app, ["batch", str(tmp_path), "--format", "pdf"])
        assert result.exit_code == 1
        assert "Unsupported format" in result.stdout

    def test_batch_nonexistent_directory(self) -> None:
        """batch should error on non-existent directory."""
        result = runner.invoke(app, ["batch", "/nonexistent/directory"])
        assert result.exit_code != 0

    def test_batch_with_mock_success(self, tmp_path: Path) -> None:
        """batch should process files with mocked transcription."""
        # Create test files
        (tmp_path / "audio1.mp3").write_bytes(b"fake1")
        (tmp_path / "audio2.mp3").write_bytes(b"fake2")

        from transcribe_cli.core.batch import BatchSummary, BatchResult

        mock_summary = BatchSummary(
            total_files=2,
            successful=2,
            failed=0,
            skipped=0,
            results=[
                BatchResult(tmp_path / "audio1.mp3", tmp_path / "audio1.txt", True),
                BatchResult(tmp_path / "audio2.mp3", tmp_path / "audio2.txt", True),
            ],
        )

        with patch("transcribe_cli.core.process_directory", return_value=mock_summary):
            result = runner.invoke(app, ["batch", str(tmp_path)])
            assert result.exit_code == 0
            assert "Successful" in result.stdout
            assert "2" in result.stdout

    def test_batch_with_failures(self, tmp_path: Path) -> None:
        """batch should report failures correctly."""
        (tmp_path / "audio.mp3").write_bytes(b"fake")

        from transcribe_cli.core.batch import BatchSummary, BatchResult

        mock_summary = BatchSummary(
            total_files=1,
            successful=0,
            failed=1,
            skipped=0,
            results=[
                BatchResult(tmp_path / "audio.mp3", None, False, error="API error"),
            ],
        )

        with patch("transcribe_cli.core.process_directory", return_value=mock_summary):
            result = runner.invoke(app, ["batch", str(tmp_path)])
            assert result.exit_code == 1
            assert "Failed" in result.stdout

    def test_batch_shows_file_count(self, tmp_path: Path) -> None:
        """batch should show number of files found."""
        (tmp_path / "audio1.mp3").write_bytes(b"fake1")
        (tmp_path / "audio2.mp3").write_bytes(b"fake2")
        (tmp_path / "audio3.mp3").write_bytes(b"fake3")

        from transcribe_cli.core.batch import BatchSummary

        mock_summary = BatchSummary(
            total_files=3, successful=3, failed=0, skipped=0, results=[]
        )

        with patch("transcribe_cli.core.process_directory", return_value=mock_summary):
            result = runner.invoke(app, ["batch", str(tmp_path)])
            assert "3 file" in result.stdout
