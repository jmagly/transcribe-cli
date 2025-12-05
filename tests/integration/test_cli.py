"""Integration tests for CLI commands."""

from pathlib import Path
from unittest.mock import patch

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
