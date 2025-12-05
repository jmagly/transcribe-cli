"""Unit tests for batch processing module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcribe_cli.core.batch import (
    BatchResult,
    BatchSummary,
    scan_directory,
)


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_successful_result(self) -> None:
        """Successful result has correct attributes."""
        result = BatchResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.txt"),
            success=True,
            error=None,
            result=None,
        )
        assert result.success is True
        assert result.error is None

    def test_failed_result(self) -> None:
        """Failed result stores error message."""
        result = BatchResult(
            input_path=Path("test.mp3"),
            output_path=None,
            success=False,
            error="API error",
            result=None,
        )
        assert result.success is False
        assert result.error == "API error"


class TestBatchSummary:
    """Tests for BatchSummary dataclass."""

    def test_success_rate_all_successful(self) -> None:
        """100% success rate when all files succeed."""
        summary = BatchSummary(
            total_files=10,
            successful=10,
            failed=0,
            skipped=0,
            results=[],
        )
        assert summary.success_rate == 100.0

    def test_success_rate_partial(self) -> None:
        """Correct success rate with some failures."""
        summary = BatchSummary(
            total_files=10,
            successful=7,
            failed=3,
            skipped=0,
            results=[],
        )
        assert summary.success_rate == 70.0

    def test_success_rate_all_failed(self) -> None:
        """0% success rate when all fail."""
        summary = BatchSummary(
            total_files=5,
            successful=0,
            failed=5,
            skipped=0,
            results=[],
        )
        assert summary.success_rate == 0.0

    def test_success_rate_empty(self) -> None:
        """0% success rate for empty batch."""
        summary = BatchSummary(
            total_files=0,
            successful=0,
            failed=0,
            skipped=0,
            results=[],
        )
        assert summary.success_rate == 0.0


class TestScanDirectory:
    """Tests for directory scanning."""

    def test_finds_audio_files(self, tmp_path: Path) -> None:
        """Audio files are found."""
        (tmp_path / "audio1.mp3").write_bytes(b"fake")
        (tmp_path / "audio2.wav").write_bytes(b"fake")
        (tmp_path / "audio3.flac").write_bytes(b"fake")

        files = scan_directory(tmp_path)
        assert len(files) == 3
        extensions = {f.suffix.lower() for f in files}
        assert extensions == {".mp3", ".wav", ".flac"}

    def test_finds_video_files(self, tmp_path: Path) -> None:
        """Video files are found."""
        (tmp_path / "video1.mp4").write_bytes(b"fake")
        (tmp_path / "video2.mkv").write_bytes(b"fake")
        (tmp_path / "video3.avi").write_bytes(b"fake")

        files = scan_directory(tmp_path)
        assert len(files) == 3
        extensions = {f.suffix.lower() for f in files}
        assert extensions == {".mp4", ".mkv", ".avi"}

    def test_ignores_unsupported_files(self, tmp_path: Path) -> None:
        """Non-media files are ignored."""
        (tmp_path / "audio.mp3").write_bytes(b"fake")
        (tmp_path / "document.pdf").write_bytes(b"fake")
        (tmp_path / "image.jpg").write_bytes(b"fake")
        (tmp_path / "notes.txt").write_text("notes")

        files = scan_directory(tmp_path)
        assert len(files) == 1
        assert files[0].suffix == ".mp3"

    def test_ignores_subdirectories_by_default(self, tmp_path: Path) -> None:
        """Subdirectories are not scanned by default."""
        (tmp_path / "audio.mp3").write_bytes(b"fake")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.mp3").write_bytes(b"fake")

        files = scan_directory(tmp_path, recursive=False)
        assert len(files) == 1

    def test_recursive_scan(self, tmp_path: Path) -> None:
        """Recursive scan finds files in subdirectories."""
        (tmp_path / "audio.mp3").write_bytes(b"fake")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.mp3").write_bytes(b"fake")

        files = scan_directory(tmp_path, recursive=True)
        assert len(files) == 2

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        files = scan_directory(tmp_path)
        assert files == []

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        """Non-existent directory raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            scan_directory(tmp_path / "nonexistent")

    def test_file_not_directory_raises(self, tmp_path: Path) -> None:
        """File path raises ValueError."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")
        with pytest.raises(ValueError, match="not a directory"):
            scan_directory(file_path)

    def test_sorted_output(self, tmp_path: Path) -> None:
        """Files are returned in sorted order."""
        (tmp_path / "c_audio.mp3").write_bytes(b"fake")
        (tmp_path / "a_audio.mp3").write_bytes(b"fake")
        (tmp_path / "b_audio.mp3").write_bytes(b"fake")

        files = scan_directory(tmp_path)
        names = [f.name for f in files]
        assert names == sorted(names)

    def test_case_insensitive_extensions(self, tmp_path: Path) -> None:
        """Extensions are matched case-insensitively."""
        (tmp_path / "audio.MP3").write_bytes(b"fake")
        (tmp_path / "video.MKV").write_bytes(b"fake")

        files = scan_directory(tmp_path)
        assert len(files) == 2


class TestProcessBatch:
    """Tests for batch processing (with mocks)."""

    def test_empty_batch_returns_empty_summary(self) -> None:
        """Empty file list returns summary with zeros."""
        from transcribe_cli.core.batch import process_batch

        summary = process_batch(files=[])
        assert summary.total_files == 0
        assert summary.successful == 0
        assert summary.failed == 0

    def test_process_batch_with_mock(self, tmp_path: Path) -> None:
        """Batch processing calls transcription for each file."""
        from transcribe_cli.core.batch import process_batch
        from transcribe_cli.core.transcriber import TranscriptionResult

        # Create test files
        (tmp_path / "audio1.mp3").write_bytes(b"fake1")
        (tmp_path / "audio2.mp3").write_bytes(b"fake2")
        files = list(tmp_path.glob("*.mp3"))

        # Mock transcription
        mock_result = MagicMock(spec=TranscriptionResult)
        mock_result.text = "Test"
        mock_result.segments = []
        mock_result.language = "en"
        mock_result.duration = 1.0

        with patch("transcribe_cli.core.batch.transcribe_file", return_value=mock_result):
            with patch("transcribe_cli.output.formatters.save_formatted_transcript") as mock_save:
                mock_save.return_value = tmp_path / "output.txt"
                summary = process_batch(
                    files=files,
                    output_format="txt",
                    concurrency=2,
                    api_key="sk-test",
                )

        assert summary.total_files == 2
        assert summary.successful == 2
        assert summary.failed == 0


class TestProcessDirectory:
    """Tests for directory processing."""

    def test_process_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns summary with zeros."""
        from transcribe_cli.core.batch import process_directory

        summary = process_directory(tmp_path)
        assert summary.total_files == 0

    def test_process_directory_scans_and_processes(self, tmp_path: Path) -> None:
        """Directory is scanned and files processed."""
        from transcribe_cli.core.batch import process_directory
        from transcribe_cli.core.transcriber import TranscriptionResult

        # Create test file
        (tmp_path / "audio.mp3").write_bytes(b"fake")

        mock_result = MagicMock(spec=TranscriptionResult)
        mock_result.text = "Test"
        mock_result.segments = []
        mock_result.language = "en"
        mock_result.duration = 1.0

        with patch("transcribe_cli.core.batch.transcribe_file", return_value=mock_result):
            with patch("transcribe_cli.output.formatters.save_formatted_transcript") as mock_save:
                mock_save.return_value = tmp_path / "audio.txt"
                summary = process_directory(tmp_path, api_key="sk-test")

        assert summary.total_files == 1
        assert summary.successful == 1
