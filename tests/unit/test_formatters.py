"""Unit tests for output formatters."""

from datetime import timedelta
from pathlib import Path

import pytest
import srt

from transcribe_cli.core.transcriber import TranscriptionResult, TranscriptionSegment
from transcribe_cli.output.formatters import (
    _seconds_to_timedelta,
    format_as_srt,
    format_as_txt,
    format_transcript,
    get_output_extension,
    save_formatted_transcript,
)


class TestSecondsToTimedelta:
    """Tests for timestamp conversion."""

    def test_zero_seconds(self) -> None:
        """Zero seconds converts correctly."""
        result = _seconds_to_timedelta(0.0)
        assert result == timedelta(seconds=0)

    def test_whole_seconds(self) -> None:
        """Whole seconds convert correctly."""
        result = _seconds_to_timedelta(5.0)
        assert result == timedelta(seconds=5)

    def test_milliseconds(self) -> None:
        """Milliseconds are preserved."""
        result = _seconds_to_timedelta(1.5)
        assert result == timedelta(seconds=1.5)

    def test_minutes(self) -> None:
        """Minutes convert correctly."""
        result = _seconds_to_timedelta(125.0)
        assert result == timedelta(minutes=2, seconds=5)

    def test_hours(self) -> None:
        """Hours convert correctly."""
        result = _seconds_to_timedelta(3665.5)
        assert result == timedelta(hours=1, minutes=1, seconds=5, milliseconds=500)


class TestFormatAsTxt:
    """Tests for TXT formatter."""

    def test_simple_text(self) -> None:
        """Simple text is returned unchanged."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.txt"),
            text="Hello world.",
            segments=[],
            language="en",
            duration=5.0,
        )
        assert format_as_txt(result) == "Hello world."

    def test_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace is stripped."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.txt"),
            text="  Hello world.  \n",
            segments=[],
            language="en",
            duration=5.0,
        )
        assert format_as_txt(result) == "Hello world."

    def test_unicode_preserved(self) -> None:
        """Unicode characters are preserved."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.txt"),
            text="Héllo wörld! 日本語",
            segments=[],
            language="en",
            duration=5.0,
        )
        assert "Héllo" in format_as_txt(result)
        assert "日本語" in format_as_txt(result)


class TestFormatAsSrt:
    """Tests for SRT formatter."""

    def test_single_segment(self) -> None:
        """Single segment formats correctly."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.srt"),
            text="Hello world.",
            segments=[
                TranscriptionSegment(id=0, start=0.0, end=2.5, text="Hello world."),
            ],
            language="en",
            duration=2.5,
        )
        srt_output = format_as_srt(result)

        # Parse and validate
        subtitles = list(srt.parse(srt_output))
        assert len(subtitles) == 1
        assert subtitles[0].index == 1
        assert subtitles[0].content == "Hello world."

    def test_multiple_segments(self) -> None:
        """Multiple segments format correctly."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.srt"),
            text="Hello world. How are you?",
            segments=[
                TranscriptionSegment(id=0, start=0.0, end=2.0, text="Hello world."),
                TranscriptionSegment(id=1, start=2.0, end=4.0, text="How are you?"),
            ],
            language="en",
            duration=4.0,
        )
        srt_output = format_as_srt(result)

        subtitles = list(srt.parse(srt_output))
        assert len(subtitles) == 2
        assert subtitles[0].index == 1
        assert subtitles[1].index == 2
        assert subtitles[0].content == "Hello world."
        assert subtitles[1].content == "How are you?"

    def test_timestamps_format(self) -> None:
        """Timestamps are in correct SRT format."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.srt"),
            text="Test",
            segments=[
                TranscriptionSegment(id=0, start=65.5, end=68.123, text="Test"),
            ],
            language="en",
            duration=70.0,
        )
        srt_output = format_as_srt(result)

        subtitles = list(srt.parse(srt_output))
        # Check timestamps
        assert subtitles[0].start == timedelta(seconds=65.5)
        assert subtitles[0].end.total_seconds() == pytest.approx(68.123, rel=0.001)

    def test_no_segments_with_duration(self) -> None:
        """No segments but has duration creates single subtitle."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.srt"),
            text="Full transcript text.",
            segments=[],
            language="en",
            duration=10.0,
        )
        srt_output = format_as_srt(result)

        subtitles = list(srt.parse(srt_output))
        assert len(subtitles) == 1
        assert subtitles[0].content == "Full transcript text."

    def test_no_segments_no_duration_raises(self) -> None:
        """No segments and no duration raises ValueError."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.srt"),
            text="",
            segments=[],
            language="en",
            duration=None,
        )
        with pytest.raises(ValueError, match="no segments"):
            format_as_srt(result)

    def test_whitespace_stripped_from_segments(self) -> None:
        """Segment text whitespace is stripped."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.srt"),
            text="Test",
            segments=[
                TranscriptionSegment(id=0, start=0.0, end=2.0, text="  Test  "),
            ],
            language="en",
            duration=2.0,
        )
        srt_output = format_as_srt(result)
        subtitles = list(srt.parse(srt_output))
        assert subtitles[0].content == "Test"


class TestFormatTranscript:
    """Tests for format_transcript dispatcher."""

    def test_txt_format(self) -> None:
        """TXT format dispatches correctly."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.txt"),
            text="Hello",
            segments=[],
            language="en",
            duration=1.0,
        )
        output = format_transcript(result, "txt")
        assert output == "Hello"

    def test_srt_format(self) -> None:
        """SRT format dispatches correctly."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.srt"),
            text="Hello",
            segments=[
                TranscriptionSegment(id=0, start=0.0, end=1.0, text="Hello"),
            ],
            language="en",
            duration=1.0,
        )
        output = format_transcript(result, "srt")
        assert "Hello" in output
        assert "-->" in output  # SRT timestamp separator

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.txt"),
            text="Hello",
            segments=[],
            language="en",
            duration=1.0,
        )
        with pytest.raises(ValueError, match="Unsupported"):
            format_transcript(result, "pdf")  # type: ignore


class TestSaveFormattedTranscript:
    """Tests for saving formatted transcripts."""

    def test_save_txt(self, tmp_path: Path) -> None:
        """TXT file is saved correctly."""
        result = TranscriptionResult(
            input_path=tmp_path / "test.mp3",
            output_path=tmp_path / "test.txt",
            text="Hello world.",
            segments=[],
            language="en",
            duration=1.0,
        )
        output_path = tmp_path / "output.txt"
        saved = save_formatted_transcript(result, output_path, "txt")

        assert saved == output_path
        assert output_path.exists()
        assert output_path.read_text(encoding="utf-8") == "Hello world."

    def test_save_srt(self, tmp_path: Path) -> None:
        """SRT file is saved correctly."""
        result = TranscriptionResult(
            input_path=tmp_path / "test.mp3",
            output_path=tmp_path / "test.srt",
            text="Hello",
            segments=[
                TranscriptionSegment(id=0, start=0.0, end=1.0, text="Hello"),
            ],
            language="en",
            duration=1.0,
        )
        output_path = tmp_path / "output.srt"
        saved = save_formatted_transcript(result, output_path, "srt")

        assert saved == output_path
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "Hello" in content
        assert "-->" in content

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Parent directories are created if needed."""
        result = TranscriptionResult(
            input_path=tmp_path / "test.mp3",
            output_path=None,
            text="Test",
            segments=[],
            language="en",
            duration=1.0,
        )
        output_path = tmp_path / "nested" / "deep" / "output.txt"
        saved = save_formatted_transcript(result, output_path, "txt")

        assert saved.exists()


class TestGetOutputExtension:
    """Tests for output extension helper."""

    def test_txt_extension(self) -> None:
        """TXT returns .txt."""
        assert get_output_extension("txt") == ".txt"

    def test_srt_extension(self) -> None:
        """SRT returns .srt."""
        assert get_output_extension("srt") == ".srt"
