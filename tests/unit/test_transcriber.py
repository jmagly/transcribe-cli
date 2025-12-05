"""Unit tests for transcription module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, RateLimitError

from transcribe_cli.core.transcriber import (
    APIKeyMissingError,
    FileTooLargeError,
    TranscriptionError,
    TranscriptionResult,
    TranscriptionSegment,
    _check_file_size,
    _create_client,
    _parse_segments,
    save_transcript,
)


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""

    def test_segment_duration(self) -> None:
        """Duration is calculated correctly."""
        segment = TranscriptionSegment(id=0, start=1.5, end=4.5, text="Hello world")
        assert segment.duration == 3.0

    def test_segment_attributes(self) -> None:
        """Segment attributes are stored correctly."""
        segment = TranscriptionSegment(id=5, start=10.0, end=15.0, text="Test text")
        assert segment.id == 5
        assert segment.start == 10.0
        assert segment.end == 15.0
        assert segment.text == "Test text"


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_word_count(self) -> None:
        """Word count is calculated correctly."""
        result = TranscriptionResult(
            input_path=Path("test.mp3"),
            output_path=Path("test.txt"),
            text="This is a test with five words.",
            segments=[],
            language="en",
            duration=10.0,
        )
        assert result.word_count == 7  # "This is a test with five words."

    def test_result_attributes(self) -> None:
        """Result attributes are stored correctly."""
        result = TranscriptionResult(
            input_path=Path("input.mp3"),
            output_path=Path("output.txt"),
            text="Hello",
            segments=[],
            language="en",
            duration=5.5,
        )
        assert result.input_path == Path("input.mp3")
        assert result.output_path == Path("output.txt")
        assert result.language == "en"
        assert result.duration == 5.5


class TestCheckFileSize:
    """Tests for file size validation."""

    def test_small_file_passes(self, tmp_path: Path) -> None:
        """Small files pass validation."""
        small_file = tmp_path / "small.mp3"
        small_file.write_bytes(b"x" * 1024)  # 1KB
        _check_file_size(small_file)  # Should not raise

    def test_large_file_raises(self, tmp_path: Path) -> None:
        """Files over 25MB raise FileTooLargeError."""
        large_file = tmp_path / "large.mp3"
        # Create a file just over 25MB
        large_file.write_bytes(b"x" * (26 * 1024 * 1024))

        with pytest.raises(FileTooLargeError) as exc_info:
            _check_file_size(large_file)

        assert exc_info.value.size_mb > 25.0
        assert "too large" in str(exc_info.value).lower()


class TestCreateClient:
    """Tests for OpenAI client creation."""

    def test_client_with_api_key(self) -> None:
        """Client is created with provided API key."""
        with patch("transcribe_cli.core.transcriber.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.api_key = "sk-test"
            mock_openai.return_value = mock_client

            client = _create_client("sk-test")
            assert client is mock_client
            mock_openai.assert_called_once_with(api_key="sk-test")

    def test_client_missing_key_raises(self) -> None:
        """Missing API key raises APIKeyMissingError."""
        with patch("transcribe_cli.core.transcriber.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_client.api_key = None
            mock_openai.return_value = mock_client

            with pytest.raises(APIKeyMissingError):
                _create_client(None)


class TestParseSegments:
    """Tests for segment parsing."""

    def test_parse_segments_success(self) -> None:
        """Segments are parsed correctly from API response."""
        response = {
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.5, "text": " Hello world."},
                {"id": 1, "start": 2.5, "end": 5.0, "text": " How are you?"},
            ]
        }

        segments = _parse_segments(response)
        assert len(segments) == 2
        assert segments[0].text == "Hello world."
        assert segments[1].start == 2.5

    def test_parse_empty_segments(self) -> None:
        """Empty segments list returns empty list."""
        response = {"segments": []}
        segments = _parse_segments(response)
        assert segments == []

    def test_parse_missing_segments(self) -> None:
        """Missing segments key returns empty list."""
        response = {"text": "Hello"}
        segments = _parse_segments(response)
        assert segments == []


class TestSaveTranscript:
    """Tests for saving transcripts."""

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Transcript is saved to file."""
        result = TranscriptionResult(
            input_path=tmp_path / "input.mp3",
            output_path=tmp_path / "output.txt",
            text="Hello world.",
            segments=[],
            language="en",
            duration=5.0,
        )

        saved_path = save_transcript(result)
        assert saved_path.exists()
        assert saved_path.read_text(encoding="utf-8") == "Hello world."

    def test_save_with_custom_path(self, tmp_path: Path) -> None:
        """Transcript is saved to custom path."""
        result = TranscriptionResult(
            input_path=tmp_path / "input.mp3",
            output_path=None,
            text="Custom content.",
            segments=[],
            language="en",
            duration=5.0,
        )

        custom_path = tmp_path / "custom" / "transcript.txt"
        saved_path = save_transcript(result, custom_path)

        assert saved_path == custom_path
        assert custom_path.exists()
        assert custom_path.read_text(encoding="utf-8") == "Custom content."

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        """Parent directory is created if needed."""
        result = TranscriptionResult(
            input_path=tmp_path / "input.mp3",
            output_path=tmp_path / "nested" / "deep" / "output.txt",
            text="Test",
            segments=[],
            language="en",
            duration=5.0,
        )

        saved_path = save_transcript(result)
        assert saved_path.exists()

    def test_save_utf8_content(self, tmp_path: Path) -> None:
        """Non-ASCII content is saved with UTF-8 encoding."""
        result = TranscriptionResult(
            input_path=tmp_path / "input.mp3",
            output_path=tmp_path / "output.txt",
            text="Héllo wörld! 日本語",
            segments=[],
            language="en",
            duration=5.0,
        )

        saved_path = save_transcript(result)
        content = saved_path.read_text(encoding="utf-8")
        assert "Héllo" in content
        assert "日本語" in content


class TestAPIKeyMissingError:
    """Tests for APIKeyMissingError."""

    def test_error_message_includes_instructions(self) -> None:
        """Error message includes setup instructions."""
        error = APIKeyMissingError()
        message = str(error)
        assert "OPENAI_API_KEY" in message
        assert "export" in message or "environment" in message.lower()


class TestFileTooLargeError:
    """Tests for FileTooLargeError."""

    def test_error_message_includes_size(self) -> None:
        """Error message includes file size and limit."""
        error = FileTooLargeError(Path("large.mp3"), 30.5, 25.0)
        message = str(error)
        assert "30.5" in message
        assert "25" in message

    def test_error_stores_attributes(self) -> None:
        """Error stores path and size attributes."""
        error = FileTooLargeError(Path("test.mp3"), 30.0, 25.0)
        assert error.path == Path("test.mp3")
        assert error.size_mb == 30.0
        assert error.max_mb == 25.0


class TestTranscribeFileWithMocks:
    """Tests for transcribe_file with mocked API."""

    def test_transcribe_audio_file(self, tmp_path: Path) -> None:
        """Audio file is transcribed successfully."""
        from transcribe_cli.core.transcriber import transcribe_file

        # Create fake audio file
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio content")

        # Mock API response
        mock_response = {
            "text": "This is the transcribed text.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": " This is"},
                {"id": 1, "start": 2.0, "end": 4.0, "text": " the transcribed text."},
            ],
            "language": "english",
            "duration": 4.0,
        }

        with patch("transcribe_cli.core.transcriber._create_client") as mock_create:
            with patch("transcribe_cli.core.transcriber._transcribe_audio_file") as mock_transcribe:
                mock_transcribe.return_value = mock_response
                mock_create.return_value = MagicMock()

                result = transcribe_file(audio_file, api_key="sk-test")

                assert result.text == "This is the transcribed text."
                assert result.language == "english"
                assert len(result.segments) == 2

    def test_transcribe_file_not_found(self, tmp_path: Path) -> None:
        """Non-existent file raises FileNotFoundError."""
        from transcribe_cli.core.transcriber import transcribe_file

        with pytest.raises(FileNotFoundError):
            transcribe_file(tmp_path / "nonexistent.mp3", api_key="sk-test")

    def test_transcribe_rate_limit_error(self, tmp_path: Path) -> None:
        """Rate limit error is converted to TranscriptionError."""
        from transcribe_cli.core.transcriber import transcribe_file

        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        with patch("transcribe_cli.core.transcriber._create_client") as mock_create:
            with patch("transcribe_cli.core.transcriber._transcribe_audio_file") as mock_transcribe:
                mock_create.return_value = MagicMock()
                # Create a proper RateLimitError mock
                mock_response = MagicMock()
                mock_response.status_code = 429
                mock_transcribe.side_effect = RateLimitError(
                    "Rate limit exceeded",
                    response=mock_response,
                    body=None,
                )

                with pytest.raises(TranscriptionError) as exc_info:
                    transcribe_file(audio_file, api_key="sk-test")

                assert "rate limit" in str(exc_info.value).lower()
