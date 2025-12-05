"""Unit tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from transcribe_cli.config.settings import Settings


class TestSettings:
    """Tests for Settings configuration class."""

    def test_settings_loads_from_env(self) -> None:
        """Settings should load API key from environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}, clear=True):
            settings = Settings(_env_file=None)
            assert settings.openai_api_key.get_secret_value() == "sk-test-key"

    def test_settings_requires_api_key(self) -> None:
        """Settings should raise error when API key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError):
                Settings(_env_file=None)

    def test_settings_default_values(self) -> None:
        """Settings should have correct default values."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            settings = Settings(_env_file=None)
            assert settings.output_format == "txt"
            assert settings.concurrency == 5
            assert settings.language == "auto"
            assert settings.verbose is False
            assert settings.quiet is False

    def test_settings_concurrency_validation_min(self) -> None:
        """Concurrency should not be less than 1."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test", "TRANSCRIBE_CONCURRENCY": "0"},
            clear=True,
        ):
            with pytest.raises(ValidationError) as exc_info:
                Settings(_env_file=None)
            assert "at least 1" in str(exc_info.value)

    def test_settings_concurrency_validation_max(self) -> None:
        """Concurrency should not exceed 20."""
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "sk-test", "TRANSCRIBE_CONCURRENCY": "25"},
            clear=True,
        ):
            with pytest.raises(ValidationError) as exc_info:
                Settings(_env_file=None)
            assert "cannot exceed 20" in str(exc_info.value)

    def test_settings_output_dir_resolved(self) -> None:
        """Output directory should be resolved to absolute path."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            settings = Settings(_env_file=None, output_dir=Path("."))
            assert settings.output_dir.is_absolute()

    def test_settings_api_key_not_in_repr(self) -> None:
        """API key should not appear in string representation (security)."""
        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-secret-key-12345"}, clear=True
        ):
            settings = Settings(_env_file=None)
            repr_str = repr(settings)
            assert "sk-secret-key-12345" not in repr_str
            assert "SecretStr" in repr_str or "**" in repr_str
