"""Unit tests for configuration module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from transcribe_cli.config.settings import (
    Settings,
    create_default_config,
    find_config_file,
    get_config_locations,
    load_config_file,
)


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

    def test_settings_recursive_default(self) -> None:
        """Recursive should default to False."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            settings = Settings(_env_file=None)
            assert settings.recursive is False


class TestFindConfigFile:
    """Tests for config file discovery."""

    def test_returns_none_when_no_config(self, tmp_path: Path) -> None:
        """Returns None when no config file exists."""
        with patch(
            "transcribe_cli.config.settings.CONFIG_LOCATIONS",
            [tmp_path / "nonexistent.toml"],
        ):
            result = find_config_file()
            assert result is None

    def test_finds_first_existing_config(self, tmp_path: Path) -> None:
        """Finds first existing config in order."""
        config1 = tmp_path / "first.toml"
        config2 = tmp_path / "second.toml"
        config1.write_text("[output]\nformat = 'txt'\n")
        config2.write_text("[output]\nformat = 'srt'\n")

        with patch(
            "transcribe_cli.config.settings.CONFIG_LOCATIONS",
            [config1, config2],
        ):
            result = find_config_file()
            assert result == config1


class TestLoadConfigFile:
    """Tests for config file loading."""

    def test_loads_valid_toml(self, tmp_path: Path) -> None:
        """Loads valid TOML config."""
        config = tmp_path / "config.toml"
        config.write_text("""
[output]
format = "srt"

[processing]
concurrency = 10
language = "es"
""")
        result = load_config_file(config)
        assert result["output"]["format"] == "srt"
        assert result["processing"]["concurrency"] == 10
        assert result["processing"]["language"] == "es"

    def test_returns_empty_for_invalid_toml(self, tmp_path: Path) -> None:
        """Returns empty dict for invalid TOML."""
        config = tmp_path / "config.toml"
        config.write_text("this is not valid toml [[[")

        result = load_config_file(config)
        assert result == {}

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        """Returns empty dict for missing file."""
        result = load_config_file(tmp_path / "nonexistent.toml")
        assert result == {}

    def test_returns_empty_when_no_path_and_no_config(self, tmp_path: Path) -> None:
        """Returns empty when no path given and no config found."""
        with patch(
            "transcribe_cli.config.settings.CONFIG_LOCATIONS",
            [tmp_path / "nonexistent.toml"],
        ):
            result = load_config_file()
            assert result == {}


class TestCreateDefaultConfig:
    """Tests for config file creation."""

    def test_creates_config_file(self, tmp_path: Path) -> None:
        """Creates a config file at specified path."""
        config_path = tmp_path / "test.toml"
        result = create_default_config(config_path)

        assert result == config_path
        assert config_path.exists()
        content = config_path.read_text()
        assert "[output]" in content
        assert "[processing]" in content
        assert "concurrency = 5" in content

    def test_creates_nested_directories(self, tmp_path: Path) -> None:
        """Creates parent directories if needed."""
        config_path = tmp_path / "nested" / "deep" / "config.toml"
        result = create_default_config(config_path)

        assert result.exists()
        assert result.parent.exists()


class TestGetConfigLocations:
    """Tests for config locations helper."""

    def test_returns_list_of_paths(self) -> None:
        """Returns list of Path objects."""
        locations = get_config_locations()
        assert isinstance(locations, list)
        assert all(isinstance(loc, Path) for loc in locations)

    def test_returns_copy(self) -> None:
        """Returns a copy, not the original."""
        locations1 = get_config_locations()
        locations2 = get_config_locations()
        assert locations1 is not locations2
        assert locations1 == locations2
