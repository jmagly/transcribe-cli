"""Configuration settings for transcribe-cli.

Implements ADR-005: Configuration Management Strategy
- Hierarchical config: CLI args -> env vars -> config file -> defaults
- API key via environment variable only (security)
- pydantic SecretStr for sensitive data
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with hierarchical configuration.

    Priority (highest to lowest):
    1. CLI arguments (handled by Typer)
    2. Environment variables
    3. Config file (~/.transcriberc)
    4. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIBE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Configuration (required, no default)
    openai_api_key: SecretStr

    # Output settings
    output_dir: Path = Path(".")
    output_format: Literal["txt", "srt"] = "txt"

    # Processing settings
    concurrency: int = 5
    language: str = "auto"
    chunk_size_minutes: int = 10

    # Logging settings
    verbose: bool = False
    quiet: bool = False

    @field_validator("concurrency")
    @classmethod
    def validate_concurrency(cls, v: int) -> int:
        """Ensure concurrency is within reasonable bounds."""
        if v < 1:
            raise ValueError("Concurrency must be at least 1")
        if v > 20:
            raise ValueError("Concurrency cannot exceed 20 (API rate limits)")
        return v

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Ensure output directory exists or can be created."""
        v = Path(v).resolve()
        if v.exists() and not v.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {v}")
        return v


def get_settings() -> Settings:
    """Load settings from environment and config file.

    Returns:
        Settings: Validated application settings.

    Raises:
        ValidationError: If required settings are missing or invalid.
    """
    return Settings()  # type: ignore[call-arg]
