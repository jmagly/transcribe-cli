"""Configuration settings for transcribe-cli.

Implements ADR-005: Configuration Management Strategy
- Hierarchical config: CLI args -> env vars -> config file -> defaults
- API key via environment variable only (security)
- pydantic SecretStr for sensitive data
- Local inference configuration for faster-whisper
- Diarization configuration for pyannote
"""

import tomllib
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Config file locations (checked in order)
CONFIG_LOCATIONS = [
    Path.cwd() / "transcribe.toml",
    Path.cwd() / ".transcriberc",
    Path.home() / ".config" / "transcribe" / "config.toml",
    Path.home() / ".transcriberc",
]


def find_config_file() -> Optional[Path]:
    """Find the first existing config file.

    Returns:
        Path to config file if found, None otherwise.
    """
    for path in CONFIG_LOCATIONS:
        if path.exists() and path.is_file():
            return path
    return None


def load_config_file(path: Optional[Path] = None) -> dict[str, Any]:
    """Load configuration from TOML file.

    Args:
        path: Explicit path to config file (optional).

    Returns:
        Dictionary of configuration values.
    """
    config_path = path or find_config_file()
    if not config_path:
        return {}

    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


class Settings(BaseSettings):
    """Application settings with hierarchical configuration.

    Priority (highest to lowest):
    1. CLI arguments (handled by Typer)
    2. Environment variables
    3. Config file (transcribe.toml or .transcriberc)
    4. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIBE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Backend selection: "local" for GPU inference, "api" for OpenAI API
    backend: Literal["local", "api"] = "local"

    # API Configuration (only needed if backend="api")
    # Uses OPENAI_API_KEY (no prefix) for compatibility with OpenAI conventions
    openai_api_key: SecretStr = Field(
        default=None,  # type: ignore[assignment]
        validation_alias="OPENAI_API_KEY",
    )

    # Local inference settings (only used if backend="local")
    model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = "medium"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    compute_type: Literal["int8", "float16", "float32", "auto"] = "auto"
    beam_size: int = 5
    vad_filter: bool = True

    # Diarization settings
    diarization: bool = False
    hf_token: SecretStr = Field(
        default=None,  # type: ignore[assignment]
        validation_alias="HF_TOKEN",
    )
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    # Output settings
    output_dir: Path = Path(".")
    output_format: Literal["txt", "srt", "diarized"] = "txt"

    # Processing settings
    concurrency: int = 5
    language: str = "auto"
    chunk_size_minutes: int = 10
    recursive: bool = False

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


def get_settings(config_path: Optional[Path] = None) -> Settings:
    """Load settings from environment and config file.

    Args:
        config_path: Explicit path to config file (optional).

    Returns:
        Settings: Validated application settings.

    Raises:
        ValidationError: If required settings are missing or invalid.
    """
    # Load config file values first
    file_config = load_config_file(config_path)

    # Create settings - env vars will override file config
    return Settings(**file_config)  # type: ignore[arg-type]


def get_config_locations() -> list[Path]:
    """Get list of config file locations checked.

    Returns:
        List of paths checked for config files.
    """
    return CONFIG_LOCATIONS.copy()


def create_default_config(path: Optional[Path] = None) -> Path:
    """Create a default config file with comments.

    Args:
        path: Where to create the config file (default: ./transcribe.toml).

    Returns:
        Path to created config file.
    """
    config_path = path or Path.cwd() / "transcribe.toml"

    content = '''# Transcribe CLI Configuration
# See: transcribe config --show

[backend]
# Backend: "local" for GPU inference, "api" for OpenAI API
backend = "local"

[local]
# Model size: tiny, base, small, medium, large-v3
model_size = "medium"

# Device: cuda, cpu, auto
device = "auto"

# Compute type: int8, float16, float32, auto
# int8 recommended for older GPUs (Pascal/GTX 10xx)
compute_type = "auto"

# Beam size for decoding (higher = more accurate, slower)
beam_size = 5

# Voice activity detection filter
vad_filter = true

[diarization]
# Enable speaker diarization (requires HF_TOKEN)
enabled = false

# Speaker count hints (optional)
# min_speakers = 2
# max_speakers = 5

[output]
# Output format: "txt", "srt", or "diarized"
format = "txt"

# Default output directory (empty = same as input)
# dir = "./transcripts"

[processing]
# Maximum concurrent jobs (1-20)
concurrency = 5

# Language code or "auto" for detection
language = "auto"

# Recursively scan directories
recursive = false

[logging]
# Enable verbose output
verbose = false
'''

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content)
    return config_path
