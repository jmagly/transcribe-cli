"""Configuration settings for transcribe-cli.

Implements ADR-005: Configuration Management Strategy
- Hierarchical config: CLI args -> env vars -> config file -> defaults
- API key via environment variable only (security)
- pydantic SecretStr for sensitive data
- Local inference configuration for faster-whisper
- Diarization configuration for pyannote
"""

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import AliasChoices, Field, SecretStr, field_validator
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
        populate_by_name=True,
    )

    # Backend selection: "local" for GPU inference, "api" for OpenAI API
    backend: Literal["local", "api"] = "local"

    # API Configuration (only needed if backend="api")
    # Uses OPENAI_API_KEY (no prefix) for compatibility with OpenAI conventions
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
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
    hf_token: Optional[SecretStr] = Field(
        default=None,
        validation_alias="HF_TOKEN",
    )
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    # Output settings
    output_dir: Path = Path(".")
    output_format: Literal["txt", "srt", "diarized"] = Field(
        default="txt",
        # Keep compatibility with documented variable while supporting model-style naming.
        validation_alias=AliasChoices("TRANSCRIBE_FORMAT", "TRANSCRIBE_OUTPUT_FORMAT"),
    )

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
    # Load config file values first and flatten TOML sections to model fields.
    file_config = _flatten_config(load_config_file(config_path))

    # Create settings - env vars will override file config
    return Settings(**file_config)  # type: ignore[arg-type]


def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested TOML sections into Settings field names."""
    if not config:
        return {}

    flattened: dict[str, Any] = {}

    # Keep explicit top-level keys if provided.
    for key in (
        "backend",
        "openai_api_key",
        "model_size",
        "device",
        "compute_type",
        "beam_size",
        "vad_filter",
        "diarization",
        "hf_token",
        "min_speakers",
        "max_speakers",
        "output_dir",
        "output_format",
        "concurrency",
        "language",
        "chunk_size_minutes",
        "recursive",
        "verbose",
        "quiet",
    ):
        if key in config:
            flattened[key] = config[key]

    output = config.get("output")
    if isinstance(output, dict):
        if "dir" in output:
            flattened["output_dir"] = output["dir"]
        if "format" in output:
            flattened["output_format"] = output["format"]

    backend = config.get("backend")
    if isinstance(backend, dict):
        if "backend" in backend:
            flattened["backend"] = backend["backend"]

    local = config.get("local")
    if isinstance(local, dict):
        if "model_size" in local:
            flattened["model_size"] = local["model_size"]
        if "device" in local:
            flattened["device"] = local["device"]
        if "compute_type" in local:
            flattened["compute_type"] = local["compute_type"]
        if "beam_size" in local:
            flattened["beam_size"] = local["beam_size"]
        if "vad_filter" in local:
            flattened["vad_filter"] = local["vad_filter"]

    diarization = config.get("diarization")
    if isinstance(diarization, dict):
        if "enabled" in diarization:
            flattened["diarization"] = diarization["enabled"]
        if "min_speakers" in diarization:
            flattened["min_speakers"] = diarization["min_speakers"]
        if "max_speakers" in diarization:
            flattened["max_speakers"] = diarization["max_speakers"]

    processing = config.get("processing")
    if isinstance(processing, dict):
        if "concurrency" in processing:
            flattened["concurrency"] = processing["concurrency"]
        if "language" in processing:
            flattened["language"] = processing["language"]
        if "chunk_size_minutes" in processing:
            flattened["chunk_size_minutes"] = processing["chunk_size_minutes"]
        if "recursive" in processing:
            flattened["recursive"] = processing["recursive"]

    logging = config.get("logging")
    if isinstance(logging, dict):
        if "verbose" in logging:
            flattened["verbose"] = logging["verbose"]
        if "quiet" in logging:
            flattened["quiet"] = logging["quiet"]

    return flattened


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
