# ADR-005: Configuration Management Strategy

## Status

**Accepted**

## Date

2025-12-04

## Context

The Audio Transcription CLI Tool requires a configuration management strategy that balances:

1. **Security**: The OpenAI API key (classified as Confidential per Data Classification Document) must be protected from exposure in logs, version control, and error messages.

2. **User Experience**: Users should be able to quickly get started (sensible defaults) while having flexibility to customize behavior for different workflows.

3. **Cross-Platform Compatibility**: Configuration must work consistently on Linux, macOS, and Windows (Python 3.9+).

4. **MVP Simplicity**: Avoid over-engineering while maintaining extensibility for future features.

Key configuration settings required:
- **OPENAI_API_KEY**: Required, Confidential classification
- **Output directory**: Optional, default to current directory
- **Output format**: Optional, default to txt
- **Concurrency limit**: Optional, default to 5 parallel requests
- **Language hint**: Optional, auto-detect if not specified
- **Temp directory**: Optional, use system default
- **Verbose/debug mode**: Optional, default to INFO logging

The SAD (Section 4.1.2) identifies `config/settings.py` as responsible for "Configuration loading and validation" with interface "Environment -> Config", and specifies pydantic and python-dotenv as the configuration technology stack.

## Decision

Implement a **hierarchical configuration system** with the following precedence (highest to lowest):

1. **CLI arguments** (per-invocation overrides)
2. **Environment variables** (session/system-wide settings)
3. **Config file** (`~/.transcriberc` YAML format or `.env` in working directory)
4. **Hardcoded defaults** (sensible fallbacks)

### Configuration Source Details

| Setting | CLI Arg | Env Var | Config File Key | Default |
|---------|---------|---------|-----------------|---------|
| API Key | N/A | `OPENAI_API_KEY` | `api_key` | None (required) |
| Output Dir | `--output-dir` | `TRANSCRIBE_OUTPUT_DIR` | `output_dir` | `.` (current) |
| Output Format | `--format` | `TRANSCRIBE_FORMAT` | `format` | `txt` |
| Concurrency | `--concurrency` | `TRANSCRIBE_CONCURRENCY` | `concurrency` | `5` |
| Language | `--language` | `TRANSCRIBE_LANGUAGE` | `language` | `auto` |
| Temp Dir | `--temp-dir` | `TRANSCRIBE_TEMP_DIR` | `temp_dir` | System default |
| Verbose | `--verbose` / `-v` | `TRANSCRIBE_VERBOSE` | `verbose` | `false` |

### API Key Handling (Security Critical)

The API key receives special handling due to its Confidential classification:

1. **Environment variable only for CLI**: API key cannot be passed as CLI argument to prevent exposure in shell history and process lists
2. **Config file with restricted permissions**: If stored in `~/.transcriberc`, file MUST have 600 permissions
3. **Never logged**: Use pydantic `SecretStr` type; logs show `[REDACTED]`
4. **Never in error messages**: Errors reference `OPENAI_API_KEY environment variable` not the value

### Config File Format

**Primary: `~/.transcriberc` (YAML)**

```yaml
# Audio Transcription CLI Configuration
# API key should be set via environment variable for security
# api_key: sk-...  # NOT RECOMMENDED - use OPENAI_API_KEY env var

output_dir: ~/transcripts
format: txt
concurrency: 5
language: auto
verbose: false
```

**Secondary: `.env` (working directory)**

```bash
# .env file in project directory
OPENAI_API_KEY=sk-...
TRANSCRIBE_OUTPUT_DIR=./transcripts
TRANSCRIBE_FORMAT=txt
TRANSCRIBE_CONCURRENCY=5
```

### Config File Location Search Order

1. `.env` in current working directory (python-dotenv)
2. `~/.transcriberc` (user home directory)
3. `/etc/transcribe/config.yaml` (system-wide, future consideration)

## Rationale

### Why Hierarchical Configuration?

1. **Flexibility**: Users can set defaults in config file but override per-invocation via CLI
2. **Security**: Environment variables are the safest place for API keys (not in command history)
3. **Portability**: Different machines can have different configs without changing code
4. **12-Factor App Alignment**: Environment variables for config is industry best practice

### Why `~/.transcriberc` YAML Over Other Options?

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| `~/.transcriberc` (YAML) | Human-readable, supports comments, familiar location pattern | Requires YAML parser | **Selected** |
| `~/.config/transcribe/config.yaml` | XDG compliant | Longer path, may not exist | v2 consideration |
| `.env` only | Simple, no parsing | No comments, limited types | Supplementary |
| `transcribe.yaml` (cwd) | Per-project config | Clutters projects | Rejected |
| JSON | No extra parser | No comments, verbose | Rejected |
| TOML | Modern, Python-native (3.11+) | Python 3.9 needs library | v2 when 3.9 dropped |

### Why pydantic for Validation?

1. **Type safety**: Automatic type coercion and validation
2. **SecretStr**: Built-in support for masking sensitive values
3. **Validation errors**: Clear, actionable error messages
4. **Already in stack**: SAD specifies pydantic for configuration
5. **Settings class**: pydantic-settings provides environment variable integration

### Security Design Decisions

Per Data Classification Document (Section 3 - Confidential Data Requirements):

| Requirement | Implementation |
|-------------|----------------|
| "Never logged" | `SecretStr` type, custom `__repr__` |
| "Never committed" | `.gitignore` entries, CI secrets scanner |
| "Masked in errors" | Exception handlers sanitize messages |
| "Environment variable" | Primary source for API key |
| "0600 permissions" | Validate config file permissions on load |

## Consequences

### Positive

- **Simple setup**: Users can start with just `export OPENAI_API_KEY=...`
- **Secure by default**: API key in env var, not command line
- **Flexible overrides**: CLI args override all other sources
- **Cross-platform**: Works identically on Linux, macOS, Windows
- **Extensible**: Adding new config options is straightforward
- **Type-safe**: pydantic catches configuration errors early
- **Self-documenting**: pydantic schema can generate config documentation

### Negative

- **YAML dependency**: Requires PyYAML (already needed for other features)
- **Multiple config locations**: Users may be confused which file takes precedence
- **Permission check overhead**: Validating file permissions adds startup time
- **Windows permissions**: File permission model differs (less strict)

### Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| API key in shell history | Medium | High | No CLI argument for API key, documentation warning |
| World-readable config file | Medium | High | Permission check on load, reject 644+ permissions |
| Config file with API key committed | Medium | High | `.gitignore` template, pre-commit hook docs |
| Conflicting config sources | Low | Medium | Clear precedence documentation, `--show-config` command |
| YAML parsing errors | Low | Low | Validate syntax, provide clear error messages |

## Alternatives Rejected

### CLI Argument for API Key

**Rejected because**: Exposes key in shell history (`history` command), process list (`ps aux`), and potentially in error logs. Per Data Classification Document, API key is Confidential and must never be logged or exposed.

**When to reconsider**: Never. This is a security requirement, not a preference.

### JSON Config File

**Rejected because**: JSON does not support comments. Users cannot document their configuration choices inline. Human editing of JSON is error-prone (trailing commas, quotes).

**When to reconsider**: If YAML parser adds significant startup latency (unlikely).

### TOML Config File

**Rejected because**: Python 3.9 does not include `tomllib` (added in 3.11). Would require additional dependency for minimal benefit over YAML.

**When to reconsider**: When Python 3.9 support is dropped (v2+).

### Config File in Current Directory Only

**Rejected because**: Requires copying config file to every project. Does not support user-level defaults. Clutters project directories.

**When to reconsider**: If per-project config becomes a requested feature (add as additional search location).

### No Config File (Environment Only)

**Rejected because**: Environment variables are awkward for complex configuration (output directories, format preferences). Users would need to set many variables for persistent preferences.

**When to reconsider**: If simplicity is paramount and user feedback indicates config file is unused.

## Implementation Guidance

### Config Model (pydantic)

```python
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class TranscribeSettings(BaseSettings):
    """Configuration settings for transcribe-cli."""

    model_config = SettingsConfigDict(
        env_prefix='TRANSCRIBE_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # API Key (Confidential - special handling)
    openai_api_key: SecretStr

    # Output settings
    output_dir: Path = Path(".")
    format: Literal["txt", "srt"] = "txt"

    # Processing settings
    concurrency: int = 5
    language: str = "auto"
    temp_dir: Optional[Path] = None

    # Logging
    verbose: bool = False

    @field_validator('concurrency')
    @classmethod
    def validate_concurrency(cls, v: int) -> int:
        if not 1 <= v <= 20:
            raise ValueError('concurrency must be between 1 and 20')
        return v

    @field_validator('output_dir', 'temp_dir', mode='before')
    @classmethod
    def expand_path(cls, v: Optional[str]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v).expanduser().resolve()
```

### Config File Loader

```python
import os
import stat
import yaml
from pathlib import Path

def load_config_file() -> dict:
    """Load configuration from file with security checks."""
    config_paths = [
        Path.cwd() / '.env',  # Handled by pydantic-settings
        Path.home() / '.transcriberc',
    ]

    for config_path in config_paths:
        if config_path.suffix == '.env':
            continue  # Handled by pydantic-settings

        if config_path.exists():
            # Security check: file permissions
            if os.name != 'nt':  # Unix-like systems
                mode = config_path.stat().st_mode
                if mode & (stat.S_IRGRP | stat.S_IROTH):
                    raise PermissionError(
                        f"Config file {config_path} has insecure permissions. "
                        f"Run: chmod 600 {config_path}"
                    )

            with open(config_path) as f:
                return yaml.safe_load(f) or {}

    return {}
```

### Config Merge Logic

```python
def get_settings(cli_overrides: dict = None) -> TranscribeSettings:
    """Get merged settings from all sources."""
    # 1. Load defaults (handled by pydantic defaults)
    # 2. Load config file
    file_config = load_config_file()

    # 3. Environment variables (handled by pydantic-settings)
    # 4. CLI overrides
    if cli_overrides:
        file_config.update(cli_overrides)

    return TranscribeSettings(**file_config)
```

### CLI Integration (click)

```python
import click
from functools import wraps

def config_options(f):
    """Decorator to add common config options to commands."""
    @click.option('--output-dir', '-o', type=click.Path(),
                  help='Output directory for transcripts')
    @click.option('--format', '-f', type=click.Choice(['txt', 'srt']),
                  help='Output format')
    @click.option('--concurrency', '-c', type=int,
                  help='Number of parallel API requests (1-20)')
    @click.option('--language', '-l', type=str,
                  help='Language hint (ISO 639-1 code or "auto")')
    @click.option('--verbose', '-v', is_flag=True,
                  help='Enable verbose output')
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

@click.command()
@config_options
@click.argument('input_file', type=click.Path(exists=True))
def transcribe(input_file, **options):
    """Transcribe an audio or video file."""
    # Filter None values (not provided on CLI)
    cli_overrides = {k: v for k, v in options.items() if v is not None}
    settings = get_settings(cli_overrides)
    # ... proceed with transcription
```

### Config Subcommand

```python
@click.group()
def config():
    """Manage configuration settings."""
    pass

@config.command('show')
def config_show():
    """Display current configuration (API key masked)."""
    settings = get_settings()
    click.echo("Current Configuration:")
    click.echo(f"  Output Directory: {settings.output_dir}")
    click.echo(f"  Format: {settings.format}")
    click.echo(f"  Concurrency: {settings.concurrency}")
    click.echo(f"  Language: {settings.language}")
    click.echo(f"  Verbose: {settings.verbose}")
    click.echo(f"  API Key: {'[SET]' if settings.openai_api_key else '[NOT SET]'}")

@config.command('init')
def config_init():
    """Create a default configuration file."""
    config_path = Path.home() / '.transcriberc'
    if config_path.exists():
        if not click.confirm(f'{config_path} exists. Overwrite?'):
            return

    default_config = """\
# Audio Transcription CLI Configuration
# See: https://github.com/project/transcribe-cli#configuration

# API key should be set via environment variable for security:
#   export OPENAI_API_KEY=sk-...

# Output settings
output_dir: ~/transcripts
format: txt

# Processing settings
concurrency: 5
language: auto

# Logging
verbose: false
"""
    config_path.write_text(default_config)
    os.chmod(config_path, 0o600)  # Secure permissions
    click.echo(f"Created {config_path} with secure permissions (600)")
```

### .gitignore Template

```gitignore
# Transcribe CLI - sensitive files
.env
*.env
.transcriberc

# Never commit API keys
**/secrets.*
**/credentials.*
```

## Testing Requirements

| Test Case | Description | Validation |
|-----------|-------------|------------|
| Default values | No config, no env vars | All defaults applied correctly |
| Env var override | Set `TRANSCRIBE_FORMAT=srt` | Format is srt |
| CLI override | `--format srt` with env var set to txt | CLI wins (srt) |
| Config file loading | Valid `~/.transcriberc` | Values loaded correctly |
| Permission check | Config file with 644 permissions | Rejected with clear error |
| API key masking | Print settings | API key shows `[REDACTED]` |
| Invalid concurrency | `--concurrency 50` | Validation error |
| Path expansion | `output_dir: ~/transcripts` | Expands to absolute path |
| Missing API key | No API key configured | Clear error with setup instructions |

## Related Decisions

- **ADR-001**: FFmpeg Integration Approach (FFmpeg path may be configurable in v2)
- **ADR-002**: Batch Processing Concurrency Model (concurrency setting used here)
- **ADR-003**: Output Format Support (format setting limited to txt/srt in MVP)
- **Data Classification Document**: Section 3 (Confidential Data Requirements)
- **SAD Section 10.6.4**: Configuration File Security requirements

## References

- [pydantic-settings Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [12-Factor App - Config](https://12factor.net/config)
- [python-dotenv Documentation](https://saurabh-kumar.com/python-dotenv/)
- [YAML 1.2 Specification](https://yaml.org/spec/1.2.2/)
- Data Classification Document: `/home/manitcor/dev/tnf/.aiwg/security/data-classification.md`
- SAD Section 4.1.2: `/home/manitcor/dev/tnf/.aiwg/architecture/software-architecture-doc.md`
