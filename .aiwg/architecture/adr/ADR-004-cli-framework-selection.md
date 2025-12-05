# ADR-004: CLI Framework Selection

## Status

**Accepted**

## Date

2025-12-04

## Context

The Audio Transcription CLI Tool requires a robust CLI framework to handle:
- Multiple subcommands (`transcribe`, `extract`, `batch`, `config`)
- Progress display during long-running operations
- Configuration management (API keys, defaults, options)
- Cross-platform compatibility (Linux, macOS, Windows)
- Helpful error messages and auto-generated help text

The team has strong Python experience but varying CLI framework experience. The MVP timeline is 1-3 months, making developer productivity critical.

Two leading Python CLI frameworks were evaluated:
1. **Click** - Mature, widely-adopted CLI framework using decorators
2. **Typer** - Modern CLI framework built on Click, using Python type hints

Both frameworks support the required features (subcommands, help generation, options/arguments). The decision focuses on developer experience, integration with existing stack (particularly Rich for progress display), and long-term maintainability.

## Decision

Use **Typer** as the CLI framework for the Audio Transcription CLI Tool.

## Rationale

### Weighted Scoring Analysis

Based on project priorities (Developer Productivity: 0.35, Maintainability: 0.25, Timeline Risk: 0.25, Ecosystem Fit: 0.15):

| Criterion | Click | Typer | Notes |
|-----------|-------|-------|-------|
| Developer Productivity | 3/5 | 5/5 | Typer: less boilerplate, type hints |
| Maintainability | 4/5 | 5/5 | Type hints improve readability |
| Timeline Risk | 4/5 | 4/5 | Both well-documented |
| Ecosystem Fit | 4/5 | 5/5 | Typer has native Rich support |
| **Weighted Score** | **3.65** | **4.75** | Typer wins on developer experience |

### Key Decision Factors

1. **Type Hints for Arguments**: Typer uses Python type hints to define CLI arguments and options. This matches modern Python practices and leverages the team's Python skills without learning decorator-based patterns.

   ```python
   # Click approach
   @click.command()
   @click.option('--format', type=click.Choice(['txt', 'srt']), default='txt')
   @click.argument('file_path')
   def transcribe(file_path: str, format: str):
       pass

   # Typer approach
   def transcribe(file_path: str, format: str = "txt"):
       pass
   ```

2. **Built-in Rich Integration**: Typer has first-class Rich support (same author, Will McGuigan). This simplifies progress bar implementation, which is critical for batch processing UX.

   ```python
   from rich.progress import Progress
   import typer

   app = typer.Typer(rich_markup_mode="rich")  # Native integration
   ```

3. **Less Boilerplate**: Typer requires fewer decorators and configuration. For a tool with multiple subcommands, this reduces code volume and cognitive load.

4. **Click Foundation**: Typer is built on Click, inheriting its stability and battle-tested features. If edge cases require Click-level control, Typer can access underlying Click objects.

5. **Auto-Generated Help**: Both frameworks generate help text, but Typer's type-hint-based approach produces cleaner, more consistent help output with less effort.

6. **MVP Timeline Advantage**: Typer's simpler syntax enables faster initial development, supporting the 1-3 month timeline.

### Why Not Click Directly?

- **More boilerplate**: Decorator stacking for options/arguments adds visual noise
- **Separate type definitions**: Types defined in decorators, duplicating type hints
- **Rich integration requires extra code**: Must manually configure Rich progress with Click
- **Steeper learning curve**: Decorator patterns less intuitive than function signatures

Click remains a valid choice and would work well. Typer is preferred for its modern approach that aligns with Python best practices the team already follows.

## Consequences

### Positive

- **Faster development**: Less boilerplate accelerates feature implementation
- **Better type safety**: Type hints enable mypy checking of CLI code
- **Simpler progress integration**: Rich works natively with Typer
- **Improved maintainability**: Type hints make code self-documenting
- **Team familiarity**: Type hint patterns match team's Python experience
- **Click compatibility**: Can use Click features when needed

### Negative

- **Newer library**: Typer (2019) is younger than Click (2014); smaller community
- **Abstraction leakage**: Complex cases may require understanding Click underneath
- **Documentation gaps**: Some advanced patterns better documented for Click
- **Dependency chain**: Typer depends on Click, adding indirect dependency

### Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Typer limitations for edge cases | Low | Medium | Typer exposes Click internals; can drop down when needed |
| Smaller community/fewer examples | Medium | Low | Click documentation applies; active GitHub community |
| Breaking changes in Typer | Low | Medium | Pin version, Typer follows semver |
| Team learning curve | Low | Low | Type hint syntax is intuitive; good documentation |

## Alternatives Rejected

### Click (Direct Usage)

**Rejected because**: More boilerplate, separate type definitions from type hints, requires manual Rich integration. Team would need to learn decorator patterns rather than leveraging existing Python skills.

**When to reconsider**: If Typer encounters significant bugs or limitations that block progress. If team has strong existing Click experience from other projects.

### argparse (Standard Library)

**Rejected because**: Too low-level for subcommand-heavy CLI. Would require significant boilerplate for help generation, argument parsing, and validation. No built-in Rich integration.

**When to reconsider**: If project needs to minimize dependencies (e.g., embedded distribution scenarios).

### Fire (Google)

**Rejected because**: Too magical; converts functions to CLI automatically but with less control over help text, validation, and argument types. Poor fit for user-facing tool requiring polished UX.

## Implementation Guidance

### Project Structure

```python
# src/transcribe_cli/cli/main.py
import typer
from rich.console import Console

app = typer.Typer(
    name="transcribe",
    help="Audio Transcription CLI Tool",
    rich_markup_mode="rich",
    add_completion=True,
)
console = Console()

# Subcommands
from .commands import transcribe, extract, batch, config
app.add_typer(transcribe.app, name="transcribe")
app.add_typer(extract.app, name="extract")
app.add_typer(batch.app, name="batch")
app.add_typer(config.app, name="config")
```

### Command Implementation Pattern

```python
# src/transcribe_cli/cli/commands/transcribe.py
import typer
from pathlib import Path
from typing import Optional
from enum import Enum

class OutputFormat(str, Enum):
    txt = "txt"
    srt = "srt"

app = typer.Typer(help="Transcribe audio files")

@app.command()
def file(
    input_file: Path = typer.Argument(
        ...,
        help="Audio or video file to transcribe",
        exists=True,
        readable=True,
    ),
    format: OutputFormat = typer.Option(
        OutputFormat.txt,
        "--format", "-f",
        help="Output format",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path (default: input file with new extension)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
):
    """Transcribe a single audio or video file."""
    # Implementation here
    pass
```

### Progress Integration

```python
from rich.progress import Progress, SpinnerColumn, TextColumn

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    transient=True,
) as progress:
    task = progress.add_task("Transcribing...", total=None)
    result = await transcribe_file(input_file)
    progress.update(task, completed=True)
```

### Error Handling

```python
import typer

def handle_error(error: Exception):
    """Display user-friendly error messages."""
    console.print(f"[red]Error:[/red] {error}")
    raise typer.Exit(code=1)
```

### Entry Point Configuration

```toml
# pyproject.toml
[project.scripts]
transcribe = "transcribe_cli.cli.main:app"
```

## Testing CLI Commands

```python
from typer.testing import CliRunner
from transcribe_cli.cli.main import app

runner = CliRunner()

def test_transcribe_help():
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "Transcribe audio files" in result.stdout

def test_transcribe_file():
    result = runner.invoke(app, ["transcribe", "file", "tests/fixtures/sample.mp3"])
    assert result.exit_code == 0
```

## Related Decisions

- ADR-001: FFmpeg Integration Approach (CLI invokes FFmpeg wrapper)
- ADR-002: Batch Processing Concurrency Model (CLI exposes concurrency options)
- ADR-003: Output Format Support (CLI --format option)

## References

- [Typer Documentation](https://typer.tiangolo.com/)
- [Click Documentation](https://click.palletsprojects.com/)
- [Rich Library](https://rich.readthedocs.io/)
- [FastAPI/Typer Tutorial Pattern](https://typer.tiangolo.com/tutorial/)
- SAD Section 7.1: Technology Stack
- SAD OQ-001: CLI Framework decision item
