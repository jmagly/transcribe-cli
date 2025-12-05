"""Main CLI entry point for transcribe-cli.

Implements ADR-004: CLI Framework Selection (Typer)
- Type-hint based argument parsing
- Rich integration for progress display
- Subcommands for different operations
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from transcribe_cli import __version__

app = typer.Typer(
    name="transcribe",
    help="Transcribe audio and video files using OpenAI Whisper API.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def version_callback(value: bool) -> None:
    """Display version and exit."""
    if value:
        console.print(f"transcribe-cli version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Audio Transcription CLI Tool.

    Transcribe audio and video files using OpenAI Whisper API.
    Supports batch processing, multiple output formats, and large files.
    """
    pass


@app.command()
def transcribe(
    file: Path = typer.Argument(
        ...,
        help="Audio or video file to transcribe.",
        exists=True,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for transcript. Defaults to current directory.",
    ),
    format: str = typer.Option(
        "txt",
        "--format",
        "-f",
        help="Output format: txt, srt",
    ),
    language: str = typer.Option(
        "auto",
        "--language",
        "-l",
        help="Language code (e.g., 'en', 'es') or 'auto' for detection.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Transcribe a single audio or video file.

    Examples:
        transcribe audio.mp3
        transcribe video.mkv --format srt
        transcribe recording.wav --output-dir ./transcripts
    """
    from transcribe_cli.core import (
        APIKeyMissingError,
        FFmpegNotFoundError,
        FileTooLargeError,
        TranscriptionError,
        UnsupportedFormatError,
        get_media_info,
        is_video_file,
        save_transcript,
        transcribe_file,
    )

    # Validate output format
    if format not in ("txt", "srt"):
        console.print(f"[red]Error:[/red] Unsupported format '{format}'. Use 'txt' or 'srt'.")
        raise typer.Exit(1)

    # SRT format not yet implemented
    if format == "srt":
        console.print("[yellow]SRT format will be available in a future release.[/yellow]")
        console.print("[yellow]Using TXT format for now.[/yellow]")
        format = "txt"

    try:
        # Show file info if verbose
        if verbose:
            console.print(f"[dim]Analyzing: {file}[/dim]")
            try:
                info = get_media_info(file)
                console.print(f"[dim]  Format: {info.format_name}[/dim]")
                console.print(f"[dim]  Duration: {info.duration_display}[/dim]")
                if info.has_audio:
                    console.print(f"[dim]  Audio codec: {info.audio_codec}[/dim]")
                if is_video_file(file):
                    console.print(f"[dim]  Type: Video (audio will be extracted)[/dim]")
            except Exception:
                pass  # Don't fail on info gathering

        console.print(f"[bold blue]Transcribing:[/bold blue] {file}")

        # Determine output path
        output_path = None
        if output_dir:
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{file.stem}.{format}"

        # Perform transcription
        with console.status("[bold green]Transcribing...[/bold green]"):
            result = transcribe_file(
                input_path=file,
                output_path=output_path,
                language=language,
            )

        # Save transcript
        saved_path = save_transcript(result, output_path)

        console.print(f"[green]Success![/green] Transcript saved to: {saved_path}")
        if verbose:
            console.print(f"[dim]  Language: {result.language}[/dim]")
            console.print(f"[dim]  Words: {result.word_count}[/dim]")
            if result.duration:
                console.print(f"[dim]  Duration: {result.duration:.1f}s[/dim]")

    except APIKeyMissingError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except FileTooLargeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except FFmpegNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except UnsupportedFormatError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except TranscriptionError as e:
        console.print(f"[red]Transcription failed:[/red] {e}")
        raise typer.Exit(1)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def extract(
    file: Path = typer.Argument(
        ...,
        help="Video file to extract audio from.",
        exists=True,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output audio file path.",
    ),
    format: str = typer.Option(
        "mp3",
        "--format",
        "-f",
        help="Output audio format: mp3, wav",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Extract audio from a video file without transcribing.

    Examples:
        transcribe extract video.mkv
        transcribe extract video.mp4 --output audio.mp3
        transcribe extract video.avi --format wav
    """
    from transcribe_cli.core import (
        ExtractionError,
        FFmpegNotFoundError,
        FFmpegVersionError,
        NoAudioStreamError,
        UnsupportedFormatError,
        extract_audio,
        get_media_info,
    )

    # Validate format
    if format not in ("mp3", "wav"):
        console.print(f"[red]Error:[/red] Unsupported format '{format}'. Use 'mp3' or 'wav'.")
        raise typer.Exit(1)

    try:
        # Show file info if verbose
        if verbose:
            console.print(f"[dim]Analyzing: {file}[/dim]")
            info = get_media_info(file)
            console.print(f"[dim]  Format: {info.format_name}[/dim]")
            console.print(f"[dim]  Duration: {info.duration_display}[/dim]")
            console.print(f"[dim]  Audio codec: {info.audio_codec}[/dim]")

        console.print(f"[bold blue]Extracting audio from:[/bold blue] {file}")

        # Perform extraction
        result = extract_audio(
            input_path=file,
            output_path=output,
            output_format=format,  # type: ignore
        )

        console.print(f"[green]Success![/green] Audio extracted to: {result.output_path}")
        if verbose:
            console.print(f"[dim]  Size: {result.file_size_display}[/dim]")
            if result.duration:
                console.print(f"[dim]  Duration: {result.duration:.1f}s[/dim]")

    except FFmpegNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except FFmpegVersionError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except UnsupportedFormatError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except NoAudioStreamError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except ExtractionError as e:
        console.print(f"[red]Extraction failed:[/red] {e}")
        raise typer.Exit(1)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def batch(
    directory: Path = typer.Argument(
        ...,
        help="Directory containing audio/video files.",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for transcripts.",
    ),
    format: str = typer.Option(
        "txt",
        "--format",
        "-f",
        help="Output format: txt, srt",
    ),
    concurrency: int = typer.Option(
        5,
        "--concurrency",
        "-c",
        help="Maximum concurrent transcriptions (1-20).",
        min=1,
        max=20,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Batch transcribe all audio/video files in a directory.

    Examples:
        transcribe batch ./recordings
        transcribe batch ./videos --format srt --concurrency 3
    """
    console.print(f"[bold blue]Batch processing:[/bold blue] {directory}")
    # TODO: Implement batch logic in Sprint 4
    console.print("[yellow]Batch processing not yet implemented.[/yellow]")


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show",
        help="Show current configuration.",
    ),
) -> None:
    """Manage configuration settings.

    Examples:
        transcribe config --show
    """
    if show:
        console.print("[bold]Current Configuration:[/bold]")
        console.print("  OPENAI_API_KEY: [dim](set via environment)[/dim]")
        console.print("  Output format: txt")
        console.print("  Concurrency: 5")
        # TODO: Load and display actual settings
    else:
        console.print("Use --show to display current configuration.")
        console.print("Set OPENAI_API_KEY environment variable for API access.")


if __name__ == "__main__":
    app()
