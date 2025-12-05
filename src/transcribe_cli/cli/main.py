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

    from transcribe_cli.output import save_formatted_transcript

    # Validate output format
    if format not in ("txt", "srt"):
        console.print(f"[red]Error:[/red] Unsupported format '{format}'. Use 'txt' or 'srt'.")
        raise typer.Exit(1)

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
        else:
            output_path = file.with_suffix(f".{format}")

        # Perform transcription
        with console.status("[bold green]Transcribing...[/bold green]"):
            result = transcribe_file(
                input_path=file,
                output_path=output_path,
                language=language,
            )

        # Save transcript with formatter (supports SRT)
        saved_path = save_formatted_transcript(result, output_path, format)  # type: ignore

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
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively scan subdirectories.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview files without processing (no API calls).",
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
        transcribe batch ./media --recursive --dry-run
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    from transcribe_cli.core import (
        APIKeyMissingError,
        process_directory,
        scan_directory,
    )

    # Validate output format
    if format not in ("txt", "srt"):
        console.print(f"[red]Error:[/red] Unsupported format '{format}'. Use 'txt' or 'srt'.")
        raise typer.Exit(1)

    # Scan directory first to show file count
    try:
        files = scan_directory(directory, recursive=recursive)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if not files:
        console.print(f"[yellow]No audio/video files found in:[/yellow] {directory}")
        if recursive:
            console.print("[dim]  (searched recursively)[/dim]")
        raise typer.Exit(0)

    # Calculate total size for display
    total_size = sum(f.stat().st_size for f in files)
    size_mb = total_size / (1024 * 1024)

    console.print(f"[bold blue]Batch processing:[/bold blue] {directory}")
    console.print(f"[dim]Found {len(files)} file(s) ({size_mb:.1f} MB total)[/dim]")
    if recursive:
        console.print("[dim]  (recursive scan)[/dim]")

    # Dry run mode - show files and exit
    if dry_run:
        console.print()
        console.print("[bold yellow]DRY RUN[/bold yellow] - No files will be processed")
        console.print()
        for f in files:
            file_size = f.stat().st_size / (1024 * 1024)
            rel_path = f.relative_to(directory) if recursive else f.name
            console.print(f"  [dim]{rel_path}[/dim] ({file_size:.2f} MB)")
        console.print()
        console.print(f"[dim]Would process {len(files)} files with concurrency {concurrency}[/dim]")
        console.print(f"[dim]Output format: {format}[/dim]")
        raise typer.Exit(0)

    try:
        console.print(f"[dim]Concurrency: {concurrency}[/dim]")

        if verbose:
            for f in files:
                console.print(f"[dim]  - {f.name}[/dim]")

        # Track progress
        completed = 0
        failed_files: list[tuple[Path, str]] = []

        def progress_callback(path: Path, status: str) -> None:
            nonlocal completed
            if status == "completed":
                completed += 1
            elif status == "failed":
                completed += 1

        # Process with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[green]Processing {len(files)} files...",
                total=len(files),
            )

            # Custom callback to update progress
            def update_progress(path: Path, status: str) -> None:
                progress_callback(path, status)
                if status in ("completed", "failed"):
                    progress.update(task, advance=1)

            # Run batch processing
            summary = process_directory(
                directory=directory,
                output_dir=output_dir,
                output_format=format,  # type: ignore
                concurrency=concurrency,
                recursive=recursive,
                progress_callback=update_progress,
            )

        # Show summary
        console.print()
        console.print("[bold]Batch Processing Complete[/bold]")
        console.print(f"  [green]Successful:[/green] {summary.successful}")
        console.print(f"  [red]Failed:[/red] {summary.failed}")
        console.print(f"  [dim]Total:[/dim] {summary.total_files}")

        if summary.failed > 0:
            console.print()
            console.print("[bold red]Failed files:[/bold red]")
            for result in summary.results:
                if not result.success:
                    console.print(f"  [red]✗[/red] {result.input_path.name}")
                    if verbose and result.error:
                        console.print(f"    [dim]{result.error}[/dim]")

        if summary.failed > 0:
            raise typer.Exit(1)

    except APIKeyMissingError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(
        False,
        "--show",
        help="Show current configuration.",
    ),
    init: bool = typer.Option(
        False,
        "--init",
        help="Create a default config file in current directory.",
    ),
    locations: bool = typer.Option(
        False,
        "--locations",
        help="Show config file search locations.",
    ),
) -> None:
    """Manage configuration settings.

    Examples:
        transcribe config --show
        transcribe config --init
        transcribe config --locations
    """
    import os

    from transcribe_cli.config import (
        create_default_config,
        find_config_file,
        get_config_locations,
        get_settings,
    )

    if init:
        config_path = Path.cwd() / "transcribe.toml"
        if config_path.exists():
            console.print(f"[yellow]Config file already exists:[/yellow] {config_path}")
            raise typer.Exit(1)
        created = create_default_config(config_path)
        console.print(f"[green]Created config file:[/green] {created}")
        raise typer.Exit(0)

    if locations:
        console.print("[bold]Config file locations (checked in order):[/bold]")
        active_config = find_config_file()
        for loc in get_config_locations():
            if loc == active_config:
                console.print(f"  [green]✓[/green] {loc} [dim](active)[/dim]")
            elif loc.exists():
                console.print(f"  [yellow]•[/yellow] {loc} [dim](exists)[/dim]")
            else:
                console.print(f"  [dim]•[/dim] {loc}")
        raise typer.Exit(0)

    if show:
        console.print("[bold]Current Configuration:[/bold]")
        console.print()

        # Show config file status
        active_config = find_config_file()
        if active_config:
            console.print(f"  [bold]Config file:[/bold] {active_config}")
        else:
            console.print("  [bold]Config file:[/bold] [dim]none (using defaults)[/dim]")
        console.print()

        # Show API key status
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            console.print(f"  [bold]OPENAI_API_KEY:[/bold] {masked}")
        else:
            console.print("  [bold]OPENAI_API_KEY:[/bold] [red]not set[/red]")
        console.print()

        # Show defaults
        console.print("  [bold]Defaults:[/bold]")
        console.print("    Output format: txt")
        console.print("    Concurrency: 5")
        console.print("    Language: auto")
        console.print("    Recursive: false")
    else:
        console.print("Use [bold]--show[/bold] to display current configuration.")
        console.print("Use [bold]--init[/bold] to create a config file.")
        console.print("Use [bold]--locations[/bold] to see config file search paths.")
        console.print()
        console.print("[dim]Set OPENAI_API_KEY environment variable for API access.[/dim]")


if __name__ == "__main__":
    app()
