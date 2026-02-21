"""Main CLI entry point for transcribe-cli.

Implements ADR-004: CLI Framework Selection (Typer)
- Type-hint based argument parsing
- Rich integration for progress display
- Subcommands for different operations

Supports both local GPU inference (faster-whisper) and OpenAI API backends.
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional

import typer
from rich.console import Console

from transcribe_cli import __version__

app = typer.Typer(
    name="transcribe",
    help="Transcribe audio and video files using local GPU or OpenAI API.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


def _resolve_whisperx_command() -> list[str]:
    """Resolve whisperx executable command."""
    whisperx_bin = shutil.which("whisperx")
    if whisperx_bin:
        return [whisperx_bin]
    return [sys.executable, "-m", "whisperx"]


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

    Transcribe audio and video files using local GPU (faster-whisper) or OpenAI API.
    Supports batch processing, multiple output formats, speaker diarization, and large files.
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
        help="Output format: txt, srt, diarized",
    ),
    language: str = typer.Option(
        "auto",
        "--language",
        "-l",
        help="Language code (e.g., 'en', 'es') or 'auto' for detection.",
    ),
    # Backend selection
    backend: str = typer.Option(
        "local",
        "--backend",
        "-b",
        help="Transcription backend: local (GPU/CPU) or api (OpenAI).",
    ),
    # Local inference options
    model: str = typer.Option(
        "medium",
        "--model",
        "-m",
        help="Whisper model size: tiny, base, small, medium, large-v3",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device for local inference: cuda, cpu, auto",
    ),
    compute_type: str = typer.Option(
        "auto",
        "--compute-type",
        help="Compute type: int8, float16, float32, auto (int8 faster on older GPUs)",
    ),
    # Diarization options
    diarize: bool = typer.Option(
        False,
        "--diarize",
        "-d",
        help="Enable speaker diarization (requires HF_TOKEN).",
    ),
    postprocess_outputs: bool = typer.Option(
        True,
        "--postprocess/--no-postprocess",
        help="Generate speaker config, relabeled outputs, and reports when diarizing.",
    ),
    min_speakers: Optional[int] = typer.Option(
        None,
        "--min-speakers",
        help="Minimum number of speakers (optional hint for diarization).",
    ),
    max_speakers: Optional[int] = typer.Option(
        None,
        "--max-speakers",
        help="Maximum number of speakers (optional hint for diarization).",
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
        transcribe recording.wav --model large-v3
        transcribe meeting.mp4 --diarize --format diarized
        transcribe audio.mp3 --backend api  # Use OpenAI API
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
        # Local transcriber
        LocalTranscriberConfig,
        LocalTranscriptionError,
        transcribe_file_local,
        get_gpu_info,
        # Diarization
        DiarizationConfig,
        DiarizationError,
        DiarizationNotAvailable,
        transcribe_and_diarize,
        # Postprocess
        apply_speaker_mapping,
        generate_reports,
        generate_speaker_config,
        load_speaker_config,
        segments_from_whisperx_json,
        write_diarized_outputs,
    )

    from transcribe_cli.output import format_transcript, save_formatted_transcript

    # Validate backend
    if backend not in ("local", "api"):
        console.print(f"[red]Error:[/red] Unsupported backend '{backend}'. Use 'local' or 'api'.")
        raise typer.Exit(1)

    # Validate output format
    valid_formats = ("txt", "srt", "diarized")
    if format not in valid_formats:
        console.print(f"[red]Error:[/red] Unsupported format '{format}'. Use: {', '.join(valid_formats)}")
        raise typer.Exit(1)

    # Diarized format requires diarization
    if format == "diarized" and not diarize:
        console.print("[yellow]Note:[/yellow] Enabling diarization for diarized output format.")
        diarize = True

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

            # Show GPU info for local backend
            if backend == "local":
                try:
                    gpu_info = get_gpu_info()
                    if gpu_info.get("cuda_available"):
                        console.print(f"[dim]  GPU: {gpu_info.get('device_name', 'Unknown')}[/dim]")
                        console.print(f"[dim]  VRAM: {gpu_info.get('vram_total_gb', 0):.1f} GB[/dim]")
                    else:
                        console.print("[dim]  GPU: Not available (using CPU)[/dim]")
                except Exception:
                    pass

        backend_display = "local GPU" if backend == "local" else "OpenAI API"
        console.print(f"[bold blue]Transcribing ({backend_display}):[/bold blue] {file}")

        # Determine output path
        output_ext = format if format != "diarized" else "txt"
        output_path = None
        if output_dir:
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{file.stem}.{output_ext}"
        else:
            output_path = file.with_suffix(f".{output_ext}")

        # Perform transcription based on backend
        if backend == "local":
            if diarize:
                # Use combined transcription + diarization
                with console.status(f"[bold green]Transcribing with {model} model + diarization...[/bold green]"):
                    transcriber_config = LocalTranscriberConfig(
                        model_size=model,  # type: ignore
                        device=device,  # type: ignore
                        compute_type=compute_type,  # type: ignore
                    )
                    diarization_config = DiarizationConfig(
                        device=device,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                    )
                    result = transcribe_and_diarize(
                        input_path=file,
                        output_path=output_path,
                        language=language,
                        transcriber_config=transcriber_config,
                        diarization_config=diarization_config,
                    )
            else:
                # Local transcription only
                with console.status(f"[bold green]Transcribing with {model} model...[/bold green]"):
                    config = LocalTranscriberConfig(
                        model_size=model,  # type: ignore
                        device=device,  # type: ignore
                        compute_type=compute_type,  # type: ignore
                    )
                    result = transcribe_file_local(
                        input_path=file,
                        output_path=output_path,
                        language=language,
                        config=config,
                    )
        else:
            # API backend (original behavior)
            if diarize:
                console.print("[yellow]Warning:[/yellow] Diarization not supported with API backend. Using local diarization.")
                # Transcribe via API, then diarize locally
                with console.status("[bold green]Transcribing via API...[/bold green]"):
                    result = transcribe_file(
                        input_path=file,
                        output_path=output_path,
                        language=language,
                    )
            else:
                with console.status("[bold green]Transcribing via API...[/bold green]"):
                    result = transcribe_file(
                        input_path=file,
                        output_path=output_path,
                        language=language,
                    )

        # Save transcript with appropriate format
        formatted_content = format_transcript(result, format)  # type: ignore
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(formatted_content, encoding="utf-8")

        console.print(f"[green]Success![/green] Transcript saved to: {output_path}")
        if verbose:
            console.print(f"[dim]  Language: {result.language}[/dim]")
            if hasattr(result, 'word_count'):
                console.print(f"[dim]  Words: {result.word_count}[/dim]")
            if result.duration:
                console.print(f"[dim]  Duration: {result.duration:.1f}s[/dim]")
            if diarize and hasattr(result, 'speakers'):
                console.print(f"[dim]  Speakers: {', '.join(result.speakers)}[/dim]")

        if diarize and postprocess_outputs:
            if hasattr(result, "segments") and hasattr(result, "speakers"):
                post_dir = output_path.parent / "postprocess"
                post_dir.mkdir(parents=True, exist_ok=True)
                speaker_config = post_dir / "speakers.json"

                if not speaker_config.exists():
                    generate_speaker_config(speaker_config, file, list(result.speakers))
                    console.print(f"[green]Generated speaker config:[/green] {speaker_config}")

                mapping = load_speaker_config(speaker_config)
                segments = []
                for seg in result.segments:
                    segments.append(
                        {
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text,
                            "speaker": getattr(seg, "speaker", None),
                        }
                    )
                data = {
                    "segments": segments,
                    "word_segments": [],
                    "language": result.language,
                }

                updated = apply_speaker_mapping(data, mapping)
                updated_segments = segments_from_whisperx_json(updated)
                outputs = write_diarized_outputs(output_path.stem, post_dir, updated, updated_segments)
                reports_dir = post_dir / "reports"
                generate_reports(updated_segments, reports_dir, file)

                console.print(f"[green]Postprocess complete:[/green] {post_dir}")
                if verbose:
                    for label, path in outputs.items():
                        console.print(f"[dim]  {label}: {path}[/dim]")
                    console.print(f"[dim]  reports: {reports_dir}[/dim]")
            else:
                console.print("[yellow]Note:[/yellow] Postprocess skipped (no diarized segments available).")

    except APIKeyMissingError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("[dim]Set OPENAI_API_KEY or use --backend local[/dim]")
        raise typer.Exit(1)
    except LocalTranscriptionError as e:
        console.print(f"[red]Local transcription failed:[/red] {e}")
        raise typer.Exit(1)
    except DiarizationNotAvailable as e:
        console.print(f"[red]Diarization not available:[/red] {e}")
        raise typer.Exit(1)
    except DiarizationError as e:
        console.print(f"[red]Diarization failed:[/red] {e}")
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
def whisperx(
    file: Path = typer.Argument(
        ...,
        help="Audio or video file to transcribe with WhisperX.",
        exists=True,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        "-o",
        help="Output directory for WhisperX artifacts.",
    ),
    format: str = typer.Option(
        "srt",
        "--format",
        "-f",
        help="WhisperX output format: txt, srt, vtt, json.",
    ),
    model: str = typer.Option(
        "small",
        "--model",
        "-m",
        help="WhisperX model size/name.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Device: cuda or cpu.",
    ),
    compute_type: str = typer.Option(
        "float16",
        "--compute-type",
        help="Compute type for WhisperX model execution.",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Optional language code (e.g., en, es).",
    ),
) -> None:
    """Run WhisperX directly as a thin wrapper command."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_formats = ("txt", "srt", "vtt", "json")
    if format not in valid_formats:
        console.print(f"[red]Error:[/red] Unsupported format '{format}'. Use: {', '.join(valid_formats)}")
        raise typer.Exit(1)

    cmd = _resolve_whisperx_command()
    cmd.extend(
        [
            str(file),
            "--output_dir",
            str(output_dir),
            "--output_format",
            format,
            "--model",
            model,
            "--device",
            device,
            "--compute_type",
            compute_type,
        ]
    )

    if language:
        cmd.extend(["--language", language])

    console.print(f"[bold blue]Running WhisperX:[/bold blue] {file}")
    with console.status("[bold green]Executing WhisperX...[/bold green]"):
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        if "No module named whisperx" in err or "command not found" in err:
            console.print("[red]Error:[/red] WhisperX is not installed.")
            console.print("[dim]Install with: pip install whisperx[/dim]")
        else:
            console.print("[red]WhisperX failed:[/red]")
            console.print(err if err else "[dim]No error output captured[/dim]")
        raise typer.Exit(1)

    console.print(f"[green]Success![/green] WhisperX outputs saved to: {output_dir}")
    if result.stdout.strip():
        console.print(f"[dim]{result.stdout.strip()}[/dim]")


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
def postprocess(
    whisperx_json: Path = typer.Argument(
        ...,
        help="WhisperX diarized JSON to post-process.",
        exists=True,
        readable=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory for updated speaker outputs and reports.",
    ),
    speaker_config: Optional[Path] = typer.Option(
        None,
        "--speaker-config",
        help="Speaker mapping JSON. If missing, a template will be generated.",
    ),
    top_terms: int = typer.Option(
        20,
        "--top-terms",
        help="Top terms to include in analysis summary.",
    ),
) -> None:
    """Post-process a diarized WhisperX JSON.

    Generates a speaker config template (if needed), applies speaker labels,
    writes updated diarized outputs, and generates reports.
    """
    from transcribe_cli.core import (
        apply_speaker_mapping,
        extract_speaker_labels,
        generate_reports,
        generate_speaker_config,
        load_speaker_config,
        load_whisperx_json,
        segments_from_whisperx_json,
        write_diarized_outputs,
    )

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_whisperx_json(whisperx_json)
    labels = extract_speaker_labels(data)

    if not speaker_config:
        speaker_config = output_dir / "speakers.json"

    if not speaker_config.exists():
        generate_speaker_config(speaker_config, whisperx_json, labels)
        console.print(f"[green]Generated speaker config:[/green] {speaker_config}")

    mapping = load_speaker_config(speaker_config)
    updated = apply_speaker_mapping(data, mapping)

    segments = segments_from_whisperx_json(updated)
    base_name = Path(whisperx_json).stem

    outputs = write_diarized_outputs(base_name, output_dir, updated, segments)
    reports_dir = output_dir / "reports"
    generate_reports(segments, reports_dir, whisperx_json, top_n=top_terms)

    console.print("[green]Post-process complete.[/green]")
    for label, path in outputs.items():
        console.print(f"  {label}: {path}")
    console.print(f"  reports: {reports_dir}")


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

        # Show local backend status
        console.print("  [bold]Local Backend (faster-whisper):[/bold]")
        try:
            from transcribe_cli.core import get_gpu_info, get_available_models
            gpu_info = get_gpu_info()
            if gpu_info.get("cuda_available"):
                console.print(f"    [green]✓[/green] GPU: {gpu_info.get('device_name', 'Unknown')}")
                console.print(f"    [green]✓[/green] VRAM: {gpu_info.get('vram_total_gb', 0):.1f} GB")
            else:
                console.print("    [yellow]⚠[/yellow] GPU: Not available (CPU fallback)")
            models = get_available_models()
            console.print(f"    [dim]Available models: {', '.join(models)}[/dim]")
        except ImportError:
            console.print("    [red]✗[/red] faster-whisper not installed")
            console.print("    [dim]Install: pip install transcribe-cli[local][/dim]")
        console.print()

        # Show diarization status
        console.print("  [bold]Diarization (pyannote):[/bold]")
        try:
            import pyannote.audio
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                masked = hf_token[:8] + "..." if len(hf_token) > 8 else "***"
                console.print(f"    [green]✓[/green] HF_TOKEN: {masked}")
            else:
                console.print("    [yellow]⚠[/yellow] HF_TOKEN: [dim]not set[/dim]")
                console.print("    [dim]Required for speaker diarization[/dim]")
        except ImportError:
            console.print("    [dim]Not installed (optional)[/dim]")
            console.print("    [dim]Install: pip install transcribe-cli[diarization][/dim]")
        console.print()

        # Show API key status
        console.print("  [bold]API Backend (OpenAI):[/bold]")
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            console.print(f"    [green]✓[/green] OPENAI_API_KEY: {masked}")
        else:
            console.print("    [dim]OPENAI_API_KEY: not set (local backend is default)[/dim]")
        console.print()

        # Show defaults
        console.print("  [bold]Defaults:[/bold]")
        console.print("    Backend: local")
        console.print("    Model: medium")
        console.print("    Output format: txt")
        console.print("    Language: auto")
    else:
        console.print("Use [bold]--show[/bold] to display current configuration.")
        console.print("Use [bold]--init[/bold] to create a config file.")
        console.print("Use [bold]--locations[/bold] to see config file search paths.")
        console.print()
        console.print("[dim]Local GPU inference is the default backend.[/dim]")
        console.print("[dim]Set HF_TOKEN for speaker diarization.[/dim]")


@app.command()
def setup(
    check: bool = typer.Option(
        False,
        "--check",
        help="Check if all dependencies are installed.",
    ),
    install_ffmpeg: bool = typer.Option(
        False,
        "--install-ffmpeg",
        help="Attempt to install FFmpeg automatically.",
    ),
) -> None:
    """Check and install required dependencies.

    Examples:
        transcribe setup --check
        transcribe setup --install-ffmpeg
    """
    import shutil
    import subprocess
    import sys

    from transcribe_cli.core import validate_ffmpeg

    if check or (not install_ffmpeg):
        console.print("[bold]Dependency Check[/bold]")
        console.print()

        # Check FFmpeg
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")

        ffmpeg_healthy = False
        if ffmpeg_path and ffprobe_path:
            try:
                info = validate_ffmpeg()
                console.print(f"  [green]✓[/green] FFmpeg {info.version_display} ({ffmpeg_path})")
                ffmpeg_healthy = True
            except Exception as e:
                console.print(f"  [yellow]⚠[/yellow] FFmpeg found but has issues: {e}")
        else:
            console.print("  [red]✗[/red] FFmpeg not found")
            if sys.platform.startswith("linux"):
                console.print("    [dim]Install: sudo apt install ffmpeg[/dim]")
            elif sys.platform == "darwin":
                console.print("    [dim]Install: brew install ffmpeg[/dim]")
            elif sys.platform == "win32":
                console.print("    [dim]Install: choco install ffmpeg[/dim]")

        # Check Python version
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 9):
            console.print(f"  [green]✓[/green] Python {py_version}")
        else:
            console.print(f"  [red]✗[/red] Python {py_version} (requires 3.9+)")

        # Check local transcription backend
        console.print()
        console.print("[bold]Local Transcription:[/bold]")
        try:
            from transcribe_cli.core import get_gpu_info
            gpu_info = get_gpu_info()
            if gpu_info.get("cuda_available"):
                console.print(f"  [green]✓[/green] GPU: {gpu_info.get('device_name', 'Unknown')}")
                console.print(f"  [green]✓[/green] CUDA: Available")
                console.print(f"  [dim]    VRAM: {gpu_info.get('vram_total_gb', 0):.1f} GB[/dim]")
            else:
                console.print("  [yellow]⚠[/yellow] GPU: Not available")
                console.print("  [dim]    Will use CPU (slower)[/dim]")
        except ImportError:
            console.print("  [red]✗[/red] faster-whisper not installed")
            console.print("  [dim]    Install: pip install transcribe-cli[local][/dim]")

        # Check diarization
        console.print()
        console.print("[bold]Diarization (optional):[/bold]")
        try:
            import pyannote.audio
            console.print("  [green]✓[/green] pyannote-audio installed")
            import os
            if os.environ.get("HF_TOKEN"):
                console.print("  [green]✓[/green] HF_TOKEN is set")
            else:
                console.print("  [yellow]⚠[/yellow] HF_TOKEN not set")
                console.print("  [dim]    Required for speaker diarization[/dim]")
                console.print("  [dim]    Get token: https://huggingface.co/settings/tokens[/dim]")
        except ImportError:
            console.print("  [dim]Not installed (optional feature)[/dim]")
            console.print("  [dim]    Install: pip install transcribe-cli[diarization][/dim]")

        # Check WhisperX wrapper runtime
        console.print()
        console.print("[bold]WhisperX (optional/minimal wrapper backend):[/bold]")
        whisperx_bin = shutil.which("whisperx")
        if whisperx_bin:
            console.print(f"  [green]✓[/green] whisperx executable: {whisperx_bin}")
        else:
            try:
                import whisperx  # type: ignore
                console.print("  [green]✓[/green] whisperx python module available")
            except ImportError:
                console.print("  [dim]Not installed (optional feature)[/dim]")
                console.print("  [dim]    Install: pip install whisperx[/dim]")

        # Check API key (optional now)
        console.print()
        console.print("[bold]API Backend (optional):[/bold]")
        import os
        if os.environ.get("OPENAI_API_KEY"):
            console.print("  [green]✓[/green] OPENAI_API_KEY is set")
        else:
            console.print("  [dim]OPENAI_API_KEY not set (use --backend api if needed)[/dim]")

        console.print()

        if not check and not install_ffmpeg:
            console.print("Use [bold]--check[/bold] to verify dependencies.")
            console.print("Use [bold]--install-ffmpeg[/bold] to install FFmpeg.")

        if not ffmpeg_healthy:
            raise typer.Exit(1)

    if install_ffmpeg:
        console.print("[bold]Installing FFmpeg...[/bold]")
        console.print()

        if sys.platform.startswith("linux"):
            # Detect package manager
            if shutil.which("apt-get"):
                cmd = ["sudo", "apt-get", "update"]
                cmd_install = ["sudo", "apt-get", "install", "-y", "ffmpeg"]
                console.print("[dim]Using apt-get...[/dim]")
            elif shutil.which("dnf"):
                cmd = None
                cmd_install = ["sudo", "dnf", "install", "-y", "ffmpeg"]
                console.print("[dim]Using dnf...[/dim]")
            elif shutil.which("pacman"):
                cmd = None
                cmd_install = ["sudo", "pacman", "-S", "--noconfirm", "ffmpeg"]
                console.print("[dim]Using pacman...[/dim]")
            else:
                console.print("[red]Error:[/red] No supported package manager found.")
                console.print("[dim]Please install FFmpeg manually.[/dim]")
                raise typer.Exit(1)

            try:
                if cmd:
                    subprocess.run(cmd, check=True)
                subprocess.run(cmd_install, check=True)
                console.print("[green]FFmpeg installed successfully![/green]")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Installation failed:[/red] {e}")
                raise typer.Exit(1)

        elif sys.platform == "darwin":
            if shutil.which("brew"):
                console.print("[dim]Using Homebrew...[/dim]")
                try:
                    subprocess.run(["brew", "install", "ffmpeg"], check=True)
                    console.print("[green]FFmpeg installed successfully![/green]")
                except subprocess.CalledProcessError as e:
                    console.print(f"[red]Installation failed:[/red] {e}")
                    raise typer.Exit(1)
            else:
                console.print("[red]Error:[/red] Homebrew not found.")
                console.print("[dim]Install Homebrew first: https://brew.sh[/dim]")
                raise typer.Exit(1)

        elif sys.platform == "win32":
            if shutil.which("choco"):
                console.print("[dim]Using Chocolatey...[/dim]")
                try:
                    subprocess.run(["choco", "install", "ffmpeg", "-y"], check=True)
                    console.print("[green]FFmpeg installed successfully![/green]")
                except subprocess.CalledProcessError as e:
                    console.print(f"[red]Installation failed:[/red] {e}")
                    raise typer.Exit(1)
            elif shutil.which("scoop"):
                console.print("[dim]Using Scoop...[/dim]")
                try:
                    subprocess.run(["scoop", "install", "ffmpeg"], check=True)
                    console.print("[green]FFmpeg installed successfully![/green]")
                except subprocess.CalledProcessError as e:
                    console.print(f"[red]Installation failed:[/red] {e}")
                    raise typer.Exit(1)
            else:
                console.print("[red]Error:[/red] No supported package manager found.")
                console.print("[dim]Install Chocolatey or Scoop, or download FFmpeg manually.[/dim]")
                raise typer.Exit(1)

        else:
            console.print(f"[red]Error:[/red] Unsupported platform: {sys.platform}")
            console.print("[dim]Please install FFmpeg manually from https://ffmpeg.org[/dim]")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
