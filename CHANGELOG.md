# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Speaker diarization now targets pyannote.audio 4.x and the
  `pyannote/speaker-diarization-community-1` model (pyannote 3.x is no longer supported).
- Audio for diarization is decoded in-process, so m4a and video inputs work
  without pyannote's own audio backend.

### Fixed
- Pin PyAV below 18: faster-whisper passes `metadata_errors` to `av.open()`,
  which PyAV 18+ removed, breaking all local transcription.
- Re-running `postprocess` with an edited `speakers.json` now applies the new
  names (mapping keys on the original diarizer label).
- Hugging Face token falls back to a saved `hf auth login` when `HF_TOKEN` is unset.
- Python 3.9/3.10: fall back to `tomli` when `tomllib` is unavailable (config loading crashed on import).
- `dev` extra now includes `openai`, which the test suite imports.
- README and CLI help showed `transcribe <file>`; the command is `transcribe transcribe <file>`.

### Added
- README section documenting the diarization and speaker-naming workflow.
- MIT `LICENSE` file (the package already declared MIT in `pyproject.toml`).

### Infrastructure
- CI workflow moved to `.gitea/workflows/` so it runs only on Gitea; the public GitHub mirror runs no Actions.

## [0.1.0] - 2024-12-04

### Added
- Initial release of transcribe-cli
- **Audio Transcription**: Transcribe MP3, WAV, FLAC, AAC, M4A files using OpenAI Whisper API
- **Video Support**: Extract and transcribe audio from MKV, MP4, AVI, MOV, WEBM files
- **Batch Processing**: Process entire directories with concurrent API calls
- **Multiple Output Formats**: Plain text (TXT) and subtitles (SRT)
- **FFmpeg Integration**: Automatic audio extraction from video files
- **Configuration System**:
  - TOML config file support (`transcribe.toml`, `.transcriberc`)
  - Environment variable configuration
  - CLI flag overrides
- **CLI Commands**:
  - `transcribe <file>` - Transcribe single audio/video file
  - `transcribe batch <directory>` - Batch process directory
  - `transcribe extract <file>` - Extract audio from video
  - `transcribe config` - Manage configuration
- **Batch Options**:
  - `--dry-run` - Preview files without processing
  - `--recursive` - Scan subdirectories
  - `--concurrency` - Control parallel API calls (1-20)
- **Progress Display**: Rich progress bars for batch processing
- **Error Handling**: Retry logic with exponential backoff for API rate limits

### Technical Details
- Built with Typer CLI framework
- Pydantic for configuration validation
- Async batch processing with semaphore concurrency control
- 174 tests with 78%+ code coverage
- Supports Python 3.9, 3.10, 3.11, 3.12
- Cross-platform: Linux, macOS, Windows

[0.1.0]: https://github.com/jmagly/transcribe-cli/releases/tag/v0.1.0
