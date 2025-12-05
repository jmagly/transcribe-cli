# transcribe-cli

CLI tool for transcribing audio and video files using OpenAI Whisper API.

## Features

- **Audio Transcription**: Transcribe MP3, WAV, FLAC, AAC, M4A files
- **Video Support**: Extract and transcribe audio from MKV, MP4, AVI, MOV
- **Batch Processing**: Process entire directories with concurrent API calls
- **Multiple Output Formats**: Plain text (TXT) and subtitles (SRT)
- **Large File Support**: Automatic chunking for files >25MB
- **Resume Support**: Continue interrupted transcriptions

## Requirements

- Python 3.9+
- FFmpeg 4.0+
- OpenAI API key

## Installation

### 1. Install FFmpeg

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update && sudo apt install ffmpeg -y
```

**macOS (Homebrew)**:
```bash
brew install ffmpeg
```

**Windows (Chocolatey)**:
```bash
choco install ffmpeg -y
```

### 2. Install transcribe-cli

```bash
pip install transcribe-cli
```

Or install from source:
```bash
git clone https://github.com/jmagly/transcribe-cli.git
cd transcribe-cli
pip install -e .
```

### 3. Configure API Key

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Quick Start

```bash
# Transcribe a single file
transcribe audio.mp3

# Transcribe video (extracts audio automatically)
transcribe video.mkv

# Output as SRT subtitles
transcribe audio.mp3 --format srt

# Batch process a directory
transcribe batch ./recordings

# Extract audio only (no transcription)
transcribe extract video.mkv
```

## Usage

### Transcribe Command

```bash
transcribe <file> [OPTIONS]

Options:
  -o, --output-dir PATH   Output directory (default: current)
  -f, --format TEXT       Output format: txt, srt (default: txt)
  -l, --language TEXT     Language code or 'auto' (default: auto)
  --verbose               Enable verbose output
  --help                  Show help message
```

### Batch Command

```bash
transcribe batch <directory> [OPTIONS]

Options:
  -o, --output-dir PATH   Output directory
  -f, --format TEXT       Output format: txt, srt
  -c, --concurrency INT   Max concurrent jobs (1-20, default: 5)
  --verbose               Enable verbose output
  --help                  Show help message
```

### Extract Command

```bash
transcribe extract <file> [OPTIONS]

Options:
  -o, --output PATH       Output audio file path
  --verbose               Enable verbose output
  --help                  Show help message
```

## Configuration

Settings can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `TRANSCRIBE_OUTPUT_DIR` | Default output directory | `.` |
| `TRANSCRIBE_FORMAT` | Default output format | `txt` |
| `TRANSCRIBE_CONCURRENCY` | Max concurrent jobs | `5` |
| `TRANSCRIBE_LANGUAGE` | Default language | `auto` |

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/jmagly/transcribe-cli.git
cd transcribe-cli

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src/transcribe_cli --cov-report=html

# Run linting
black src tests
flake8 src tests
mypy src
```

### Project Structure

```
src/transcribe_cli/
├── cli/          # CLI commands (Typer)
├── config/       # Configuration management
├── core/         # Audio extraction, transcription
├── output/       # Output formatters (TXT, SRT)
├── models/       # Data models
└── utils/        # Utilities
```

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `pytest` and `pre-commit run --all-files`
5. Submit a pull request

## Acknowledgments

- [OpenAI Whisper](https://openai.com/research/whisper) for speech recognition
- [FFmpeg](https://ffmpeg.org/) for audio/video processing
- [Typer](https://typer.tiangolo.com/) for CLI framework
