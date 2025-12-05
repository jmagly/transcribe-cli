# Project Intake Form

**Document Type**: Greenfield Project
**Generated**: 2025-12-04
**Source**: Project description + interactive user responses

## Metadata

- **Project name**: Audio Transcription CLI Tool
- **Requestor/owner**: Engineering Team
- **Date**: 2025-12-04
- **Stakeholders**: Engineering (development and maintenance), Product (feature prioritization), Engineering Team Members (primary users)

## System Overview

**Purpose**: A command-line tool for extracting audio from video files (MKV format) and transcribing audio in common formats (MP3, AAC, FLAC, etc.) using OpenAI Whisper API. Designed to streamline transcription workflows for team productivity.

**Current Status**: Planning (new project)

**Users**: Small team (2-10 people), internal engineering team members

**Tech Stack** (proposed):
- **Language**: Python 3.9+ (rich ecosystem for audio/video processing)
- **Core Libraries**:
  - `ffmpeg-python` or `pydub`: Audio extraction from MKV and format conversion
  - `openai`: Whisper API integration for transcription
  - `click` or `typer`: Modern CLI framework with argument parsing
  - `rich`: Enhanced terminal output with progress bars
  - `pydantic`: Configuration validation
- **Audio/Video Processing**: FFmpeg (external dependency, widely available)
- **Deployment**: Python package (pip installable), distributed as CLI tool
- **Testing**: pytest for unit/integration tests
- **CI/CD**: GitHub Actions for automated testing and releases

## Problem and Outcomes

**Problem Statement**: Team members currently lack an efficient, standardized way to transcribe audio from video files (meeting recordings, interviews, lectures in MKV format) and standalone audio files. Manual transcription is time-consuming and inconsistent. Existing tools require multiple steps (extract audio manually, upload to service, download transcript) leading to workflow friction.

**Target Personas**:
- **Primary**: Engineering team members needing to transcribe meeting recordings, technical interviews, or video content for documentation and reference
- **Secondary**: Content creators or researchers processing podcasts, lectures, or interview recordings in the team

**Success Metrics (KPIs)**:
- **Team adoption**: 80% of team (6-8 users out of 10) using the tool regularly for transcription tasks within 2 months
- **Time savings**: Reduce transcription workflow time by 70% compared to manual process (from ~30 min setup to <5 min automated)
- **Ease of use**: Single command execution for transcription (`transcribe audio.mp3` or `transcribe video.mkv --extract-audio`)
- **Accuracy**: Transcription accuracy >90% (leveraging Whisper API's quality)
- **Reliability**: Successfully process 95%+ of submitted files without errors

## Current Scope and Features

**Core Features** (in-scope for first release):

1. **Audio Extraction from MKV**:
   - Extract audio track from MKV video files
   - Support multiple audio codec formats (AAC, MP3, FLAC embedded in MKV)
   - Preserve audio quality during extraction

2. **Direct Audio Transcription**:
   - Transcribe common audio formats: MP3, AAC, FLAC, WAV, M4A
   - Integration with OpenAI Whisper API
   - Automatic file format detection and conversion if needed

3. **Batch Processing**:
   - Process multiple files in a single command
   - Directory/folder processing (transcribe all files in folder)
   - Progress indicators for batch operations

4. **Large File Handling**:
   - Support files >1GB (long recordings, 2+ hour podcasts/lectures)
   - Automatic chunking if required by API limits
   - Resume support for interrupted transcriptions

5. **Output Formats**:
   - Plain text (.txt) transcripts
   - Timestamped transcripts (SRT, VTT, JSON formats)
   - Speaker identification (if supported by Whisper API)
   - AI-generated summaries/key points from transcript

6. **Configuration & Usability**:
   - API key management (environment variable or config file)
   - Configurable output directory
   - Verbose/quiet modes for debugging or automation
   - Progress bars and status updates

**Out-of-Scope** (explicitly excluded for now, may revisit later):

- Real-time transcription (streaming audio/video)
- Video subtitle embedding (SRT overlay directly in video files)
- Custom model training or fine-tuning
- GUI/web interface (CLI only for MVP)
- Multi-language translation (transcription only, no translation)
- Local Whisper model support (API-based only for MVP)
- Cloud storage integration (Google Drive, Dropbox upload)

**Future Considerations** (post-MVP, if project succeeds):

- Add support for local Whisper model (whisper.cpp) for offline use
- Cloud storage integration (Google Drive, S3) for batch processing
- Web dashboard for job status and transcript management
- Speaker diarization enhancements (custom speaker labeling)
- Multi-language detection and translation
- Video subtitle generation and embedding
- Integration with note-taking tools (Notion, Obsidian)

## Architecture (Proposed)

**Architecture Style**: Simple CLI Application (Monolith)

**Chosen**: **Simple CLI Monolith** - **Rationale**: Small team (2-10 users), straightforward workflow (extract → transcribe → output), single responsibility (transcription), minimal operational overhead. Python's rich ecosystem for audio/video processing makes this the most pragmatic choice for fast iteration and ease of maintenance.

**Components** (proposed):

1. **CLI Entry Point** (`cli.py`):
   - Argument parsing (click/typer framework)
   - Command routing: `transcribe`, `extract`, `batch`
   - Configuration loading (API keys, defaults)
   - **Technology**: Python click/typer
   - **Rationale**: Modern CLI framework with auto-generated help, subcommands, argument validation

2. **Audio Extraction Module** (`extractors/audio_extractor.py`):
   - FFmpeg wrapper for MKV audio extraction
   - Format detection and conversion
   - Audio codec handling (AAC, MP3, FLAC)
   - **Technology**: ffmpeg-python or subprocess calls to FFmpeg
   - **Rationale**: FFmpeg is industry-standard for audio/video processing, widely available

3. **Transcription Module** (`transcribers/whisper_client.py`):
   - OpenAI Whisper API integration
   - File chunking for large files (>25MB API limit)
   - Retry logic and error handling
   - **Technology**: openai Python SDK
   - **Rationale**: Official SDK, automatic retries, well-documented

4. **Output Formatter** (`formatters/transcript_formatter.py`):
   - Plain text output
   - Timestamped formats (SRT, VTT, JSON)
   - Speaker identification formatting
   - Summary generation (optional OpenAI GPT call)
   - **Technology**: Python string formatting, json module, srt library
   - **Rationale**: Simple text processing, multiple format support

5. **Batch Processor** (`processors/batch_processor.py`):
   - Directory scanning and file discovery
   - Parallel processing (asyncio or concurrent.futures)
   - Progress tracking (rich progress bars)
   - Error aggregation and reporting
   - **Technology**: Python asyncio + rich
   - **Rationale**: Async for I/O-bound API calls, rich for beautiful CLI output

6. **Configuration Manager** (`config/settings.py`):
   - Environment variable loading (OPENAI_API_KEY)
   - Config file support (.transcriberc or YAML)
   - Validation (pydantic)
   - **Technology**: pydantic + python-dotenv
   - **Rationale**: Type-safe configuration, easy validation

**Data Models** (estimated):

1. **TranscriptionJob**:
   - `file_path`: str (input file)
   - `file_type`: str (audio, video, format)
   - `output_path`: str (transcript destination)
   - `status`: str (pending, processing, completed, failed)
   - `transcript`: Optional[str] (result)
   - `metadata`: dict (duration, language, speaker_count)

2. **TranscriptionConfig**:
   - `api_key`: str (OpenAI API key)
   - `output_dir`: Path (default: ./transcripts)
   - `output_formats`: List[str] (txt, srt, vtt, json)
   - `language`: Optional[str] (auto-detect or specify)
   - `enable_summary`: bool (generate summaries)

3. **AudioFile**:
   - `path`: Path (file location)
   - `format`: str (mp3, flac, aac, mkv)
   - `duration`: float (seconds)
   - `size`: int (bytes)
   - `sample_rate`: int (Hz)

**Integration Points**:

- **OpenAI Whisper API**: RESTful API for audio transcription
  - Endpoint: `https://api.openai.com/v1/audio/transcriptions`
  - Authentication: Bearer token (API key)
  - Rate limits: Consider API quotas, implement backoff

- **FFmpeg**: External system command for audio extraction
  - Required installation: `ffmpeg` binary in PATH
  - Subprocess calls via Python

## Scale and Performance (Target)

**Target Capacity**:
- **Initial users**: 2-10 team members (small engineering team)
- **6-month projection**: 10-20 users (expanded to other departments if successful)
- **2-year vision**: Revisit post-MVP (could expand to open-source if valuable)

**Performance Targets**:
- **Latency**:
  - Audio extraction: <30 seconds for 1-hour MKV video
  - Transcription: <5 minutes for 1-hour audio (depends on Whisper API)
  - Total workflow: <6 minutes for 1-hour MKV → transcript
- **Throughput**:
  - Batch processing: Handle 10 files concurrently (limited by API rate limits)
  - Sustained: ~12 hours of audio transcribed per hour (parallel processing)
- **Availability**: Best-effort (CLI tool, no uptime SLA)
  - Depends on OpenAI API availability (~99.9%)
  - Graceful error handling and retry logic

**Performance Strategy**:
- **Async Processing**: Use Python asyncio for concurrent API calls in batch mode
- **File Chunking**: Split large files (>25MB) into chunks for API limits
- **Caching**: Cache transcripts to avoid re-processing (hash-based file tracking)
- **Progress Feedback**: Real-time progress bars (rich library) for user experience
- **Retry Logic**: Exponential backoff for API failures (openai SDK handles this)
- **Resource Management**: Temp file cleanup, memory-efficient streaming for large files

## Security and Compliance (Requirements)

**Security Posture**: **Baseline**

**Chosen**: Baseline - **Rationale**: Internal team tool, no storage of PII (audio files and transcripts remain on user's local machine), API key is user-managed (environment variable). Primary risks are API key exposure and dependency vulnerabilities.

**Data Classification**: **Internal**

**Identified**: Internal - **Evidence**: Audio files and transcripts handled locally by team members. No centralized storage or PII collection. Audio content is team-generated (meetings, interviews) and remains on user's filesystem. OpenAI API processes audio temporarily but does not retain it (per Whisper API terms).

**Security Controls** (required):

1. **API Key Management**:
   - Store API keys in environment variables (OPENAI_API_KEY)
   - Support for `.env` file (never committed to git, in .gitignore)
   - Config file option with restrictive permissions (0600)
   - Clear documentation: Never hardcode API keys

2. **Dependency Security**:
   - Use `pip-audit` or `safety` to scan for vulnerable dependencies
   - Pin dependency versions in `requirements.txt`
   - Regular updates for security patches
   - Minimal dependency footprint

3. **Input Validation**:
   - Validate file paths (prevent directory traversal)
   - File type validation (check magic bytes, not just extensions)
   - File size limits (warn for extremely large files)
   - Sanitize output filenames

4. **Data Protection**:
   - HTTPS for all OpenAI API communication (enforced by SDK)
   - No logging of API keys or sensitive content
   - Temp file cleanup after processing
   - Option to delete source audio after transcription (user choice)

5. **Secrets Management**:
   - Environment variables for MVP (OPENAI_API_KEY)
   - Document secure practices in README:
     - Use `.env` file (never commit)
     - Use OS keychain integration (future enhancement)
     - Rotate API keys periodically

**Compliance Requirements**: **None** (no regulatory requirements)

**Identified**: None - **Rationale**: Internal tool, no PII storage, no regulatory industry. Audio files are team-generated content. OpenAI Whisper API is GDPR-compliant for processing, but we're not storing user data centrally.

**Security Best Practices**:
- Include SBOM (Software Bill of Materials) via `pip freeze`
- Automated security scanning in CI (GitHub Dependabot)
- Clear security documentation in README
- Responsible disclosure policy if open-sourced

## Team and Operations (Planned)

**Team Size**: Small team (2-5 developers for initial development, 2-10 users total)

**Team Skills**:
- **Python**: Strong (primary development language)
- **CLI Tools**: Familiar (using click/typer, standard Python tooling)
- **Audio/Video Processing**: Learning (FFmpeg, codecs - will document)
- **API Integration**: Strong (OpenAI SDK, REST APIs)
- **DevOps**: Moderate (GitHub Actions for CI/CD, pip packaging)

**Development Velocity** (target):
- **Sprint length**: 2 weeks (agile, iterative development)
- **Release frequency**:
  - MVP: 1-3 months (initial release)
  - Post-MVP: Bi-weekly releases (bug fixes, minor features)
  - Stable: Monthly releases (maintenance mode)

**Process Maturity** (planned):

1. **Version Control**:
   - Git with feature branches (GitHub/GitLab)
   - Conventional commits (semantic versioning)
   - Tag releases (v0.1.0, v0.2.0, v1.0.0)

2. **Code Review**:
   - Pull requests required for main branch
   - Single reviewer approval (small team)
   - Automated checks: linting, tests, security scans

3. **Testing**:
   - **Target coverage**: 60% for MVP (core logic: extraction, transcription, formatting)
   - **Unit tests**: pytest for business logic
   - **Integration tests**: End-to-end CLI tests (click.testing.CliRunner)
   - **Manual testing**: Audio file samples (various formats, sizes)

4. **CI/CD**:
   - **GitHub Actions**:
     - Lint: black, flake8, mypy (type checking)
     - Test: pytest with coverage report
     - Security: pip-audit for dependency scanning
     - Build: Package verification (pip install .)
   - **Automated releases**: GitHub Releases + PyPI publishing (manual for MVP)

5. **Documentation**:
   - **README**: Installation, usage examples, configuration, troubleshooting
   - **API docs**: Docstrings (Google style), auto-generated with Sphinx (optional)
   - **CHANGELOG**: Keep updated with releases
   - **Runbook**: Troubleshooting guide for common issues (FFmpeg not found, API errors)

**Operational Support** (planned):

1. **Monitoring**:
   - **Logs**: Structured logging to stdout (JSON format for parsing)
   - **Metrics**: None for MVP (CLI tool, no telemetry)
   - **Errors**: Clear error messages with troubleshooting hints

2. **Logging**:
   - Python `logging` module with configurable levels
   - Default: INFO level (progress updates)
   - Debug mode: --verbose flag for troubleshooting
   - No centralized logging (local CLI tool)

3. **Alerting**:
   - None for MVP (users see errors in terminal)
   - Email notifications: Optional future enhancement for batch job completion

4. **On-call**:
   - None (internal tool, best-effort support)
   - Bug reports via GitHub Issues
   - Team members self-support with documentation

## Dependencies and Infrastructure

**Third-Party Services** (required):

1. **OpenAI Whisper API**:
   - **Purpose**: Audio transcription (core functionality)
   - **Cost**: Pay-per-use ($0.006/minute of audio as of 2024)
   - **Estimated cost**: ~$5-20/month for team usage (10-50 hours of audio)
   - **API**: RESTful, official Python SDK
   - **Risk**: API downtime or rate limits (mitigation: retry logic, clear error messages)

**Third-Party Services** (optional/future):

- **OpenAI GPT API**: Summary generation (optional feature)
- **Cloud Storage**: Google Drive, S3 for batch processing (future)

**Infrastructure** (proposed):

- **Hosting**: N/A (CLI tool, runs locally on user's machine)
- **Development Environment**:
  - Python 3.9+ (support 3.9-3.12)
  - FFmpeg installed (system dependency)
  - Git for version control
- **Distribution**:
  - **PyPI**: Python Package Index (pip install transcribe-cli)
  - **GitHub Releases**: Binary distributions (optional, via PyInstaller)
  - **Package managers**: Consider Homebrew (macOS), apt/yum (Linux) for future

**External Dependencies**:

- **FFmpeg**: Required system installation
  - Installation guide for Linux, macOS, Windows
  - Version check on CLI startup with helpful error if missing
  - Minimum version: 4.0+ (widely available)

**Python Dependencies** (estimated):

```
openai>=1.0.0          # Whisper API client
ffmpeg-python>=0.2.0   # FFmpeg wrapper
click>=8.0.0           # CLI framework
rich>=13.0.0           # Terminal output formatting
pydantic>=2.0.0        # Configuration validation
python-dotenv>=1.0.0   # .env file support
srt>=3.5.0             # SRT subtitle format
pytest>=7.0.0          # Testing framework
black>=23.0.0          # Code formatting
flake8>=6.0.0          # Linting
mypy>=1.0.0            # Type checking
pip-audit>=2.0.0       # Security scanning
```

## Known Risks and Uncertainties

**Technical Risks**:

1. **FFmpeg Installation Complexity**:
   - **Description**: Users may not have FFmpeg installed or in PATH, causing extraction to fail
   - **Likelihood**: High (especially Windows users)
   - **Impact**: Medium (blocks core functionality)
   - **Mitigation**:
     - Clear installation documentation (README with platform-specific guides)
     - Startup check with helpful error message: "FFmpeg not found. Install from: https://ffmpeg.org/download.html"
     - Consider bundling FFmpeg binaries (licensing complexity)

2. **Large File Handling (>1GB)**:
   - **Description**: Processing very large files may hit API limits, memory constraints, or timeout issues
   - **Likelihood**: Medium (2+ hour recordings are common)
   - **Impact**: High (user frustration, failed transcriptions)
   - **Mitigation**:
     - Implement file chunking (split audio into <25MB segments for API)
     - Progress indicators and checkpointing (resume interrupted jobs)
     - Document recommended file sizes and expected processing times
     - Test with 2-3 hour sample files

3. **Whisper API Rate Limits**:
   - **Description**: OpenAI may rate-limit requests during batch processing or high usage
   - **Likelihood**: Low-Medium (depends on team usage patterns)
   - **Impact**: Medium (delays, failed batch jobs)
   - **Mitigation**:
     - Exponential backoff retry logic (openai SDK handles this)
     - Configurable concurrency limit (default: 5 concurrent requests)
     - Clear error messages when rate-limited with retry suggestions

4. **Audio Format Compatibility**:
   - **Description**: Unusual codecs or corrupted files may fail extraction or transcription
   - **Likelihood**: Medium (real-world files have format quirks)
   - **Impact**: Medium (specific files fail, not all)
   - **Mitigation**:
     - Extensive format testing (create test suite with varied samples)
     - Graceful error handling with format detection
     - FFmpeg conversion fallback (convert to WAV before transcription)
     - Document supported formats and known limitations

**Integration Risks**:

1. **OpenAI API Changes**:
   - **Description**: Whisper API may change pricing, rate limits, or behavior
   - **Likelihood**: Low (stable API, versioned)
   - **Impact**: Medium-High (cost increases, breaking changes)
   - **Mitigation**:
     - Use official SDK (automatic updates)
     - Pin SDK version in requirements.txt (test before upgrading)
     - Monitor OpenAI changelog and announcements

2. **OpenAI API Reliability**:
   - **Description**: Temporary outages or degraded performance
   - **Likelihood**: Low (~99.9% uptime typically)
   - **Impact**: Medium (delays, user frustration)
   - **Mitigation**:
     - Retry logic with exponential backoff
     - Clear error messages: "Whisper API unavailable. Check status: https://status.openai.com"
     - Offline mode with local Whisper (future enhancement)

**Timeline Risks**:

1. **Scope Creep**:
   - **Description**: 1-3 month timeline with comprehensive feature set (extraction, transcription, batch, multiple formats)
   - **Likelihood**: Medium (feature requests during development)
   - **Impact**: High (delays, rushed quality)
   - **Mitigation**:
     - Strict MVP scope: Core features only (extract MKV, transcribe audio, txt + SRT output)
     - Defer nice-to-have features: Speaker ID, summaries, VTT, JSON (v2)
     - 2-week sprint reviews to track progress and adjust scope

2. **FFmpeg Learning Curve**:
   - **Description**: Team may need time to learn FFmpeg options, codecs, audio processing
   - **Likelihood**: Medium (if team unfamiliar with multimedia processing)
   - **Impact**: Low-Medium (slower initial development)
   - **Mitigation**:
     - Use ffmpeg-python library (abstracts complexity)
     - Reference documentation and examples
     - Allocate 1-2 sprints for audio extraction proof-of-concept

**Team Risks**:

1. **Small Team Capacity**:
   - **Description**: 2-5 developers with other priorities may slow development
   - **Likelihood**: Medium (typical for internal tools)
   - **Impact**: Medium (timeline extension)
   - **Mitigation**:
     - Part-time allocation (20-40% capacity) is OK for 1-3 month timeline
     - Prioritize ruthlessly: Core features first, polish later
     - Leverage existing libraries (ffmpeg-python, openai SDK) to reduce custom code

2. **Python Packaging Expertise**:
   - **Description**: Team may lack experience with pip packaging, distribution, versioning
   - **Likelihood**: Low-Medium
   - **Impact**: Low (delays distribution, but doesn't block development)
   - **Mitigation**:
     - Use modern tools: Poetry or setuptools with pyproject.toml
     - Reference Python packaging guide: https://packaging.python.org
     - Start with local installation (pip install -e .) for team testing
     - Defer PyPI publishing until v1.0

## Why This Intake Now?

**Context**: Starting new project to improve team productivity around transcription workflows. Team currently uses manual or ad-hoc methods (online services, separate tools) which are time-consuming and inconsistent. This intake establishes requirements, architecture, and scope before development starts to ensure alignment and structured progress.

**Goals**:
- Establish clear feature scope (extraction, transcription, batch processing) to avoid scope creep
- Align team on architecture (Python CLI with FFmpeg + Whisper API) and technology choices
- Identify risks early (FFmpeg installation, large file handling, API rate limits)
- Enable structured SDLC process (Inception → Elaboration → Construction → Transition)
- Set measurable success criteria (80% team adoption, 70% time savings)

**Triggers**:
- New project kickoff (greenfield development)
- Team need for standardized transcription workflow
- Planning phase before Sprint 1 begins
- Seeking SDLC structure for organized development and quality

## Attachments

- Solution profile: `.aiwg/intake/solution-profile.md`
- Option matrix: `.aiwg/intake/option-matrix.md`

## Next Steps

**Your intake documents are now complete and ready for the next phase!**

1. **Review** generated intake files for accuracy
2. **Proceed directly to Inception** using natural language or explicit commands:
   - Natural language: "Start Inception" or "Let's transition to Inception"
   - Explicit command: `/flow-concept-to-inception .`

**Note**: You do NOT need to run `/intake-start` - that command is only for teams who manually created their own intake documents. The `intake-wizard` produces validated intake ready for immediate use.
