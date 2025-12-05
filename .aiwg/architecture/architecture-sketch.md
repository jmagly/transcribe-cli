# Architecture Sketch: Audio Transcription CLI Tool

**Document Type**: Architecture Sketch (Inception Phase)
**Version**: 1.0
**Created**: 2025-12-04
**Author**: Architecture Designer Agent
**Status**: DRAFT - Pending Review

---

## 1. Architecture Overview

### Style: Simple CLI Monolith (Modular Design)

The Audio Transcription CLI Tool adopts a **Simple CLI Monolith** architecture with a modular internal structure. This is a single-process application distributed as a Python package that runs locally on the user's machine.

### Rationale

| Factor | Context | Decision Driver |
|--------|---------|-----------------|
| **Team Size** | 2-5 developers | Simple architecture reduces coordination overhead |
| **User Base** | 2-10 team members (internal) | No need for distributed systems or multi-tenancy |
| **Workflow** | Linear pipeline: Extract -> Transcribe -> Format | Single responsibility, straightforward data flow |
| **Operational Overhead** | Minimal (CLI tool, runs locally) | No servers to maintain, no deployment infrastructure |
| **Timeline** | 1-3 months MVP | Monolith enables faster iteration than microservices |
| **Scale Expectations** | <20 users, <100 files/day | No horizontal scaling needed |

### Architecture Characteristics

| Characteristic | Priority | Notes |
|----------------|----------|-------|
| **Simplicity** | High | Single codebase, single deployment unit |
| **Modularity** | High | Clean separation of concerns enables testing and future refactoring |
| **Portability** | High | Runs on Linux, macOS, Windows with Python 3.9+ |
| **Extensibility** | Medium | Plugin-style modules for future format support |
| **Performance** | Medium | Async I/O for batch processing, but not latency-critical |

---

## 2. Component Diagram

```text
                           USER
                             |
                             v
    +-------------------------------------------------------------+
    |                    CLI Entry Point                          |
    |                    (cli.py - click/typer)                   |
    +-----------+-------------------+-------------------+---------+
    | transcribe|      extract      |      batch        | config  |
    +-----------+-------------------+-------------------+---------+
                             |
          +------------------+------------------+
          |                  |                  |
          v                  v                  v
    +-----------+    +---------------+    +-----------+
    |   Audio   |    | Transcription |    |  Output   |
    | Extractor |    |    Client     |    | Formatter |
    | Module    |    |    Module     |    |  Module   |
    +-----------+    +---------------+    +-----------+
          |                  |                  |
          v                  v                  v
    +-----------+    +---------------+    +-----------+
    |  FFmpeg   |    | OpenAI Whisper|    |   File    |
    | (external)|    |     API       |    |  System   |
    +-----------+    +---------------+    +-----------+


    +-------------------------------------------------------------+
    |                    Supporting Modules                        |
    +-----------+-------------------+-------------------+---------+
    |   Config  |      Batch        |     Progress      |  Utils  |
    |  Manager  |    Processor      |     Tracker       |         |
    +-----------+-------------------+-------------------+---------+
          |
          v
    +-----------+
    | .env /    |
    | Config    |
    | Files     |
    +-----------+
```

### Detailed Component Architecture

```text
+--------------------------------------------------------------------+
|                        transcribe-cli                               |
+--------------------------------------------------------------------+
|                                                                     |
|  +-----------------+    +-----------------+    +-----------------+  |
|  |   cli/          |    |   core/         |    |   output/       |  |
|  |                 |    |                 |    |                 |  |
|  | - main.py       |    | - extractor.py  |    | - formatter.py  |  |
|  | - commands/     |    | - transcriber.py|    | - txt.py        |  |
|  |   - transcribe  |    | - chunker.py    |    | - srt.py        |  |
|  |   - extract     |    | - processor.py  |    | - vtt.py        |  |
|  |   - batch       |    |                 |    | - json.py       |  |
|  |   - config      |    |                 |    |                 |  |
|  +-----------------+    +-----------------+    +-----------------+  |
|                                                                     |
|  +-----------------+    +-----------------+    +-----------------+  |
|  |   config/       |    |   utils/        |    |   models/       |  |
|  |                 |    |                 |    |                 |  |
|  | - settings.py   |    | - file_utils.py |    | - job.py        |  |
|  | - validators.py |    | - progress.py   |    | - config.py     |  |
|  | - defaults.py   |    | - logging.py    |    | - audio.py      |  |
|  |                 |    | - errors.py     |    | - transcript.py |  |
|  +-----------------+    +-----------------+    +-----------------+  |
|                                                                     |
+--------------------------------------------------------------------+
```

---

## 3. Component Descriptions

| Component | Responsibility | Technology | Key Interfaces |
|-----------|---------------|------------|----------------|
| **CLI Entry Point** | Argument parsing, command routing, user interaction | `click` or `typer` | User commands, help text, config loading |
| **Audio Extractor** | FFmpeg wrapper, audio extraction from video, format detection, codec handling | `ffmpeg-python` | File paths, audio streams, codec info, temp files |
| **Transcription Client** | Whisper API integration, request handling, response parsing, chunking for large files | `openai` SDK | Audio bytes, transcript text, timestamps, metadata |
| **Output Formatter** | Format conversion (txt, SRT, VTT, JSON), timestamp formatting, file writing | Python stdlib, `srt` library | Transcript data, formatted output, file paths |
| **Batch Processor** | Directory scanning, file discovery, parallel execution, progress aggregation, error collection | `asyncio`, `rich` | File lists, job queue, progress callbacks |
| **Config Manager** | Environment loading, config file parsing, validation, defaults | `pydantic`, `python-dotenv` | Config dict, validated settings |
| **Progress Tracker** | Real-time progress display, ETA calculation, status updates | `rich` | Progress callbacks, terminal output |
| **Utils** | File operations, logging, error handling, common helpers | Python stdlib | Cross-cutting utilities |

### Component Details

#### CLI Entry Point (`cli/`)
- **Framework**: `click` (mature, widely adopted) or `typer` (modern, type-hint based)
- **Commands**:
  - `transcribe <file>` - Transcribe single audio/video file
  - `extract <file>` - Extract audio only (no transcription)
  - `batch <directory>` - Process all files in directory
  - `config [show|set|reset]` - Configuration management
- **Features**:
  - Auto-generated help text and shell completion
  - Verbose/quiet modes (`--verbose`, `--quiet`)
  - Output format selection (`--format txt|srt|vtt|json`)
  - Output directory (`--output-dir`)

#### Audio Extractor (`core/extractor.py`)
- **Technology**: `ffmpeg-python` library wrapping FFmpeg binary
- **Capabilities**:
  - Extract audio from MKV, MP4, AVI, MOV video files
  - Detect audio codec (AAC, MP3, FLAC, WAV, M4A)
  - Convert to Whisper-compatible format (MP3/WAV)
  - Stream extraction (preserve quality, no re-encoding when possible)
- **Error Handling**:
  - FFmpeg not installed detection
  - Corrupted file handling
  - Unsupported codec fallback

#### Transcription Client (`core/transcriber.py`)
- **Technology**: `openai` Python SDK
- **Capabilities**:
  - Audio transcription via Whisper API
  - Automatic language detection (or user-specified)
  - Timestamp generation (word-level or segment-level)
  - File chunking for >25MB files
- **Error Handling**:
  - Rate limit handling with exponential backoff
  - API error responses
  - Network timeout retry

#### Output Formatter (`output/`)
- **Formats**:
  - **TXT**: Plain text, paragraph-style
  - **SRT**: SubRip subtitle format with timestamps
  - **VTT** (v2): WebVTT for web players
  - **JSON** (v2): Structured data with metadata
- **Features**:
  - Timestamp formatting (HH:MM:SS,mmm)
  - Line length wrapping
  - Speaker identification (if available from API)

#### Batch Processor (`core/processor.py`)
- **Technology**: `asyncio` for concurrent API calls
- **Capabilities**:
  - Directory walking with file type filtering
  - Configurable concurrency (default: 5 simultaneous)
  - Progress tracking per file and overall
  - Error aggregation and final report
  - Resume support (skip already-processed files)

#### Config Manager (`config/`)
- **Technology**: `pydantic` for validation, `python-dotenv` for environment
- **Configuration Sources** (priority order):
  1. Command-line arguments (highest)
  2. Environment variables
  3. Config file (`~/.transcriberc` or `transcribe.yaml`)
  4. Default values (lowest)
- **Key Settings**:
  - `OPENAI_API_KEY` - Required, no default
  - `output_dir` - Default: `./transcripts`
  - `output_format` - Default: `txt`
  - `language` - Default: auto-detect
  - `concurrency` - Default: 5

---

## 4. Data Flow

### Single File Transcription Flow

```text
User: transcribe video.mkv --format srt --output-dir ./output

Step 1: CLI Parsing
  +------+     +----------+     +-----------+
  | Args | --> | CLI      | --> | Command   |
  |      |     | Parser   |     | Router    |
  +------+     +----------+     +-----------+
                                     |
Step 2: Configuration Loading        v
  +------+     +----------+     +-----------+
  | .env | --> | Config   | --> | Validated |
  | file |     | Manager  |     | Settings  |
  +------+     +----------+     +-----------+
                                     |
Step 3: File Type Detection          v
  +--------+     +----------+     +-----------+
  | video  | --> | Type     | --> | VIDEO     |
  | .mkv   |     | Detector |     | (MKV)     |
  +--------+     +----------+     +-----------+
                                     |
Step 4: Audio Extraction             v
  +--------+     +----------+     +-----------+
  | MKV    | --> | Audio    | --> | Temp      |
  | File   |     | Extractor|     | WAV/MP3   |
  +--------+     +----------+     +-----------+
                      |
                      v (FFmpeg subprocess)
               +-----------+
               |  FFmpeg   |
               |  Binary   |
               +-----------+
                                     |
Step 5: Size Check & Chunking        v
  +--------+     +----------+     +-----------+
  | Temp   | --> | Size     | --> | <25MB: OK |
  | Audio  |     | Checker  |     | >25MB:    |
  +--------+     +----------+     | Chunk     |
                                  +-----------+
                                     |
Step 6: Transcription                v
  +--------+     +----------+     +-----------+
  | Audio  | --> | Whisper  | --> | API       |
  | Bytes  |     | Client   |     | Request   |
  +--------+     +----------+     +-----------+
                      |
                      v (HTTPS POST)
               +-----------+
               | OpenAI    |
               | Whisper   |
               | API       |
               +-----------+
                      |
                      v
               +-----------+
               | JSON      |
               | Response  |
               | (text +   |
               | timestamps)|
               +-----------+
                                     |
Step 7: Output Formatting            v
  +----------+     +----------+     +-----------+
  | JSON     | --> | SRT      | --> | Formatted |
  | Response |     | Formatter|     | Subtitles |
  +----------+     +----------+     +-----------+
                                     |
Step 8: File Output                  v
  +----------+     +----------+     +-----------+
  | SRT      | --> | File     | --> | ./output/ |
  | Content  |     | Writer   |     | video.srt |
  +----------+     +----------+     +-----------+
                                     |
Step 9: Cleanup                      v
  +----------+     +----------+     +-----------+
  | Temp     | --> | Cleanup  | --> | Success   |
  | Files    |     | Handler  |     | Message   |
  +----------+     +----------+     +-----------+
```

### Batch Processing Flow

```text
User: batch ./recordings --format txt --concurrency 5

  +------------+     +------------+     +------------+
  | Directory  | --> | File       | --> | File Queue |
  | ./recordings     | Scanner    |     | [f1,f2,..] |
  +------------+     +------------+     +------------+
                                              |
        +-------------------------------------+
        |
        v
  +------------+     +------------+
  | Async      | --> | Worker     | --> [Process f1] --> [Result 1]
  | Task       |     | Pool       |
  | Manager    |     | (5 workers)|
  +------------+     +------------+ --> [Process f2] --> [Result 2]
        |                |
        |                +--------> [Process f3] --> [Result 3]
        |                |
        |                +--------> [Process f4] --> [Result 4]
        |                |
        |                +--------> [Process f5] --> [Result 5]
        |
        v
  +------------+     +------------+
  | Progress   | --> | Terminal   |
  | Tracker    |     | Display    |
  +------------+     +------------+
        |
        v
  +------------+     +------------+
  | Error      | --> | Summary    |
  | Aggregator |     | Report     |
  +------------+     +------------+

  Output: 10 files processed, 8 succeeded, 2 failed (see error log)
```

---

## 5. Technology Stack

### Core Technologies

| Layer | Technology | Version | Rationale |
|-------|------------|---------|-----------|
| **Language** | Python | 3.9+ | Rich ecosystem for audio/video, team familiarity, cross-platform |
| **CLI Framework** | `click` or `typer` | 8.x / 0.9.x | Modern CLI with auto-help, subcommands, validation |
| **Audio Processing** | `ffmpeg-python` | 0.2.x | Pythonic FFmpeg wrapper, abstracts command construction |
| **External Binary** | FFmpeg | 4.0+ | Industry-standard audio/video processing |
| **Transcription** | `openai` SDK | 1.x | Official SDK, automatic retries, well-documented |
| **Progress Display** | `rich` | 13.x | Beautiful terminal output, progress bars, tables |
| **Config Validation** | `pydantic` | 2.x | Type-safe configuration, validation, serialization |
| **Environment** | `python-dotenv` | 1.x | Load .env files for API keys |
| **Subtitle Format** | `srt` | 3.x | SRT format parsing and generation |
| **Async** | `asyncio` | stdlib | I/O-bound batch processing, concurrency |

### Development & Testing

| Tool | Version | Purpose |
|------|---------|---------|
| `pytest` | 7.x | Unit and integration testing |
| `pytest-asyncio` | 0.21+ | Async test support |
| `pytest-cov` | 4.x | Code coverage reporting |
| `black` | 23.x | Code formatting |
| `flake8` | 6.x | Linting |
| `mypy` | 1.x | Type checking |
| `pip-audit` | 2.x | Dependency security scanning |

### Dependency Summary

```text
# requirements.txt (runtime)
openai>=1.0.0
ffmpeg-python>=0.2.0
click>=8.0.0
rich>=13.0.0
pydantic>=2.0.0
python-dotenv>=1.0.0
srt>=3.5.0

# requirements-dev.txt (development)
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0
pip-audit>=2.0.0
```

---

## 6. Key Architectural Decisions

### ADR Summary

| ADR | Decision | Status |
|-----|----------|--------|
| **ADR-001** | Use `ffmpeg-python` over direct subprocess calls | PROPOSED |
| **ADR-002** | Async batch processing with `asyncio` | PROPOSED |
| **ADR-003** | MVP output formats: TXT and SRT | PROPOSED |

#### ADR-001: FFmpeg Integration Approach

**Context**: Need to extract audio from video files (MKV, MP4, etc.) reliably.

**Decision**: Use `ffmpeg-python` library as FFmpeg wrapper.

**Rationale**:
- Pythonic interface matches team skills
- Abstracts complex FFmpeg command construction
- Handles common patterns (format detection, codec conversion)
- Allows dropping down to raw FFmpeg commands for edge cases
- Weighted score: 4.15/5.0 vs 3.70 for direct subprocess

**Alternatives Considered**:
- Direct `subprocess` calls (more control, but slower development)
- `pydub` library (simpler API, but limited for MKV edge cases)

#### ADR-002: Async Batch Processing Strategy

**Context**: Batch processing multiple files requires concurrent API calls.

**Decision**: Use Python `asyncio` for concurrent processing.

**Rationale**:
- I/O-bound operations (API calls, file I/O) benefit from async
- Native Python support (no external dependencies)
- `openai` SDK supports async (`AsyncOpenAI` client)
- Configurable concurrency to respect API rate limits

**Alternatives Considered**:
- `concurrent.futures.ThreadPoolExecutor` (simpler, but less efficient for I/O)
- `multiprocessing` (overkill for I/O-bound tasks)

#### ADR-003: MVP Output Formats

**Context**: Support multiple transcript output formats.

**Decision**: MVP includes TXT and SRT only. VTT and JSON deferred to v2.

**Rationale**:
- TXT: Primary use case (readable transcript for notes, documentation)
- SRT: Required for video subtitles (common request)
- VTT: Similar to SRT, can be derived easily (v2)
- JSON: Structured data for integration (v2)

**Trade-off**: Limits initial functionality but accelerates MVP delivery.

---

## 7. Integration Points

| External System | Protocol | Authentication | Rate Limits | Error Handling |
|----------------|----------|----------------|-------------|----------------|
| **OpenAI Whisper API** | HTTPS REST | Bearer token (API key) | RPM/TPM limits apply | Retry with exponential backoff (SDK handles) |
| **FFmpeg** | Subprocess | N/A (local binary) | N/A | Exit code + stderr parsing |
| **File System** | OS syscalls | User permissions | OS limits | Exception handling, permission checks |

### OpenAI Whisper API Integration

```text
Endpoint: https://api.openai.com/v1/audio/transcriptions
Method: POST
Content-Type: multipart/form-data

Request:
  - file: Audio file (MP3, WAV, M4A, etc.)
  - model: "whisper-1"
  - language: Optional (auto-detect if omitted)
  - response_format: "verbose_json" (for timestamps)

Response (JSON):
  {
    "text": "Full transcript text...",
    "segments": [
      {
        "id": 0,
        "start": 0.0,
        "end": 4.5,
        "text": "Segment text..."
      },
      ...
    ],
    "language": "en"
  }

Error Handling:
  - 429 Too Many Requests: Retry with exponential backoff
  - 400 Bad Request: Invalid file format (log and skip)
  - 500 Server Error: Retry up to 3 times
  - Timeout: Retry with increased timeout
```

### FFmpeg Integration

```text
Usage Pattern:
  ffmpeg -i input.mkv -vn -acodec copy output.aac
  ffmpeg -i input.mkv -vn -acodec pcm_s16le output.wav

Error Handling:
  - Exit code != 0: Parse stderr for error message
  - "No such file": File not found error
  - "Invalid data": Corrupted file handling
  - "Unknown codec": Unsupported format, try conversion
```

---

## 8. Non-Functional Considerations

### Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Audio Extraction** | <30 seconds for 1-hour video | FFmpeg is fast, mostly I/O bound |
| **Transcription** | <5 minutes for 1-hour audio | Whisper API processing time |
| **Total Workflow** | <6 minutes for 1-hour MKV | End-to-end target |
| **Batch Throughput** | 10 files concurrent | Configurable, respect API limits |
| **Startup Time** | <1 second | Fast CLI invocation |

### Reliability Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Success Rate** | 95%+ for valid files | Graceful handling of edge cases |
| **Error Recovery** | Automatic retry (3x) | Transient API/network failures |
| **Resume Support** | Skip processed files | Batch interruption recovery |
| **Data Integrity** | No transcript corruption | Validate output before writing |

### Security Controls

| Control | Implementation | Priority |
|---------|----------------|----------|
| **API Key Protection** | Environment variable, .env file (0600 perms) | HIGH |
| **Input Validation** | File type checking, path sanitization | HIGH |
| **Dependency Security** | `pip-audit` in CI, pinned versions | MEDIUM |
| **Temp File Cleanup** | Automatic cleanup on exit/error | MEDIUM |
| **No Logging Secrets** | API key masking in logs | HIGH |

### Testability Design

| Component | Testing Approach | Coverage Target |
|-----------|-----------------|-----------------|
| CLI | `click.testing.CliRunner` | 60% |
| Audio Extractor | Mock FFmpeg subprocess, sample files | 70% |
| Transcription Client | Mock OpenAI API responses | 80% |
| Output Formatter | Unit tests with sample data | 90% |
| Batch Processor | Mock async operations | 60% |
| Config Manager | Validation edge cases | 80% |

---

## 9. Deployment Architecture

### Distribution Model

```text
+-------------------+     +-------------------+     +-------------------+
|   Development     |     |    Distribution   |     |    User Machine   |
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
| Python source     | --> | PyPI package      | --> | pip install       |
| Git repository    |     | GitHub Release    |     | transcribe-cli    |
|                   |     | (wheel, sdist)    |     |                   |
+-------------------+     +-------------------+     +-------------------+
                                                            |
                                                            v
                                                    +-------------------+
                                                    | Prerequisites     |
                                                    +-------------------+
                                                    | - Python 3.9+     |
                                                    | - FFmpeg binary   |
                                                    | - OPENAI_API_KEY  |
                                                    +-------------------+
```

### Installation Flow

```text
1. Install FFmpeg (prerequisite)
   - macOS: brew install ffmpeg
   - Ubuntu: apt install ffmpeg
   - Windows: choco install ffmpeg / winget install ffmpeg

2. Install transcribe-cli
   pip install transcribe-cli

3. Configure API key
   export OPENAI_API_KEY=sk-...
   # or
   echo "OPENAI_API_KEY=sk-..." > ~/.transcriberc

4. Verify installation
   transcribe --version
   transcribe --help
```

### CI/CD Pipeline

```text
[Push to main] --> [Lint & Format] --> [Type Check] --> [Unit Tests]
                                                              |
                                                              v
                   [Integration Tests] <-- [Security Scan] <--+
                           |
                           v
                   [Build Package] --> [Publish to PyPI]
                           |               (on release tag)
                           v
                   [GitHub Release]
```

---

## 10. Future Considerations

### Planned Enhancements (Post-MVP)

| Feature | Priority | Complexity | Notes |
|---------|----------|------------|-------|
| VTT output format | High | Low | Similar to SRT, easy to add |
| JSON output format | Medium | Low | Structured transcript data |
| Speaker diarization | Medium | Medium | Depends on Whisper API support |
| AI-generated summaries | Low | Medium | Additional GPT API call |
| Local Whisper model | Low | High | `whisper.cpp` integration |
| Cloud storage integration | Low | Medium | S3, Google Drive |

### Scalability Path

Current architecture supports up to:
- 20 concurrent users (API rate limits)
- 100 files/day (API costs manageable)
- Files up to 2+ hours (chunking handles)

If growth exceeds these limits:
1. Add caching layer (transcript deduplication)
2. Consider job queue for large batches
3. Explore local Whisper model for cost reduction
4. Add usage tracking and cost monitoring

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FFmpeg installation issues | High | Medium | Clear docs, startup check with helpful error |
| Large file API limits | Medium | High | Automatic chunking, progress checkpointing |
| Whisper API rate limits | Low-Medium | Medium | Exponential backoff, configurable concurrency |
| Unusual codec failures | Medium | Medium | Format detection, conversion fallback |
| API cost overruns | Low | Low | Usage warnings (future), cost estimation |

---

## 12. Open Questions

1. **CLI Framework**: `click` vs `typer` - Need team preference (both viable)
2. **Test Coverage Threshold**: 30% for PR merge vs 60% target - Confirm policy
3. **FFmpeg Version**: Minimum 4.0 or higher? Platform testing needed
4. **Resume Implementation**: File hash comparison vs timestamp-based?
5. **Transcript Caching**: Implement in MVP or defer to v2?

---

## Summary

The Audio Transcription CLI Tool architecture is a **simple, modular Python CLI monolith** optimized for:

- **Team adoption**: Easy installation, single-command usage, clear error messages
- **Reliability**: Graceful error handling, retry logic, batch resume support
- **Maintainability**: Clean separation of concerns, testable components
- **Extensibility**: Plugin-style modules for future format support

Key technology choices:
- `ffmpeg-python` for audio extraction (best balance of speed and flexibility)
- `asyncio` for concurrent batch processing (efficient for I/O-bound operations)
- `openai` SDK for Whisper API integration (official, well-maintained)
- `rich` for beautiful terminal output (progress bars, status messages)

The architecture supports the 1-3 month MVP timeline with clear paths for future enhancements (additional formats, local model, cloud storage).

---

## References

- [Project Intake](/home/manitcor/dev/tnf/.aiwg/intake/project-intake.md)
- [Option Matrix](/home/manitcor/dev/tnf/.aiwg/intake/option-matrix.md)
- [OpenAI Whisper API](https://platform.openai.com/docs/guides/speech-to-text)
- [ffmpeg-python Documentation](https://github.com/kkroening/ffmpeg-python)
- [Click Documentation](https://click.palletsprojects.com/)
- [Rich Documentation](https://rich.readthedocs.io/)

---

**Document Status**: DRAFT - Awaiting review by Technical Lead and Security Architect

**Next Steps**:
1. Review and validate architecture with team
2. Create detailed ADRs for key decisions (ADR-001, ADR-002, ADR-003)
3. Proceed to Elaboration phase with architecture baseline
