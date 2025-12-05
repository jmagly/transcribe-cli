# Master Test Plan: Audio Transcription CLI Tool

---

## Document Information

| Attribute | Value |
|-----------|-------|
| Document Type | Master Test Plan |
| Version | 1.0 |
| Status | DRAFT |
| Date | 2025-12-04 |
| Project | Audio Transcription CLI Tool |
| Phase | Elaboration |
| Owner | Test Architect |

---

## 1. Introduction and Scope

### 1.1 Purpose

This Master Test Plan defines the comprehensive testing strategy, approach, and execution plan for the Audio Transcription CLI Tool MVP. It establishes quality objectives, test coverage targets, test types, schedules, and responsibilities to ensure the system meets functional and non-functional requirements before release.

The plan supports:
- Development teams implementing test-driven development
- QA teams validating functional and non-functional requirements
- Security teams verifying security controls
- Product owners assessing release readiness
- Stakeholders understanding quality assurance approach

### 1.2 Scope

This test plan covers:

**In Scope:**
- Unit testing of core modules (extraction, transcription, formatting, batch processing)
- Integration testing of CLI workflows and external dependencies
- End-to-end testing with real audio/video samples
- Security testing (API key protection, input validation, dependency scanning)
- Performance testing (latency, throughput, resource usage)
- Error handling and edge case validation
- Cross-platform compatibility testing (Linux, macOS, Windows)

**Out of Scope:**
- Load testing beyond batch processing scenarios (not a web service)
- Accessibility testing (CLI tool, no GUI)
- Internationalization testing (Whisper API handles languages)
- User acceptance testing logistics (managed separately by Product Owner)

### 1.3 Audience

| Audience | Relevance |
|----------|-----------|
| Test Engineers | Test execution, automation, reporting |
| Developers | Test-driven development, fixture creation, mocking |
| Security Architect | Security test validation, vulnerability remediation |
| Product Owner | Release readiness, acceptance criteria validation |
| Tech Lead | Test strategy approval, resource allocation |

### 1.4 Quality Objectives

| Objective | Target | Rationale |
|-----------|--------|-----------|
| Overall Code Coverage | 60% minimum | Balance MVP velocity with quality assurance per Solution Profile |
| Critical Path Coverage | 90%+ | Core workflows (extract, transcribe, format) must be reliable |
| Success Rate | 95%+ for valid files | NFR-REL-001: Reliability requirement |
| Security Vulnerabilities | Zero critical/high | NFR-SEC-003: Dependency security requirement |
| Performance Compliance | 100% P0 NFRs met | NFR-PERF-001, 002, 003 critical for user adoption |

### 1.5 References

| Document | Location |
|----------|----------|
| Software Architecture Document | /home/manitcor/dev/tnf/.aiwg/architecture/software-architecture-doc.md |
| Non-Functional Requirements | /home/manitcor/dev/tnf/.aiwg/requirements/non-functional-requirements.md |
| Solution Profile | /home/manitcor/dev/tnf/.aiwg/intake/solution-profile.md |
| Use Case Briefs | /home/manitcor/dev/tnf/.aiwg/requirements/use-case-briefs/ |
| ADR-001 | /home/manitcor/dev/tnf/.aiwg/architecture/adr/ADR-001-ffmpeg-integration-approach.md |
| ADR-002 | /home/manitcor/dev/tnf/.aiwg/architecture/adr/ADR-002-batch-processing-concurrency.md |

---

## 2. Test Strategy

### 2.1 Testing Philosophy

**Risk-Based Testing**: Prioritize testing based on:
1. User impact (core workflows: extract, transcribe, format)
2. Technical risk (external dependencies: FFmpeg, OpenAI API)
3. Complexity (async batch processing, large file chunking)
4. Security (API key protection, input validation)

**Test Pyramid Approach**:
```
       E2E (10%)
      Manual/Real API
    /                \
   Integration (30%)
  Mocked Dependencies
 /                    \
Unit (60%)
Pure Logic Testing
```

**Quality Gates**: Tests run automatically in CI pipeline. All tests must pass before merge.

### 2.2 Test Levels

| Test Level | Coverage Target | Tools | Execution |
|------------|-----------------|-------|-----------|
| Unit Tests | 60% overall | pytest, pytest-cov | Every commit (CI) |
| Integration Tests | Critical paths | pytest, click.testing.CliRunner | Every PR (CI) |
| End-to-End Tests | Sample workflows | Manual + pytest | Pre-release |
| Security Tests | All controls | pip-audit, bandit, manual review | Daily (CI) |
| Performance Tests | P0 NFRs | pytest-benchmark, custom scripts | Pre-release |
| Platform Tests | Linux, macOS, Windows | CI matrix + manual | Every PR (CI) |

### 2.3 Component Coverage Targets

Per SAD Section 10.2 and Solution Profile testing requirements:

| Component | Target Coverage | Rationale | Priority |
|-----------|-----------------|-----------|----------|
| Output Formatter | 90% | Pure logic, easy to test, high value | P0 |
| Transcription Client | 80% | Mock API, critical path | P0 |
| Config Manager | 80% | Validation edge cases, security | P0 |
| Audio Extractor | 70% | Mock FFmpeg, command generation | P0 |
| CLI Entry Point | 60% | Integration tests, orchestration | P0 |
| Batch Processor | 60% | Complex async, test orchestration | P0 |
| Utilities | 50% | File operations, logging | P1 |

### 2.4 Test Data Strategy

**Principles**:
- Realistic: Use real audio samples covering common formats and durations
- Reproducible: Store fixtures in version control or provide download script
- Comprehensive: Cover normal cases, edge cases, and error scenarios
- Secure: No PII in test files, synthetic audio for large files

**Test Data Categories**:

| Category | Purpose | Examples |
|----------|---------|----------|
| Small Audio | Unit tests, quick validation | 5-second tone, 30-second speech |
| Medium Audio | Integration tests, realistic workflows | 5-minute meeting, 15-minute interview |
| Large Audio | Large file handling, chunking | 90-minute podcast, 3-hour conference |
| Video Files | Extraction testing | 5-minute MKV, 30-minute MP4 |
| Edge Cases | Error handling, boundary testing | Corrupted files, 25MB boundary, unsupported formats |
| Mock Responses | API testing, error simulation | Success, 429 rate limit, 500 server error |

---

## 3. Test Types and Levels

### 3.1 Unit Tests

**Objective**: Validate individual modules in isolation with mocked external dependencies.

**Scope**:
- Pure business logic (formatting, chunking, validation)
- Component interfaces (protocols, abstract classes)
- Error handling and edge cases
- Configuration parsing and validation

**Tools**:
- pytest (test framework)
- pytest-asyncio (async test support)
- unittest.mock (mocking external dependencies)
- pytest-cov (coverage reporting)

**Key Test Cases**:

**Output Formatter (90% coverage target)**:
- TXT formatter: Plain text output, UTF-8 encoding, line breaks
- SRT formatter: Sequence numbering, timestamp formatting (HH:MM:SS,mmm), text blocks
- VTT formatter (v2): WEBVTT header, cue timing, text cues
- JSON formatter (v2): Metadata structure, segments array, language detection
- Error cases: Empty transcript, missing timestamps, invalid characters

**Transcription Client (80% coverage target)**:
- API request construction (model, file, response format)
- Response parsing (text, segments, language, duration)
- Retry logic: Exponential backoff (2s, 4s, 8s), max 3 retries
- Error handling: 429 rate limit, 500 server error, timeout, network failure
- Large file detection and chunking trigger

**Config Manager (80% coverage target)**:
- Environment variable loading (OPENAI_API_KEY)
- .env file parsing (valid, missing, malformed)
- Config validation: API key presence, FFmpeg path, output directory
- Permission validation: Config file 600 or more restrictive
- Default values: TXT format, current directory output, 5 parallel workers

**Audio Extractor (70% coverage target)**:
- FFmpeg command generation (input path, output path, codec settings)
- Format detection (MKV, MP4, AVI, MOV)
- Audio stream selection (first audio track)
- Error handling: FFmpeg not found, invalid file, unsupported codec
- Temp file management: Creation, cleanup, permissions

**CLI Entry Point (60% coverage target)**:
- Argument parsing (file path, flags, options)
- Command routing (transcribe, extract, batch, config)
- Help text generation and display
- Error message formatting
- Exit code handling

**Batch Processor (60% coverage target)**:
- File discovery: Recursive scan, extension filtering, pattern matching
- Job queue management: FIFO, priority, state persistence
- Semaphore limiting: Max 5 concurrent, configurable
- Progress aggregation: Total, completed, failed, ETA
- Resume logic: Checkpoint loading, skip completed files

**Example Test Pattern**:
```python
# tests/unit/test_transcriber.py
import pytest
from unittest.mock import AsyncMock, patch
from transcribe_cli.core.transcriber import TranscriberClient
from transcribe_cli.models.transcript import TranscriptionResult

@pytest.fixture
def mock_openai_client():
    """Mock async OpenAI client for transcriber tests."""
    client = AsyncMock()
    client.audio.transcriptions.create = AsyncMock(
        return_value={
            "text": "This is a test transcript.",
            "segments": [{"id": 1, "start": 0.0, "end": 2.5, "text": "This is a test transcript."}],
            "language": "en",
            "duration": 2.5
        }
    )
    return client

@pytest.mark.asyncio
async def test_transcribe_success(mock_openai_client):
    """Test successful transcription with mocked API."""
    transcriber = TranscriberClient(client=mock_openai_client)
    result = await transcriber.transcribe("sample.mp3")

    assert isinstance(result, TranscriptionResult)
    assert result.text == "This is a test transcript."
    assert len(result.segments) == 1
    assert result.language == "en"
    mock_openai_client.audio.transcriptions.create.assert_called_once()

@pytest.mark.asyncio
async def test_transcribe_retry_on_rate_limit(mock_openai_client):
    """Test retry logic with rate limit error."""
    mock_openai_client.audio.transcriptions.create = AsyncMock(
        side_effect=[
            Exception("Rate limit exceeded (429)"),
            Exception("Rate limit exceeded (429)"),
            {"text": "Success after retries", "segments": [], "language": "en", "duration": 1.0}
        ]
    )
    transcriber = TranscriberClient(client=mock_openai_client, max_retries=3)
    result = await transcriber.transcribe("sample.mp3")

    assert result.text == "Success after retries"
    assert mock_openai_client.audio.transcriptions.create.call_count == 3
```

### 3.2 Integration Tests

**Objective**: Validate component interactions and end-to-end workflows with mocked external services.

**Scope**:
- Full CLI workflows (command → output file)
- Component integration (CLI → Core → Output → File System)
- Error propagation and handling across layers
- Configuration loading and application
- Temp file lifecycle management

**Tools**:
- pytest (test framework)
- click.testing.CliRunner (CLI testing)
- pytest-asyncio (async workflow testing)
- Mock file systems (pytest-mock, tempfile)

**Key Test Scenarios**:

**UC-001: Transcribe Single Audio File**:
- Happy path: `transcribe sample.mp3` → `sample.txt` generated
- Format selection: `transcribe sample.mp3 --format srt` → `sample.srt` generated
- Output path: `transcribe sample.mp3 -o ./output/` → file in custom directory
- Verbose mode: `transcribe sample.mp3 --verbose` → detailed logs displayed
- API key missing: Error message with setup instructions
- Invalid file: Error message listing supported formats

**UC-002: Extract and Transcribe Video File**:
- MKV extraction: `transcribe video.mkv` → audio extracted → transcribed → output
- Temp file cleanup: Verify temp audio deleted after processing
- FFmpeg not found: Error message with installation link
- Unsupported codec: Error with codec details and conversion suggestion

**UC-003: Batch Process Directory**:
- Small batch: `transcribe ./recordings/` → 5 files processed, summary displayed
- Mixed formats: Audio and video files in batch, all processed correctly
- Parallel execution: Verify 5 concurrent jobs, semaphore limiting
- Error isolation: 1 corrupted file fails, others succeed
- Resume: Interrupt batch, restart with `--resume`, skip completed files

**UC-004: Handle Large File**:
- Chunking trigger: 50MB file automatically chunked into 40-minute segments
- Timestamp continuity: Verify no gaps or overlaps at chunk boundaries
- Checkpoint save: State file created after each chunk
- Resume from checkpoint: Interrupt mid-chunk, resume successfully

**UC-005: Generate Timestamped Output**:
- SRT format: Verify sequence numbers, timestamp format, text blocks
- Segment alignment: Timestamps match Whisper API response
- Empty segments: Graceful handling of silence periods

**Example Integration Test Pattern**:
```python
# tests/integration/test_cli.py
from click.testing import CliRunner
from transcribe_cli.cli.main import cli
import pytest
from pathlib import Path

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def sample_audio(tmp_path):
    """Create a small sample audio file for testing."""
    audio_file = tmp_path / "sample.mp3"
    # Generate synthetic audio or copy fixture
    audio_file.write_bytes(b"FAKE_AUDIO_DATA")
    return audio_file

def test_transcribe_single_file_success(runner, sample_audio, mock_api_client):
    """Test end-to-end transcription of single audio file."""
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ['transcribe', str(sample_audio)])

        assert result.exit_code == 0
        assert "Success" in result.output
        assert Path("transcripts/sample.txt").exists()

        transcript_content = Path("transcripts/sample.txt").read_text()
        assert len(transcript_content) > 0

def test_transcribe_missing_api_key(runner, sample_audio, monkeypatch):
    """Test error handling when API key is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(cli, ['transcribe', str(sample_audio)])

    assert result.exit_code == 2
    assert "API key not found" in result.output
    assert "OPENAI_API_KEY" in result.output
```

### 3.3 End-to-End Tests

**Objective**: Validate complete workflows with real external dependencies (limited scope to control costs).

**Scope**:
- Real audio samples (MP3, FLAC, WAV, AAC, M4A)
- Real video samples (MKV with various codecs)
- Real OpenAI API calls (limited to pre-release testing)
- Real FFmpeg execution
- Full file system interactions

**Execution**:
- Pre-release manual testing (not CI due to cost)
- Selected sample files (5-10 files covering key scenarios)
- Platform-specific validation (Linux, macOS, Windows)

**Test Scenarios**:
1. 5-minute MP3 meeting recording → TXT output
2. 30-minute MKV Zoom recording → SRT subtitle output
3. Batch process 5 mixed files (audio + video) → all transcripts generated
4. 90-minute large audio file → chunking, merging, timestamp validation
5. Interrupt large file processing → resume → successful completion

**Success Criteria**:
- Transcripts match audio content (manual review)
- Timestamps accurate within 0.5 seconds
- File formats valid (TXT readable, SRT playable)
- Processing time meets NFR targets
- No memory leaks or resource exhaustion

### 3.4 Security Tests

**Objective**: Verify security controls and protect against common vulnerabilities.

**Scope**:
- API key protection (never logged, masked in output)
- Input validation (path traversal, command injection)
- Dependency vulnerabilities (pip-audit, safety)
- Temp file permissions (0700, secure cleanup)
- Subprocess security (shell=False enforcement)

**Tools**:
- pip-audit (dependency scanning)
- bandit (static security analysis)
- Manual security review
- Custom test cases for input validation

**Test Cases**:

**NFR-SEC-001: API Key Protection**:
- Verify API key never appears in stdout/stderr
- Verify API key never logged (grep for key pattern in logs)
- Verify API key stored as SecretStr (pydantic masking)
- Verify config file permission validation (reject >0600)

**NFR-SEC-002: Input Validation**:
- Path traversal: `../../../etc/passwd` rejected
- Null bytes: `file\x00.mp3` rejected
- Shell metacharacters: `file; rm -rf /` sanitized
- Extension whitelist: `.exe`, `.sh`, `.py` rejected
- File signature validation: `.mp3.exe` detected and rejected

**NFR-SEC-003: Dependency Security**:
- pip-audit runs daily in CI, fails on CRITICAL/HIGH vulnerabilities
- Dependabot alerts reviewed within 48 hours
- Pinned dependency versions in requirements.txt

**NFR-SEC-004: Temporary File Cleanup**:
- Temp directory created with secure permissions (0700)
- Temp files deleted on successful completion
- Temp files deleted on process exit (atexit handler)
- Temp files deleted on SIGINT/SIGTERM (signal handlers)

**NFR-SEC-005: Subprocess Security**:
- Static analysis: No `shell=True` in codebase (bandit check)
- Code review: FFmpeg wrapper library used exclusively
- Unit tests: Verify subprocess calls use list-based arguments

### 3.5 Performance Tests

**Objective**: Validate performance against NFR targets (latency, throughput, resource usage).

**Scope**:
- Single file transcription time (NFR-PERF-001)
- Audio extraction speed (NFR-PERF-002)
- Batch processing throughput (NFR-PERF-003)
- Application startup time (NFR-PERF-004)
- Memory usage efficiency (NFR-PERF-005)
- Large file processing capability (NFR-PERF-006)

**Tools**:
- pytest-benchmark (microbenchmarks)
- time command (macro benchmarks)
- memory_profiler (memory profiling)
- Custom performance scripts

**Test Cases**:

**NFR-PERF-001: Single File Transcription Time**:
- Test: 30-minute MP3 file → TXT output
- Target: <5 minutes (p90), <6 minutes (p95)
- Measurement: Time from command invocation to file written
- Validation: 10 runs, calculate percentiles

**NFR-PERF-002: Audio Extraction Speed**:
- Test: 1-hour MKV file → MP3 extraction
- Target: <30 seconds
- Measurement: FFmpeg execution time
- Validation: 5 runs, average time

**NFR-PERF-003: Batch Processing Throughput**:
- Test: 10 files (10 minutes each) → parallel processing (5 workers)
- Target: <25 minutes total (4-5x speedup vs. sequential)
- Measurement: Total batch time
- Validation: Compare sequential baseline vs. parallel

**NFR-PERF-004: Application Startup Time**:
- Test: `transcribe --help` → help text displayed
- Target: <1 second (p95)
- Measurement: Time from command invocation to output
- Validation: 20 runs, calculate p95

**NFR-PERF-005: Memory Usage Efficiency**:
- Test: 2GB large file processing
- Target: <512MB peak RSS (Resident Set Size)
- Measurement: memory_profiler or /proc/pid/status
- Validation: Monitor throughout processing

**NFR-PERF-006: Large File Processing Capability**:
- Test: 3-hour audio file (2GB)
- Target: Successful processing via chunking
- Measurement: Timestamp continuity, no errors
- Validation: Manual review of output

---

## 4. Component Coverage Matrix

Per SAD Section 10.2 and Solution Profile requirements:

| Component | Files | Target | Unit Tests | Integration Tests | Critical Paths |
|-----------|-------|--------|------------|-------------------|----------------|
| Output Formatter | formatter.py, txt.py, srt.py | 90% | 25+ cases | 5+ workflows | TXT/SRT generation, timestamp formatting |
| Transcription Client | transcriber.py | 80% | 20+ cases | 10+ workflows | API request, retry logic, error handling |
| Config Manager | settings.py, validators.py | 80% | 15+ cases | 5+ workflows | Env loading, validation, permission check |
| Audio Extractor | extractor.py | 70% | 15+ cases | 8+ workflows | FFmpeg execution, format detection, cleanup |
| CLI Entry Point | main.py, commands/* | 60% | 10+ cases | 15+ workflows | Argument parsing, command routing, help text |
| Batch Processor | processor.py | 60% | 12+ cases | 10+ workflows | Parallel execution, semaphore, resume logic |
| Utilities | file_utils.py, progress.py, logging.py | 50% | 10+ cases | 3+ workflows | File operations, progress tracking, log sanitization |

**Total Estimated Test Cases**: 107+ unit tests, 56+ integration tests, 10+ E2E tests = **173+ test cases**

**Coverage Enforcement**:
- CI pipeline enforces 60% minimum overall coverage (pytest --cov-fail-under=60)
- Component-specific targets tracked in coverage reports
- Pull requests blocked if coverage drops below threshold

---

## 5. Test Data Management

### 5.1 Test Fixtures

**Location**: `tests/fixtures/`

**Structure**:
```
tests/fixtures/
|-- audio/
|   |-- small/
|   |   |-- tone-5sec.mp3           # Synthetic 5-second tone
|   |   |-- speech-30sec.mp3        # 30-second speech sample
|   |-- medium/
|   |   |-- meeting-5min.mp3        # 5-minute meeting recording
|   |   |-- interview-15min.flac    # 15-minute interview (FLAC format)
|   |-- large/
|       |-- podcast-90min.mp3       # 90-minute podcast (45MB)
|       |-- conference-3hr.mp3      # 3-hour conference (100MB)
|-- video/
|   |-- short-5min.mkv              # 5-minute MKV with audio
|   |-- long-30min.mp4              # 30-minute MP4 with audio
|-- edge-cases/
|   |-- exact-25mb.mp3              # Exactly 25MB file (boundary test)
|   |-- corrupted.mp3               # Intentionally corrupted file
|   |-- unsupported.xyz             # Unsupported format
|   |-- empty.mp3                   # Zero-byte file
|-- api-responses/
|   |-- success.json                # Successful transcription response
|   |-- rate-limit-429.json         # Rate limit error response
|   |-- server-error-500.json       # Server error response
|   |-- invalid-file.json           # Invalid file error response
|-- golden/
    |-- tone-5sec.txt               # Expected transcript for tone-5sec.mp3
    |-- speech-30sec.srt            # Expected SRT for speech-30sec.mp3
```

### 5.2 Fixture Generation

**Small Files** (stored in version control):
- Synthetic audio via FFmpeg:
  ```bash
  # Generate 5-second tone
  ffmpeg -f lavfi -i sine=frequency=440:duration=5 -ab 192k tone-5sec.mp3

  # Generate 30-second tone
  ffmpeg -f lavfi -i sine=frequency=440:duration=30 -ab 192k tone-30sec.mp3
  ```

**Medium Files** (stored in version control or downloaded):
- Real speech samples from public domain sources
- Checksums documented for integrity verification

**Large Files** (downloaded via script):
- `tests/fixtures/download_large_fixtures.py` script
- Downloads from public URLs or generates synthetic audio
- Not committed to version control (excluded via .gitignore)

**Mock API Responses**:
- JSON files matching OpenAI Whisper API response format
- Stored in version control for unit/integration tests

### 5.3 Golden Transcripts

**Purpose**: Regression testing - verify transcription quality doesn't degrade.

**Storage**: `tests/fixtures/golden/`

**Usage**:
```python
def test_regression_tone_5sec():
    """Verify transcription matches golden reference."""
    result = transcribe("tests/fixtures/audio/small/tone-5sec.mp3")
    expected = Path("tests/fixtures/golden/tone-5sec.txt").read_text()
    assert result.text == expected
```

**Maintenance**:
- Golden files reviewed and approved by domain expert
- Updated only when intentional changes to transcription logic
- Version controlled with checksums

### 5.4 Test Data Security

**Principles**:
- No PII in test files (GDPR/privacy compliance)
- Synthetic audio preferred for large files
- Public domain sources for real speech samples
- No sensitive API keys in fixtures (use environment variables)

**Compliance**:
- All test files documented with source and license
- Large files excluded from version control (.gitignore)
- API keys never committed (use .env.test, .env.example)

---

## 6. Test Environment

### 6.1 Local Development Environment

**Prerequisites**:
- Python 3.9+ installed
- FFmpeg 4.0+ installed
- Git for version control
- Virtual environment (venv or virtualenv)

**Setup**:
```bash
# Clone repository
git clone <repo-url>
cd transcribe-cli

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .

# Download large fixtures (optional)
python tests/fixtures/download_large_fixtures.py

# Run tests
pytest tests/ -v
```

**Environment Variables**:
```bash
# .env.test
OPENAI_API_KEY=test-key-placeholder  # For unit/integration tests (mocked)
TRANSCRIBE_TEST_MODE=1                # Enable test mode (skip real API calls)
```

### 6.2 CI/CD Environment

**Platform**: GitHub Actions

**Test Matrix**:
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ['3.9', '3.10', '3.11', '3.12']
```

**Pipeline Stages**:

**Stage 1: Linting and Static Analysis** (every commit)
- black --check (code formatting)
- flake8 (linting)
- mypy (type checking)
- bandit (security static analysis)

**Stage 2: Unit Tests** (every commit)
- pytest tests/unit/ -v
- pytest-cov for coverage reporting
- Coverage threshold: 60% minimum (fail if below)

**Stage 3: Integration Tests** (every commit)
- pytest tests/integration/ -v
- Mock external dependencies (FFmpeg, OpenAI API)

**Stage 4: Security Scanning** (daily)
- pip-audit (dependency vulnerabilities)
- Fail on CRITICAL/HIGH severity

**Stage 5: Platform Testing** (every PR)
- Ubuntu: Primary platform, full test suite
- macOS: Full test suite
- Windows: Integration tests only (FFmpeg setup manual)

**Stage 6: Performance Tests** (pre-release)
- Benchmark suite (pytest-benchmark)
- Memory profiling (memory_profiler)
- Manual validation of NFR targets

### 6.3 Test Isolation

**Principles**:
- Each test runs independently (no shared state)
- Temp directories created per test (pytest tmp_path fixture)
- Mock external services (no real API calls in CI)
- Cleanup after test completion (fixtures with teardown)

**Fixture Pattern**:
```python
@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Provide isolated environment for each test."""
    # Create temp directories
    output_dir = tmp_path / "transcripts"
    output_dir.mkdir()

    # Set environment variables
    monkeypatch.setenv("TRANSCRIBE_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    yield tmp_path

    # Cleanup (automatic via tmp_path)
```

---

## 7. CI/CD Pipeline Configuration

### 7.1 GitHub Actions Workflow

**File**: `.github/workflows/test.yml`

```yaml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt

      - name: Run linting
        run: |
          black --check src/ tests/
          flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
          mypy src/ --strict

      - name: Security scan
        run: |
          bandit -r src/ -ll
          pip-audit

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install FFmpeg
        run: |
          if [ "$RUNNER_OS" == "Linux" ]; then
            sudo apt-get update && sudo apt-get install -y ffmpeg
          elif [ "$RUNNER_OS" == "macOS" ]; then
            brew install ffmpeg
          fi
        shell: bash

      - name: Verify FFmpeg
        run: ffmpeg -version

      - name: Install Python dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
          pip install -e .

      - name: Run unit tests with coverage
        run: |
          pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=term

      - name: Run integration tests
        run: |
          pytest tests/integration/ -v

      - name: Check coverage threshold
        run: |
          pytest --cov=src --cov-report=term --cov-fail-under=60

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          flags: unittests
          name: codecov-${{ matrix.os }}-py${{ matrix.python-version }}

  test-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
          pip install -e .

      - name: Run integration tests (Windows)
        run: |
          pytest tests/integration/ -v -k "not ffmpeg"
        # Note: FFmpeg tests skipped on Windows CI (manual setup required)
```

### 7.2 Coverage Reporting

**Tools**:
- pytest-cov (local coverage)
- Codecov (coverage tracking and reporting)

**Configuration**: `.coveragerc`
```ini
[run]
source = src
omit =
    tests/*
    */venv/*
    */__pycache__/*

[report]
precision = 2
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
    if TYPE_CHECKING:

[html]
directory = htmlcov
```

**Tracking**:
- Coverage badge in README (Codecov integration)
- Pull request comments with coverage delta
- Fail PR if coverage drops >2%

### 7.3 Test Execution Cadence

| Trigger | Tests Run | Duration |
|---------|-----------|----------|
| Every commit | Lint + Unit tests | ~2 minutes |
| Every PR | Lint + Unit + Integration | ~5 minutes |
| Daily (scheduled) | Security scan (pip-audit) | ~1 minute |
| Pre-release | Full suite + E2E + Performance | ~30 minutes |
| Post-release | Smoke tests + User validation | Manual |

---

## 8. Error Testing Matrix

Per SAD Section 10.3, comprehensive error handling validation:

| Error Type | Trigger Method | Expected Behavior | Test Validation |
|------------|----------------|-------------------|-----------------|
| MissingApiKeyError | Unset OPENAI_API_KEY | Error message with setup docs link | Assert message contains "OPENAI_API_KEY" and docs URL |
| FFmpegNotFoundError | Mock shutil.which() → None | Platform-specific install link | Assert message contains "brew install ffmpeg" or "apt install ffmpeg" |
| RateLimitError | Mock 429 API response | Retry with exponential backoff (2s, 4s, 8s) | Assert 3 retry attempts, verify backoff timing |
| TimeoutError | Mock API delay >60s | Graceful timeout message | Assert timeout after 60s, clear error message |
| CorruptedFileError | Provide malformed audio fixture | Clean error, no crash, no path leak | Assert error message does not contain full path |
| UnsupportedFormatError | Provide .xyz file | List supported formats | Assert message contains "MP3, AAC, FLAC, WAV, M4A" |
| WritePermissionError | Mock permission denied on output | Suggest chmod or alternate path | Assert message contains "permission" and "chmod" |
| NetworkError | Mock connection refused | Retry 2x, then clear error | Assert retry count, verify error suggests network check |
| InvalidConfigError | Provide malformed .env file | Parse error with line number | Assert message contains "line X" and fix suggestion |
| DiskFullError | Mock OSError (ENOSPC) | Clear error, partial results saved | Assert error message, verify partial transcripts exist |

**Error Message Validation**:
- No full file system paths (use relative paths from working directory)
- No stack traces in default mode (require --verbose)
- No API response bodies beyond status code
- Clear "Suggested fix:" section with actionable steps

**Example Error Test**:
```python
def test_missing_api_key_error_message(runner, sample_audio, monkeypatch):
    """Verify API key error message is clear and actionable."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(cli, ['transcribe', str(sample_audio)])

    assert result.exit_code == 2
    assert "API key not found" in result.output
    assert "OPENAI_API_KEY" in result.output
    assert "https://" in result.output  # Documentation link
    assert str(sample_audio) not in result.output  # No full path leak
```

---

## 9. Security Testing

Per NFRs SEC-001 through SEC-005:

### 9.1 API Key Protection (NFR-SEC-001)

**Test Cases**:
1. Verify API key never appears in stdout
2. Verify API key never appears in stderr
3. Verify API key never logged (grep for pattern in log files)
4. Verify API key stored as SecretStr (repr() shows masked value)
5. Verify config file permission validation (reject if >0600)
6. Verify .env file in .gitignore template

**Validation Method**:
```python
def test_api_key_not_logged(runner, sample_audio, tmp_path, caplog):
    """Verify API key is never logged in any mode."""
    api_key = "sk-test-key-1234567890"

    with runner.isolated_filesystem():
        with open(".env", "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}")

        result = runner.invoke(cli, ['transcribe', str(sample_audio), '--verbose'])

        # Check stdout, stderr, and log messages
        assert api_key not in result.output
        assert api_key not in caplog.text
        assert "[REDACTED]" in caplog.text or "***" in caplog.text
```

### 9.2 Input Validation (NFR-SEC-002)

**Test Cases**:

**Path Traversal**:
- `../../../etc/passwd` → Rejected with error
- `..\\..\\..\\windows\\system32\\config\\sam` → Rejected (Windows)
- Symlinks outside working directory → Rejected

**Null Bytes**:
- `file\x00.mp3` → Rejected with error

**Shell Metacharacters**:
- `file; rm -rf /.mp3` → Sanitized or rejected
- `file && echo pwned.mp3` → Sanitized or rejected
- `$(whoami).mp3` → Sanitized or rejected

**Extension Whitelist**:
- `malware.exe` → Rejected with supported formats list
- `script.sh` → Rejected
- `file.mp3.exe` → Rejected (double extension)

**File Signature Validation**:
- Rename `.exe` to `.mp3` → Detected and rejected
- Corrupted magic bytes → Rejected with corruption error

**Validation Method**:
```python
@pytest.mark.parametrize("malicious_path", [
    "../../../etc/passwd",
    "file\x00.mp3",
    "file; rm -rf /.mp3",
    "malware.exe",
    "file.mp3.exe"
])
def test_malicious_input_rejected(runner, malicious_path):
    """Verify malicious file paths are rejected."""
    result = runner.invoke(cli, ['transcribe', malicious_path])

    assert result.exit_code != 0
    assert "Invalid" in result.output or "Unsupported" in result.output
    # Verify no command execution or path traversal occurred
```

### 9.3 Dependency Security (NFR-SEC-003)

**Test Cases**:
1. pip-audit runs in CI (daily)
2. No CRITICAL or HIGH vulnerabilities in dependencies
3. Pinned versions in requirements.txt
4. Dependabot configured and alerts reviewed

**Validation Method**:
- CI pipeline stage: `pip-audit --strict` (fail on vulnerabilities)
- Manual review: `pip list --outdated` monthly
- Dependabot PR review: Within 48 hours of alert

### 9.4 Temporary File Cleanup (NFR-SEC-004)

**Test Cases**:
1. Temp directory created with 0700 permissions
2. Temp files deleted on successful completion
3. Temp files deleted on process exit (atexit handler)
4. Temp files deleted on SIGINT (Ctrl+C)
5. Temp files deleted on SIGTERM

**Validation Method**:
```python
def test_temp_file_cleanup_on_success(tmp_path, monkeypatch):
    """Verify temp files are deleted after successful processing."""
    temp_dir = tmp_path / "temp"
    monkeypatch.setenv("TRANSCRIBE_TEMP_DIR", str(temp_dir))

    # Run transcription (mocked)
    result = transcribe_file("sample.mkv")

    assert result.success
    assert not temp_dir.exists() or len(list(temp_dir.iterdir())) == 0

def test_temp_file_cleanup_on_interrupt(tmp_path, monkeypatch):
    """Verify temp files are deleted when process is interrupted."""
    temp_dir = tmp_path / "temp"
    monkeypatch.setenv("TRANSCRIBE_TEMP_DIR", str(temp_dir))

    with pytest.raises(KeyboardInterrupt):
        # Simulate Ctrl+C during processing
        transcribe_file_with_interrupt("sample.mkv")

    assert not temp_dir.exists() or len(list(temp_dir.iterdir())) == 0
```

### 9.5 Subprocess Security (NFR-SEC-005)

**Test Cases**:
1. Static analysis: No `shell=True` in codebase (bandit check)
2. Code review: FFmpeg wrapper library used exclusively
3. Unit tests: Verify subprocess calls use list-based arguments

**Validation Method**:
- CI pipeline: `bandit -r src/ -ll --format json | grep "shell=True"` → Zero results
- Code review checklist: Subprocess security verification required
- Unit test: Mock subprocess.run, assert args is list (not string)

---

## 10. Acceptance Criteria

### 10.1 MVP Release Gate (P0 NFRs)

All P0 NFRs must pass before MVP release:

**Performance**:
- [ ] NFR-PERF-001: 90th percentile transcription time <5 min for 30-min files
- [ ] NFR-PERF-002: Audio extraction <30 sec for 1-hour MKV
- [ ] NFR-PERF-003: Batch processing achieves 4x+ speedup (5 parallel workers)

**Reliability**:
- [ ] NFR-REL-001: 95%+ success rate on test suite (50+ valid files)
- [ ] NFR-REL-002: Retry logic validated with mock API failures (429, 500, timeout)

**Security**:
- [ ] NFR-SEC-001: API key never appears in logs/output (verified via grep)
- [ ] NFR-SEC-002: Input validation tests pass (malicious inputs rejected)
- [ ] NFR-SEC-005: No `shell=True` detected in codebase (bandit static analysis)

**Usability**:
- [ ] NFR-USE-001: 3/5 new users complete first transcription in <10 min (user testing)
- [ ] NFR-USE-002: 90%+ error scenarios have actionable messages (manual review)

**Maintainability**:
- [ ] NFR-MAIN-001: 60%+ code coverage achieved (CI enforced)
- [ ] NFR-MAIN-002: All linting checks pass (black, flake8, mypy)

**Portability**:
- [ ] NFR-PORT-001: Tests pass on Ubuntu, macOS, Windows
- [ ] NFR-PORT-002: Tests pass on Python 3.9, 3.10, 3.11, 3.12

**Total P0 Criteria**: 14 items (100% must pass)

### 10.2 Component-Specific Acceptance

| Component | Acceptance Criteria |
|-----------|---------------------|
| Output Formatter | 90% coverage, all format tests pass (TXT, SRT, VTT, JSON) |
| Transcription Client | 80% coverage, retry logic validated, error handling comprehensive |
| Config Manager | 80% coverage, permission validation works, env loading robust |
| Audio Extractor | 70% coverage, FFmpeg integration works on all platforms |
| CLI Entry Point | 60% coverage, help text clear, error messages actionable |
| Batch Processor | 60% coverage, parallel execution works, resume logic functional |

### 10.3 User Acceptance Criteria

**From UC-001 (Transcribe Single Audio File)**:
- Single command execution: `transcribe audio.mp3` works without flags
- Transcript matches audio content (>90% accuracy)
- Processing time <5 min for 30-min file
- Error messages clear and actionable

**From UC-003 (Batch Process Directory)**:
- Batch command: `transcribe ./recordings/` processes all files
- Parallel execution: 5 concurrent jobs by default
- Summary report: Success/failure counts, total time, error log
- Resume functionality: `--resume` skips completed files

**Validation Method**: User testing with 3-5 team members, feedback survey

---

## 11. Risk-Based Testing

### 11.1 Risk Assessment

| Risk | Likelihood | Impact | Testing Priority | Mitigation Strategy |
|------|------------|--------|------------------|---------------------|
| FFmpeg Installation Issues | High | Medium | P0 | Comprehensive platform tests, detailed docs, clear error messages |
| Large File Chunking Errors | Medium | High | P0 | Extensive chunking tests, timestamp validation, boundary testing |
| API Rate Limits | Medium | Medium | P0 | Rate limit simulation, backoff validation, concurrency tuning |
| Timestamp Drift in Chunks | Medium | Medium | P1 | Boundary tests, golden transcript comparison, manual review |
| Cross-Platform Path Issues | Low | Medium | P1 | Platform matrix testing, pathlib usage validation |
| Memory Leaks (Long Batches) | Low | High | P1 | Memory profiling, long-running batch tests |

### 11.2 Test Prioritization

**Critical Path (P0)**:
1. Single file transcription (UC-001)
2. Video extraction and transcription (UC-002)
3. Batch processing (UC-003)
4. API key protection (NFR-SEC-001)
5. Input validation (NFR-SEC-002)

**High Priority (P1)**:
1. Large file chunking (UC-004)
2. Timestamped output (UC-005)
3. Error handling (all error types)
4. Performance validation (NFR-PERF-001, 002, 003)

**Medium Priority (P2)**:
1. Edge cases (corrupted files, exact boundaries)
2. Verbose mode and logging
3. Configuration validation
4. Resume functionality

**Low Priority (P3)**:
1. Help text formatting
2. Progress bar aesthetics
3. Output filename customization

### 11.3 Risk Retirement Tests

**RISK-002: FFmpeg Installation Barrier** (High likelihood, Medium impact):
- Test: FFmpeg not found → Clear error with installation link
- Test: FFmpeg old version → Warning with upgrade suggestion
- Test: FFmpeg installation validation on Linux, macOS, Windows
- Test: User testing with fresh setup (3-5 users)

**RISK-003: Large File Handling** (Medium likelihood, High impact):
- Test: 2GB file → Automatic chunking, successful merge
- Test: Chunk boundary timestamps → No gaps, no overlaps
- Test: Interrupted chunking → Resume from checkpoint
- Test: Memory usage during large file → <512MB peak RSS

**RISK-008: API Rate Limits** (Medium likelihood, Medium impact):
- Test: Mock 429 response → Exponential backoff (2s, 4s, 8s)
- Test: Batch with rate limit → Queue pause, resume after backoff
- Test: Concurrency tuning → Verify semaphore limiting (5 max)

---

## 12. Test Schedule

### 12.1 Sprint-Based Testing Plan

**Sprint 1-2 (Inception/Elaboration)**:
- Week 1: Test environment setup, fixture generation
- Week 2: Unit test framework, initial unit tests (30% coverage)

**Sprint 3-4 (Construction - Core Features)**:
- Week 3: Single file transcription tests (UC-001)
- Week 4: Video extraction tests (UC-002)
- Coverage target: 40%

**Sprint 5-6 (Construction - Batch Processing)**:
- Week 5: Batch processing tests (UC-003)
- Week 6: Large file chunking tests (UC-004)
- Coverage target: 50%

**Sprint 7-8 (Construction - Polish)**:
- Week 7: Timestamped output tests (UC-005), error handling
- Week 8: Security tests, performance tests
- Coverage target: 60%

**Sprint 9 (Transition - Pre-Release)**:
- Week 9: E2E tests, platform validation, user acceptance testing
- Coverage validation: 60%+ achieved

**Sprint 10 (Transition - Release)**:
- Week 10: Smoke tests, final validation, release

### 12.2 Continuous Testing Activities

**Daily**:
- CI pipeline: Lint + Unit tests (every commit)
- Security scan: pip-audit (scheduled)

**Per Sprint**:
- Integration tests: Full suite
- Coverage review: Track progress toward 60%
- Bug triage: Review failed tests, prioritize fixes

**Pre-Release**:
- E2E tests: Real audio/video samples
- Performance validation: NFR benchmarks
- User acceptance testing: 3-5 team members
- Security audit: Manual review + static analysis

### 12.3 Milestone-Based Testing

| Milestone | Testing Focus | Exit Criteria |
|-----------|---------------|---------------|
| Inception Complete | Test plan approved | Stakeholder sign-off on test strategy |
| Elaboration Complete | Test framework ready, 30% coverage | Unit tests for core modules passing |
| Construction Phase 1 | Core features tested, 40% coverage | UC-001, UC-002 tests passing |
| Construction Phase 2 | Advanced features tested, 50% coverage | UC-003, UC-004 tests passing |
| Construction Phase 3 | All features tested, 60% coverage | All P0 tests passing, security validated |
| Transition Phase | User acceptance, platform validation | All acceptance criteria met, user sign-off |
| Production Release | Smoke tests, monitoring | Zero critical bugs, 95%+ success rate |

---

## 13. Entry/Exit Criteria

### 13.1 Test Phase Entry Criteria

**Unit Testing**:
- [ ] Code module implemented (feature complete)
- [ ] Test fixtures available
- [ ] Mock dependencies configured
- [ ] Test framework setup (pytest, pytest-cov)

**Integration Testing**:
- [ ] All dependent modules implemented
- [ ] Unit tests passing (90%+ for modules under test)
- [ ] Test environment configured (CliRunner, temp directories)
- [ ] External dependencies mocked

**End-to-End Testing**:
- [ ] All integration tests passing
- [ ] Real audio/video samples available
- [ ] Platform environments ready (Linux, macOS, Windows)
- [ ] OpenAI API access configured (test account with credits)

**Performance Testing**:
- [ ] All functional tests passing
- [ ] Performance benchmarks defined
- [ ] Test data prepared (30-min file, 1-hour MKV, 10-file batch)
- [ ] Measurement tools configured (time, memory_profiler)

**Security Testing**:
- [ ] All functional tests passing
- [ ] Security test cases defined (input validation, API key protection)
- [ ] Static analysis tools configured (pip-audit, bandit)
- [ ] Security review checklist prepared

### 13.2 Test Phase Exit Criteria

**Unit Testing**:
- [ ] Component coverage target achieved (60-90% per component)
- [ ] All unit tests passing (zero failures)
- [ ] No regressions introduced (compared to baseline)
- [ ] Code review completed

**Integration Testing**:
- [ ] All integration tests passing (zero failures)
- [ ] Error handling validated (all error paths tested)
- [ ] Temp file cleanup verified
- [ ] No resource leaks detected

**End-to-End Testing**:
- [ ] All E2E scenarios passing (UC-001 through UC-005)
- [ ] Transcripts match audio content (manual validation)
- [ ] Platform compatibility verified (Linux, macOS, Windows)
- [ ] No critical bugs discovered

**Performance Testing**:
- [ ] All P0 NFR performance targets met
- [ ] No performance regressions (compared to baseline)
- [ ] Memory usage within limits (<512MB for large files)
- [ ] Performance report documented

**Security Testing**:
- [ ] All P0 NFR security controls validated
- [ ] Zero CRITICAL/HIGH vulnerabilities (pip-audit)
- [ ] Input validation comprehensive (all malicious inputs rejected)
- [ ] API key protection verified (never logged)
- [ ] Security audit sign-off

### 13.3 Release Gate Criteria

**MVP Release** (all must pass):
- [ ] Overall code coverage ≥60%
- [ ] All P0 NFRs validated (14 criteria)
- [ ] All integration tests passing
- [ ] Platform tests passing (Ubuntu, macOS, Windows)
- [ ] Python version tests passing (3.9, 3.10, 3.11, 3.12)
- [ ] Security audit complete (zero critical issues)
- [ ] User acceptance testing complete (3/5 users successful)
- [ ] Documentation complete (README, installation guides, troubleshooting)
- [ ] Zero critical bugs (P0/P1 bugs resolved)

---

## Summary

### Total Test Cases Estimated

| Test Type | Count | Focus |
|-----------|-------|-------|
| Unit Tests | 107+ | Business logic, pure functions, error handling |
| Integration Tests | 56+ | Workflows, component integration, error propagation |
| End-to-End Tests | 10+ | Real audio/video, platform validation |
| Security Tests | 20+ | API key protection, input validation, dependency scanning |
| Performance Tests | 6+ | NFR validation (latency, throughput, memory) |
| **Total** | **199+ test cases** | Comprehensive coverage across all dimensions |

### Coverage by Test Type

| Test Type | Coverage Target | Validation Method |
|-----------|-----------------|-------------------|
| Unit Tests | 60% overall, 60-90% per component | pytest-cov, CI enforcement |
| Integration Tests | Critical paths (UC-001 through UC-005) | CliRunner, mocked dependencies |
| End-to-End Tests | Real workflows, platform compatibility | Manual + limited real API calls |
| Security Tests | All P0 security controls (NFR-SEC-001, 002, 005) | Static analysis, manual review |
| Performance Tests | All P0 performance NFRs (PERF-001, 002, 003) | Benchmarking, profiling |

### CI/CD Pipeline Overview

**Continuous Integration** (every commit):
- Linting: black, flake8, mypy
- Unit tests: pytest with coverage
- Duration: ~2 minutes

**Pull Request Validation** (every PR):
- All CI checks
- Integration tests
- Platform matrix: Ubuntu, macOS, Windows
- Python matrix: 3.9, 3.10, 3.11, 3.12
- Duration: ~5 minutes

**Daily Security Scan** (scheduled):
- pip-audit: Dependency vulnerabilities
- Duration: ~1 minute

**Pre-Release Validation** (manual trigger):
- Full test suite
- E2E tests with real samples
- Performance benchmarks
- User acceptance testing
- Duration: ~30 minutes

### Key Risks Addressed by Testing

| Risk | Testing Mitigation |
|------|-------------------|
| FFmpeg Installation Complexity | Platform tests, error message validation, user testing |
| Large File Chunking Errors | Boundary tests, timestamp validation, memory profiling |
| API Rate Limits | Rate limit simulation, backoff validation, concurrency tests |
| API Key Exposure | Log inspection, static analysis, security tests |
| Cross-Platform Issues | Multi-platform CI matrix, pathlib validation |
| Test Coverage Maintenance | CI enforcement, coverage tracking, component targets |

### Next Steps

1. **Review and Approval**: Stakeholder review of Master Test Plan (Product Owner, Tech Lead, Security Architect)
2. **Test Framework Setup**: Initialize pytest, fixtures, CI pipeline (Sprint 1)
3. **Fixture Generation**: Create sample audio/video files, mock API responses (Sprint 1-2)
4. **Unit Test Development**: Implement unit tests for core modules (Sprint 2-8, continuous)
5. **Integration Test Development**: Build CLI integration tests (Sprint 3-8, continuous)
6. **Pre-Release Validation**: Execute E2E, performance, security tests (Sprint 9)
7. **User Acceptance Testing**: Coordinate with Product Owner (Sprint 9)
8. **Release Readiness**: Final validation against release gate criteria (Sprint 10)

---

## Appendices

### Appendix A: Test Automation Tools

| Tool | Version | Purpose |
|------|---------|---------|
| pytest | 7.x | Test framework |
| pytest-asyncio | 0.21+ | Async test support |
| pytest-cov | 4.x | Code coverage reporting |
| pytest-timeout | 2.x | Prevent hung tests |
| pytest-benchmark | 4.x | Performance microbenchmarks |
| click.testing.CliRunner | 8.x | CLI integration testing |
| unittest.mock | stdlib | Mocking external dependencies |
| memory_profiler | 0.61+ | Memory usage profiling |
| pip-audit | 2.x | Dependency security scanning |
| bandit | 1.7+ | Security static analysis |

### Appendix B: Test Environment Variables

```bash
# .env.test (for local testing)
OPENAI_API_KEY=test-key-placeholder
TRANSCRIBE_TEST_MODE=1
TRANSCRIBE_OUTPUT_DIR=./test-output
TRANSCRIBE_TEMP_DIR=/tmp/transcribe-test
TRANSCRIBE_LOG_LEVEL=DEBUG
```

### Appendix C: Glossary

| Term | Definition |
|------|------------|
| Coverage | Percentage of code lines executed during testing |
| Fixture | Test data or setup code reused across multiple tests |
| Mock | Simulated object that mimics real dependency behavior |
| NFR | Non-Functional Requirement (performance, security, usability) |
| P0/P1/P2 | Priority levels: P0 (Critical), P1 (High), P2 (Medium) |
| Regression | Bug reintroduced after previous fix |
| UC | Use Case (functional scenario) |

### Appendix D: Related Documents

| Document | Location |
|----------|----------|
| Software Architecture Document | /home/manitcor/dev/tnf/.aiwg/architecture/software-architecture-doc.md |
| Non-Functional Requirements | /home/manitcor/dev/tnf/.aiwg/requirements/non-functional-requirements.md |
| Solution Profile | /home/manitcor/dev/tnf/.aiwg/intake/solution-profile.md |
| Vision Document | /home/manitcor/dev/tnf/.aiwg/requirements/vision-document.md |
| Use Case Briefs | /home/manitcor/dev/tnf/.aiwg/requirements/use-case-briefs/ |

### Appendix E: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-04 | Test Architect Agent | Initial Master Test Plan based on SAD, NFRs, Solution Profile, Use Cases |

---

**Document End**
