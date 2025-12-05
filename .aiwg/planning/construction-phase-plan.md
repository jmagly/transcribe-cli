# Construction Phase Plan

**Project**: Audio Transcription CLI Tool
**Phase**: Construction
**Duration**: 12 weeks (6 Sprints)
**Target Milestone**: Initial Operational Capability (IOC)
**Start Date**: 2025-12-04
**Target End Date**: 2025-02-28

---

## Phase Objectives

1. **Implement MVP Features**: All 8 core features functional
2. **Achieve Quality Gates**: 60% test coverage, CI/CD green
3. **Validate Risks**: Retire FFmpeg and large file risks via working code
4. **Prepare for Rollout**: Documentation, installation guides complete
5. **Achieve IOC Gate**: Ready for team adoption

---

## Sprint Overview

| Sprint | Weeks | Focus | Key Deliverables |
|--------|-------|-------|------------------|
| Sprint 1 | 1-2 | Foundation | Project setup, CI/CD, Config Manager |
| Sprint 2 | 3-4 | Core Extraction | Audio Extractor, FFmpeg integration |
| Sprint 3 | 5-6 | Transcription | Transcription Client, API integration |
| Sprint 4 | 7-8 | Output & Batch | Output Formatter, Batch Processor |
| Sprint 5 | 9-10 | Large Files | Chunking, Resume, Error handling |
| Sprint 6 | 11-12 | Polish | Testing, Documentation, Release prep |

---

## Sprint 1: Foundation (Weeks 1-2)

### Goals
- Project structure and development environment
- CI/CD pipeline operational
- Config Manager module complete
- CLI entry point skeleton

### User Stories

| ID | Story | Priority | Points |
|----|-------|----------|--------|
| US-1.1 | As a developer, I can clone the repo and run tests | P0 | 3 |
| US-1.2 | As a developer, I can configure API key via environment variable | P0 | 2 |
| US-1.3 | As a developer, I see CI pipeline run on every PR | P0 | 5 |
| US-1.4 | As a user, I can run `transcribe --help` and see available commands | P0 | 3 |
| US-1.5 | As a user, I can run `transcribe --version` to check installation | P1 | 1 |

### Technical Tasks

1. **Project Setup**
   - Initialize Python package structure
   - Create pyproject.toml with dependencies
   - Set up virtual environment
   - Configure pre-commit hooks (black, flake8, mypy)

2. **CI/CD Pipeline**
   - GitHub Actions workflow for lint/test/security
   - Multi-platform matrix (Ubuntu, macOS, Windows)
   - Python version matrix (3.9-3.12)
   - Coverage reporting with 60% gate

3. **Config Manager Module**
   - Pydantic settings model
   - Environment variable loading
   - Config file support (~/.transcriberc)
   - API key validation (SecretStr)

4. **CLI Entry Point**
   - Typer application skeleton
   - Command structure (transcribe, extract, batch, config)
   - Help text and version display
   - Verbose/quiet mode flags

### Definition of Done
- [ ] `pip install -e .` works
- [ ] `pytest` runs with >30% coverage
- [ ] CI pipeline green on all platforms
- [ ] Config loads from env var and file
- [ ] `transcribe --help` displays commands

---

## Sprint 2: Core Extraction (Weeks 3-4)

### Goals
- Audio Extractor module complete
- FFmpeg integration validated
- Format detection working
- Error handling for extraction failures

### User Stories

| ID | Story | Priority | Points |
|----|-------|----------|--------|
| US-2.1 | As a user, I can extract audio from an MKV file | P0 | 5 |
| US-2.2 | As a user, I see clear error if FFmpeg is not installed | P0 | 3 |
| US-2.3 | As a user, I can extract audio from MP4, AVI, MOV files | P1 | 3 |
| US-2.4 | As a user, I see progress during extraction | P1 | 2 |

### Technical Tasks

1. **Audio Extractor Module**
   - ffmpeg-python wrapper
   - Format detection (ffprobe)
   - Audio stream extraction
   - Temporary file management

2. **FFmpeg Validation**
   - Startup check with helpful error
   - Version validation (4.0+)
   - Platform-specific installation guidance

3. **Error Handling**
   - Corrupted file detection
   - Unsupported codec fallback
   - Permission errors

4. **Testing**
   - Mock FFmpeg subprocess
   - Sample file fixtures
   - Error scenario tests

### Definition of Done
- [ ] `transcribe extract video.mkv` produces audio file
- [ ] Clear error message if FFmpeg missing
- [ ] 70% coverage on extractor module
- [ ] Works on Linux, macOS, Windows

---

## Sprint 3: Transcription (Weeks 5-6)

### Goals
- Transcription Client module complete
- OpenAI Whisper API integration
- Basic transcript output (TXT)
- Rate limit handling

### User Stories

| ID | Story | Priority | Points |
|----|-------|----------|--------|
| US-3.1 | As a user, I can transcribe an MP3 file to text | P0 | 5 |
| US-3.2 | As a user, I can transcribe extracted audio from video | P0 | 3 |
| US-3.3 | As a user, I see progress during transcription | P0 | 2 |
| US-3.4 | As a user, transcription retries on transient API errors | P1 | 3 |

### Technical Tasks

1. **Transcription Client Module**
   - OpenAI SDK integration
   - Audio file upload
   - Response parsing
   - Timestamp extraction

2. **API Integration**
   - Async client setup
   - Retry with exponential backoff
   - Rate limit handling
   - Timeout configuration

3. **Output Generation**
   - Plain text formatter
   - File writing with encoding
   - Output directory handling

4. **Testing**
   - Mock OpenAI API responses
   - Error scenario tests
   - Integration tests with CLI

### Definition of Done
- [ ] `transcribe audio.mp3` produces transcript.txt
- [ ] `transcribe video.mkv` extracts and transcribes
- [ ] Retry logic works for 429 errors
- [ ] 80% coverage on transcriber module

---

## Sprint 4: Output & Batch (Weeks 7-8)

### Goals
- Output Formatter with TXT and SRT
- Batch Processor for directories
- Progress tracking for batch jobs
- Concurrent API calls

### User Stories

| ID | Story | Priority | Points |
|----|-------|----------|--------|
| US-4.1 | As a user, I can output transcript as SRT subtitle file | P0 | 5 |
| US-4.2 | As a user, I can transcribe all files in a directory | P0 | 5 |
| US-4.3 | As a user, I see overall progress for batch jobs | P0 | 3 |
| US-4.4 | As a user, I can set concurrency limit for batch processing | P1 | 2 |

### Technical Tasks

1. **Output Formatter Module**
   - TXT formatter (paragraph style)
   - SRT formatter (timestamps)
   - Timestamp formatting (HH:MM:SS,mmm)
   - File writing with proper encoding

2. **Batch Processor Module**
   - Directory scanning
   - File filtering (audio/video extensions)
   - Async task queue
   - Semaphore-controlled concurrency

3. **Progress Tracking**
   - Rich progress bars
   - Per-file and overall progress
   - ETA calculation
   - Error summary at end

4. **Testing**
   - SRT format validation
   - Batch with multiple files
   - Concurrency tests

### Definition of Done
- [ ] `transcribe audio.mp3 --format srt` produces valid SRT
- [ ] `transcribe batch ./recordings` processes all files
- [ ] Progress bar shows during batch
- [ ] 90% coverage on formatter, 60% on processor

---

## Sprint 5: Large Files (Weeks 9-10)

### Goals
- Large file chunking (>25MB)
- Resume support for interrupted jobs
- Timestamp synchronization
- Memory-efficient processing

### User Stories

| ID | Story | Priority | Points |
|----|-------|----------|--------|
| US-5.1 | As a user, I can transcribe files larger than 1GB | P1 | 8 |
| US-5.2 | As a user, I can resume an interrupted transcription | P1 | 5 |
| US-5.3 | As a user, I see chunk progress for large files | P1 | 3 |
| US-5.4 | As a user, SRT timestamps are continuous across chunks | P1 | 5 |

### Technical Tasks

1. **Chunking Implementation**
   - Time-based splitting (10-min segments)
   - FFmpeg segment command
   - Chunk boundary overlap (5 sec)
   - Memory-efficient streaming

2. **Resume Support**
   - Checkpoint file format
   - State persistence
   - Resume CLI flag
   - Cleanup of partial jobs

3. **Timestamp Synchronization**
   - Cumulative offset tracking
   - Overlap deduplication
   - Transcript merging
   - SRT continuity validation

4. **Testing**
   - Large file fixtures (generate programmatically)
   - Interrupt/resume scenarios
   - Timestamp accuracy validation

### Definition of Done
- [ ] 2-hour audio file transcribes successfully
- [ ] `--resume` continues interrupted job
- [ ] SRT plays correctly with original video
- [ ] Memory usage <512MB for 2GB file

---

## Sprint 6: Polish (Weeks 11-12)

### Goals
- 60% test coverage achieved
- Documentation complete
- Installation guides tested
- Release candidate ready

### User Stories

| ID | Story | Priority | Points |
|----|-------|----------|--------|
| US-6.1 | As a new user, I can install and use the tool in <10 minutes | P0 | 5 |
| US-6.2 | As a user, I can find troubleshooting help in documentation | P0 | 3 |
| US-6.3 | As a user, error messages tell me how to fix problems | P0 | 3 |
| US-6.4 | As a developer, I can contribute with clear guidelines | P1 | 2 |

### Technical Tasks

1. **Testing Completion**
   - Reach 60% overall coverage
   - E2E tests with real files
   - Platform-specific validation
   - Performance benchmarks

2. **Documentation**
   - README with quick start
   - Installation guides (Linux, macOS, Windows)
   - Troubleshooting guide
   - API key setup guide

3. **Error Message Polish**
   - Actionable error messages
   - Helpful suggestions
   - Link to documentation

4. **Release Preparation**
   - Version 0.1.0 tagging
   - CHANGELOG
   - PyPI packaging
   - GitHub Release

### Definition of Done
- [ ] 60% coverage achieved
- [ ] README complete with examples
- [ ] Windows installation tested with 3 users
- [ ] All P0 NFRs validated
- [ ] Release candidate tagged

---

## Quality Gates

### Per-Sprint Gates

| Gate | Criteria |
|------|----------|
| Code Quality | black, flake8, mypy pass |
| Test Coverage | Sprint target met |
| CI Pipeline | All platforms green |
| PR Review | At least 1 approval |
| Security | pip-audit clean (no HIGH/CRITICAL) |

### IOC Gate (End of Sprint 6)

| Criterion | Target |
|-----------|--------|
| Test Coverage | 60% overall |
| P0 NFRs | 14/14 validated |
| CI Pipeline | 100% green |
| Documentation | README, install guides, troubleshooting |
| Platform Support | Linux, macOS, Windows tested |
| User Validation | 3+ early adopters successful |

---

## Risk Monitoring

| Risk | Sprint Focus | Validation |
|------|--------------|------------|
| RISK-002: FFmpeg | Sprint 2 | Windows installation with 3 users |
| RISK-003: Large Files | Sprint 5 | 2-hour file transcription |
| RISK-008: Rate Limits | Sprint 4 | Batch of 20 files |
| RISK-009: Format Compatibility | Sprint 2-3 | 15+ format test matrix |

---

## Team Capacity

| Role | Allocation | Sprints |
|------|------------|---------|
| Lead Developer | 40% | All |
| Developer 2 | 30% | Sprint 2-5 |
| QA/Test | 20% | Sprint 4-6 |
| Tech Writer | 20% | Sprint 6 |

**Velocity Assumption**: 15-20 points per sprint

---

## Communication

| Cadence | Activity |
|---------|----------|
| Daily | Async standup (Slack) |
| Bi-weekly | Sprint review + planning |
| Weekly | Risk/blocker review |
| Monthly | Stakeholder update |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Sprint Velocity | 15-20 pts | Story points completed |
| Test Coverage | 60% | pytest-cov |
| Bug Escape Rate | <5 critical | Post-release defects |
| User Satisfaction | 80% adoption | Month 2 survey |

---

## Related Documents

- ABM Report: .aiwg/reports/abm-report.md
- SAD: .aiwg/architecture/software-architecture-doc.md
- Master Test Plan: .aiwg/testing/master-test-plan.md
- NFRs: .aiwg/requirements/non-functional-requirements.md

---

**Plan Status**: APPROVED
**Next Review**: Sprint 1 Retrospective
