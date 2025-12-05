# Sprint 1 Backlog

**Sprint**: 1 - Foundation
**Duration**: Weeks 1-2
**Goal**: Project setup, CI/CD operational, Config Manager complete
**Velocity Target**: 15-20 points

---

## Sprint Goal

Establish the development foundation:
- Developers can clone, install, and run tests
- CI/CD pipeline validates every PR
- Config management module complete and tested
- CLI skeleton with all command stubs

---

## User Stories

### US-1.1: Developer Environment Setup (P0, 3 pts)
**As a** developer
**I want to** clone the repo and run tests
**So that** I can contribute to the project

**Acceptance Criteria**:
- [ ] `git clone` + `pip install -e ".[dev]"` works
- [ ] `pytest` runs without errors
- [ ] Pre-commit hooks install and run
- [ ] README has setup instructions

**Tasks**:
- [x] Create pyproject.toml with dependencies
- [x] Create src/transcribe_cli package structure
- [x] Create tests directory structure
- [ ] Write README quick start section
- [ ] Verify install on clean venv

---

### US-1.2: API Key Configuration (P0, 2 pts)
**As a** developer
**I want to** configure API key via environment variable
**So that** the tool can access OpenAI Whisper API

**Acceptance Criteria**:
- [x] OPENAI_API_KEY loaded from environment
- [x] SecretStr masks key in logs/repr
- [x] Validation error if key missing
- [x] .env file support

**Tasks**:
- [x] Implement Settings class with pydantic-settings
- [x] Add SecretStr for API key
- [x] Create .env.example template
- [x] Write unit tests for config

---

### US-1.3: CI Pipeline (P0, 5 pts)
**As a** developer
**I want to** see CI pipeline run on every PR
**So that** code quality is enforced automatically

**Acceptance Criteria**:
- [x] GitHub Actions workflow created
- [x] Lint stage (black, flake8, mypy)
- [x] Test stage (pytest with coverage)
- [x] Security stage (pip-audit)
- [x] Multi-platform matrix (Ubuntu, macOS, Windows)
- [x] Python version matrix (3.9-3.12)

**Tasks**:
- [x] Create .github/workflows/ci.yml
- [x] Configure coverage reporting
- [x] Add coverage gate (60% threshold)
- [x] Create pre-commit config

---

### US-1.4: CLI Help Text (P0, 3 pts)
**As a** user
**I want to** run `transcribe --help` and see available commands
**So that** I understand how to use the tool

**Acceptance Criteria**:
- [x] `transcribe --help` shows main commands
- [x] `transcribe transcribe --help` shows transcribe options
- [x] `transcribe extract --help` shows extract options
- [x] `transcribe batch --help` shows batch options
- [x] `transcribe config --help` shows config options

**Tasks**:
- [x] Create Typer app with subcommands
- [x] Add help text for all commands
- [x] Add argument/option descriptions
- [x] Write CLI integration tests

---

### US-1.5: Version Display (P1, 1 pt)
**As a** user
**I want to** run `transcribe --version` to check installation
**So that** I can verify the tool is installed correctly

**Acceptance Criteria**:
- [x] `transcribe --version` displays version number
- [x] Version matches pyproject.toml

**Tasks**:
- [x] Add --version flag to CLI
- [x] Read version from __init__.py
- [x] Write test for version output

---

## Technical Debt

None yet (new project)

---

## Risks / Blockers

| Risk | Mitigation | Status |
|------|------------|--------|
| FFmpeg not on Windows CI | choco install step | Resolved |
| pip-audit false positives | Review findings | Monitor |

---

## Definition of Done (Sprint)

- [ ] All P0 stories complete
- [ ] All tests passing on CI
- [ ] Coverage >30% (Sprint 1 target)
- [ ] Pre-commit hooks configured
- [ ] README has setup instructions

---

## Sprint Progress

| Story | Points | Status | Assignee |
|-------|--------|--------|----------|
| US-1.1 | 3 | In Progress | - |
| US-1.2 | 2 | Complete | - |
| US-1.3 | 5 | Complete | - |
| US-1.4 | 3 | Complete | - |
| US-1.5 | 1 | Complete | - |

**Total Points**: 14
**Completed**: 11
**Remaining**: 3 (README, final verification)

---

## Notes

- Project structure created with all modules
- CI/CD pipeline ready for first push
- Config module complete with tests
- CLI skeleton with all commands stubbed
- Ready to verify installation and write README

---

## Next Sprint Preview (Sprint 2)

Focus: Audio Extractor module
- FFmpeg integration
- Format detection
- Error handling
- Platform testing
