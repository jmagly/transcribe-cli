# CLAUDE.md

This file provides guidance to Claude Code when working with this codebase.

## Repository Purpose

CLI tool for transcribing audio and video files using OpenAI Whisper API.

## Tech Stack

- Languages: Python
- Runtime: Python >=3.9
- Package Manager: pip (PEP 621 via `pyproject.toml`)
- Core Libraries: OpenAI, ffmpeg-python, Typer, Rich, Pydantic
- CI/CD: GitHub Actions (1 workflow)

## Development Commands

- Install dependencies: `pip install -e .[dev]`
- Run CLI: `transcribe <input-file>`
- Run tests: `pytest`
- Coverage report (HTML): `pytest --cov=src/transcribe_cli --cov-report=html`
- Format code: `black src tests`
- Lint code: `flake8 src tests`
- Type-check: `mypy src`
- Security audit: `pip-audit`

## Testing

- Framework: Pytest
- Test Location: `tests/` (`unit/`, `integration/`, `fixtures/`)
- Config: `pyproject.toml` (`[tool.pytest.ini_options]`)
- Coverage Gate: `fail_under = 60`

## Architecture

- Primary package: `src/transcribe_cli`
- Modules: `cli`, `config`, `core`, `models`, `output`, `utils`
- Entry point: `transcribe = transcribe_cli.cli.main:app`
- Project shape: Python CLI application with layered modules for config, core transcription, models, output formatting, and utilities.

## Important Files

- `pyproject.toml`
- `README.md`
- `.github/workflows/ci.yml`
- `src/transcribe_cli/cli/main.py`
- `src/transcribe_cli/core/transcriber.py`
- `tests/`
- `.aiwg/frameworks/registry.json`

---

## Team Directives & Standards

<!-- PRESERVED SECTION - Content maintained across regeneration -->

## Project-Specific Notes

<!-- User: Add project-specific guidance, conventions, and rules below -->

<!-- /PRESERVED SECTION -->

---

## AIWG Framework Integration

This repository includes AIWG-managed framework assets and should preserve framework-driven workflows.

### Installed Frameworks

- media-curator v1.0.0
- media-marketing-kit v1.0.0
- research-complete v1.0.0
- sdlc-complete v1.0.0

### Available Agents

- `.claude/agents`: 0
- `.codex/agents`: 130

### Available Commands

- `.claude/commands`: 0

### Orchestration

Use the platform orchestrator and deployed AIWG framework instructions as the source of truth for lifecycle phases, governance, and execution patterns in this repository.

---

<!--
  USER NOTES
  Add team directives, conventions, or project-specific notes below.
  Content in this file's preserved sections is maintained during regeneration.
  Use <!-- PRESERVE --> markers for content that must be kept.
-->
