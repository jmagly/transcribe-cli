# Sprint 5: Polish & UX

## Sprint Goal
Enhance user experience with better feedback, configuration options, and batch processing flexibility.

## User Stories

### US-5.1: Dry Run Mode
**As a** user
**I want** to preview what files will be processed
**So that** I can verify before committing to API calls

**Acceptance Criteria:**
- `--dry-run` flag shows files that would be processed
- Displays file count, total size, estimated cost
- No API calls made in dry-run mode

### US-5.2: Recursive Directory Processing
**As a** user
**I want** to process nested directories
**So that** I can transcribe organized folder structures

**Acceptance Criteria:**
- `--recursive` flag scans subdirectories
- Default behavior remains non-recursive
- Progress shows relative paths for clarity

### US-5.3: Configuration File
**As a** user
**I want** to save my preferences
**So that** I don't repeat options every time

**Acceptance Criteria:**
- Support `.transcriberc` or `transcribe.toml` config file
- Override defaults for format, concurrency, language
- CLI flags override config file settings

### US-5.4: Enhanced Feedback
**As a** user
**I want** clearer progress and error messages
**So that** I understand what's happening

**Acceptance Criteria:**
- Show file sizes and estimated duration
- Better error context with suggestions
- Summary statistics after batch processing

## Sprint Backlog

| Task | Story Points | Priority |
|------|--------------|----------|
| Implement --dry-run for batch | 3 | High |
| Implement --recursive for batch | 2 | High |
| Add config file loading | 5 | Medium |
| Enhance verbose output | 2 | Medium |
| Improve error messages | 2 | Medium |
| Add batch summary statistics | 2 | Low |
| Unit tests | 3 | High |
| Integration tests | 2 | High |

**Total: 21 Story Points**

## Definition of Done
- All features implemented with tests
- Coverage maintained above 75%
- CLI help text updated
- All existing tests pass
