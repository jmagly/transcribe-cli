# Sprint 4 Backlog

**Sprint**: 4 - Output & Batch
**Duration**: Weeks 7-8
**Goal**: SRT output format and batch directory processing
**Velocity Target**: 15 points

---

## Sprint Goal

Implement output formatting and batch processing:
- SRT subtitle format with timestamps
- Batch processing for entire directories
- Concurrent API calls with rate limiting
- Progress tracking for batch jobs

---

## User Stories

### US-4.1: SRT Subtitle Output (P0, 5 pts)
**As a** user
**I want to** output transcript as SRT subtitle file
**So that** I can use subtitles with video players

**Acceptance Criteria**:
- [ ] `transcribe audio.mp3 --format srt` produces valid SRT
- [ ] Timestamps in HH:MM:SS,mmm format
- [ ] Sequential subtitle numbering
- [ ] UTF-8 encoding for international characters

**Tasks**:
- [ ] Create SRT formatter in output/formatters.py
- [ ] Implement timestamp conversion
- [ ] Update transcribe command to support SRT
- [ ] Add SRT validation tests

---

### US-4.2: Batch Directory Processing (P0, 5 pts)
**As a** user
**I want to** transcribe all files in a directory
**So that** I can process multiple recordings at once

**Acceptance Criteria**:
- [ ] `transcribe batch ./recordings` processes all audio/video
- [ ] Skip non-media files silently
- [ ] Output files to same directory or --output-dir
- [ ] Summary at end with success/failure counts

**Tasks**:
- [ ] Create BatchProcessor in core/batch.py
- [ ] Implement directory scanning
- [ ] Wire batch CLI command
- [ ] Handle errors gracefully (continue on failure)

---

### US-4.3: Batch Progress Display (P0, 3 pts)
**As a** user
**I want to** see progress during batch processing
**So that** I know how much is complete

**Acceptance Criteria**:
- [ ] Progress bar showing overall completion
- [ ] Current file being processed
- [ ] Files completed / total count
- [ ] ETA for completion

**Tasks**:
- [ ] Add rich progress bar integration
- [ ] Track per-file progress
- [ ] Calculate and display ETA
- [ ] Show error count in progress

---

### US-4.4: Concurrency Control (P1, 2 pts)
**As a** user
**I want to** set concurrency limit for batch processing
**So that** I can balance speed vs API rate limits

**Acceptance Criteria**:
- [ ] --concurrency option (1-20, default 5)
- [ ] Semaphore-controlled parallel requests
- [ ] Respect rate limits

**Tasks**:
- [ ] Implement asyncio semaphore
- [ ] Add concurrency parameter to batch
- [ ] Test concurrent execution

---

## Technical Debt

None (new modules)

---

## Risks / Blockers

| Risk | Mitigation | Status |
|------|------------|--------|
| Rate limiting with high concurrency | Default to 5, max 20 | Implement |
| Large directories | Progress feedback | Implement |
| Mixed success/failure | Error summary | Implement |

---

## Definition of Done (Sprint)

- [ ] All P0 stories complete
- [ ] All tests passing on CI
- [ ] Coverage >60% overall
- [ ] SRT output validated
- [ ] Batch command functional

---

## Sprint Progress

| Story | Points | Status | Assignee |
|-------|--------|--------|----------|
| US-4.1 | 5 | Not Started | - |
| US-4.2 | 5 | Not Started | - |
| US-4.3 | 3 | Not Started | - |
| US-4.4 | 2 | Not Started | - |

**Total Points**: 15
**Completed**: 0
**Remaining**: 15

---

## Notes

- SRT format: index, timestamp line, text, blank line
- Timestamp format: HH:MM:SS,mmm --> HH:MM:SS,mmm
- Uses srt library for validation
- asyncio for concurrent processing

---

## Next Sprint Preview (Sprint 5)

Focus: Large File Support
- Chunking for files >25MB
- Resume interrupted transcriptions
- Timestamp synchronization across chunks
