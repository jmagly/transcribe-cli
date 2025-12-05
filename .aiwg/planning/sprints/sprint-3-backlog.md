# Sprint 3 Backlog

**Sprint**: 3 - Transcription
**Duration**: Weeks 5-6
**Goal**: Transcription Client module with OpenAI Whisper API integration
**Velocity Target**: 13 points

---

## Sprint Goal

Implement the core transcription functionality:
- OpenAI Whisper API integration
- Audio file transcription to TXT
- Retry logic for transient errors
- Progress feedback during transcription

---

## User Stories

### US-3.1: Transcribe MP3 to Text (P0, 5 pts)
**As a** user
**I want to** transcribe an MP3 file to text
**So that** I can get a written transcript of audio

**Acceptance Criteria**:
- [ ] `transcribe audio.mp3` produces transcript.txt
- [ ] Output saved to current directory or --output-dir
- [ ] UTF-8 encoding for international characters
- [ ] File overwrites with warning if exists

**Tasks**:
- [ ] Create TranscriptionClient class in core/transcriber.py
- [ ] Implement OpenAI Whisper API call
- [ ] Create TXT output formatter
- [ ] Wire transcribe CLI command

---

### US-3.2: Transcribe Video Files (P0, 3 pts)
**As a** user
**I want to** transcribe video files directly
**So that** I don't need to manually extract audio first

**Acceptance Criteria**:
- [ ] `transcribe video.mkv` extracts audio and transcribes
- [ ] Temporary audio file cleaned up after transcription
- [ ] Both audio and video files supported in one command

**Tasks**:
- [ ] Integrate extractor with transcriber
- [ ] Handle temporary file lifecycle
- [ ] Support all video formats from Sprint 2

---

### US-3.3: Transcription Progress (P0, 2 pts)
**As a** user
**I want to** see progress during transcription
**So that** I know the tool is working

**Acceptance Criteria**:
- [ ] Show "Transcribing..." message during API call
- [ ] Show file info (duration, size) before transcription
- [ ] Success message with output file path

**Tasks**:
- [ ] Add rich console progress indicators
- [ ] Display file metadata before transcription
- [ ] Show completion stats

---

### US-3.4: Retry on API Errors (P1, 3 pts)
**As a** user
**I want to** have transcription retry on transient errors
**So that** temporary API issues don't fail my job

**Acceptance Criteria**:
- [ ] Retry on 429 (rate limit) with backoff
- [ ] Retry on 500/502/503 (server errors)
- [ ] Max 3 retries with exponential backoff
- [ ] Clear error message after max retries

**Tasks**:
- [ ] Implement retry decorator with tenacity
- [ ] Configure exponential backoff (1s, 2s, 4s)
- [ ] Handle specific HTTP error codes
- [ ] Add timeout configuration

---

## Technical Debt

None (new module)

---

## Risks / Blockers

| Risk | Mitigation | Status |
|------|------------|--------|
| API key not set | Clear error message with setup guide | Implement |
| Rate limiting | Retry with backoff | Implement |
| Large file >25MB | Sprint 5 chunking | Deferred |

---

## Definition of Done (Sprint)

- [ ] All P0 stories complete
- [ ] All tests passing on CI
- [ ] Coverage >60% overall
- [ ] transcribe command functional for audio/video
- [ ] Retry logic validated

---

## Sprint Progress

| Story | Points | Status | Assignee |
|-------|--------|--------|----------|
| US-3.1 | 5 | Not Started | - |
| US-3.2 | 3 | Not Started | - |
| US-3.3 | 2 | Not Started | - |
| US-3.4 | 3 | Not Started | - |

**Total Points**: 13
**Completed**: 0
**Remaining**: 13

---

## Notes

- Uses OpenAI SDK (openai package)
- Whisper API endpoint: /v1/audio/transcriptions
- Max file size: 25MB (larger files deferred to Sprint 5)
- tenacity library for retry logic

---

## Next Sprint Preview (Sprint 4)

Focus: Output Formatter & Batch Processing
- SRT subtitle format
- Batch directory processing
- Concurrent API calls
