# Sprint 2 Backlog

**Sprint**: 2 - Core Extraction
**Duration**: Weeks 3-4
**Goal**: Audio Extractor module complete with FFmpeg integration
**Velocity Target**: 13 points

---

## Sprint Goal

Implement the core audio extraction functionality:
- FFmpeg validation and error handling
- Audio extraction from video files (MKV, MP4, AVI, MOV)
- Format detection using ffprobe
- Progress feedback during extraction

---

## User Stories

### US-2.1: Extract Audio from MKV (P0, 5 pts)
**As a** user
**I want to** extract audio from an MKV file
**So that** I can prepare it for transcription

**Acceptance Criteria**:
- [ ] `transcribe extract video.mkv` produces audio file
- [ ] Extracted audio is valid MP3/WAV
- [ ] Output filename matches input (video.mp3)
- [ ] Custom output path with -o flag

**Tasks**:
- [ ] Create AudioExtractor class in core/extractor.py
- [ ] Implement ffmpeg-python extraction
- [ ] Handle audio stream selection
- [ ] Add temporary file cleanup

---

### US-2.2: FFmpeg Missing Error (P0, 3 pts)
**As a** user
**I want to** see a clear error if FFmpeg is not installed
**So that** I know how to fix the problem

**Acceptance Criteria**:
- [ ] Clear error message if FFmpeg not found
- [ ] Installation instructions for Linux, macOS, Windows
- [ ] Error includes link to documentation
- [ ] Version check for FFmpeg 4.0+

**Tasks**:
- [ ] Implement FFmpeg detection in core/ffmpeg.py
- [ ] Add version validation
- [ ] Create platform-specific installation guidance
- [ ] Write tests for FFmpeg detection

---

### US-2.3: Multiple Video Formats (P1, 3 pts)
**As a** user
**I want to** extract audio from MP4, AVI, MOV files
**So that** I can work with common video formats

**Acceptance Criteria**:
- [ ] MP4 extraction works
- [ ] AVI extraction works
- [ ] MOV extraction works
- [ ] Format detection via ffprobe

**Tasks**:
- [ ] Implement format detection using ffprobe
- [ ] Support common video containers
- [ ] Handle files with no audio stream
- [ ] Add format validation tests

---

### US-2.4: Extraction Progress (P1, 2 pts)
**As a** user
**I want to** see progress during extraction
**So that** I know the tool is working

**Acceptance Criteria**:
- [ ] Progress indicator during extraction
- [ ] Duration-based progress percentage
- [ ] Completion message with file path

**Tasks**:
- [ ] Add rich progress bar
- [ ] Parse FFmpeg progress output
- [ ] Show file size and duration info

---

## Technical Debt

None (new module)

---

## Risks / Blockers

| Risk | Mitigation | Status |
|------|------------|--------|
| FFmpeg not on Windows CI | choco install step | Resolved in Sprint 1 |
| Unusual codec failures | Fallback to subprocess | Monitor |
| Large file memory | Streaming extraction | Sprint 5 |

---

## Definition of Done (Sprint)

- [ ] All P0 stories complete
- [ ] All tests passing on CI
- [ ] Coverage >50% overall
- [ ] FFmpeg detection works on all platforms
- [ ] extract command functional

---

## Sprint Progress

| Story | Points | Status | Assignee |
|-------|--------|--------|----------|
| US-2.1 | 5 | Not Started | - |
| US-2.2 | 3 | Not Started | - |
| US-2.3 | 3 | Not Started | - |
| US-2.4 | 2 | Not Started | - |

**Total Points**: 13
**Completed**: 0
**Remaining**: 13

---

## Notes

- Uses ffmpeg-python library per ADR-001
- FFmpeg 4.0+ required
- Output format: MP3 at 192kbps (optimal for Whisper API)

---

## Next Sprint Preview (Sprint 3)

Focus: Transcription Client
- OpenAI Whisper API integration
- Basic TXT output
- Rate limit handling
