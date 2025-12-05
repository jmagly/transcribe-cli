# Use Case Brief: UC-001 - Transcribe Single Audio File

---

## Use Case Identification

**UC ID**: UC-001
**Title**: Transcribe Single Audio File
**Version**: 1.0
**Priority**: P0 (Critical for MVP)
**Status**: Draft
**Date**: 2025-12-04

---

## Use Case Overview

### Actor(s)
- **Primary Actor**: Engineering Team Member
- **Supporting Actors**: OpenAI Whisper API (external system)

### Goal/Purpose
Transcribe a single audio file (MP3, AAC, FLAC, WAV, M4A) to text format, enabling quick conversion of audio content to searchable, analyzable text with minimal user effort.

### Preconditions
1. CLI tool is installed on user's system
2. Python 3.9+ is available
3. OpenAI API key is configured (environment variable `OPENAI_API_KEY` or `.env` file)
4. Valid audio file exists in a supported format (MP3, AAC, FLAC, WAV, M4A)
5. User has network connectivity to OpenAI API
6. User has sufficient API quota/credits available

### Postconditions (Success)
1. Transcript file is saved to output directory (default: `./transcripts/`)
2. User is notified of successful completion with file location
3. Original audio file remains unmodified
4. API usage is logged/tracked (if telemetry enabled)

### Postconditions (Failure)
1. Clear error message displayed to user
2. Partial transcription artifacts cleaned up (if any)
3. Original audio file remains unmodified
4. Error logged for troubleshooting

---

## Main Success Scenario

**User Action → System Response Pattern**

1. **User runs transcribe command**
   - Command: `transcribe audio.mp3`
   - User triggers transcription workflow

2. **System validates input file**
   - Check file exists at specified path
   - Verify file format is supported (MP3, AAC, FLAC, WAV, M4A)
   - Check file size against API limits (25MB)
   - Validate file is readable and not corrupted

3. **System validates environment**
   - Confirm OpenAI API key is present
   - Test API connectivity (optional health check)
   - Verify output directory exists or create it

4. **System processes audio file**
   - Display progress indicator (spinner or progress bar)
   - Upload file to Whisper API
   - Wait for transcription response
   - Show estimated time remaining (if available)

5. **System receives transcription**
   - Parse Whisper API response
   - Extract transcript text
   - Extract metadata (duration, language detected, confidence)

6. **System saves transcript**
   - Generate output filename (e.g., `audio.txt` or `audio-transcript.txt`)
   - Save plain text transcript to output directory
   - Preserve file permissions and timestamps

7. **System notifies user**
   - Display success message with file location
   - Show transcript summary (word count, duration, language)
   - Display API cost (if available)

**Example Output**:
```
Transcribing: audio.mp3 [========================================] 100%
Success: Transcript saved to ./transcripts/audio.txt
Duration: 5 minutes 32 seconds
Words: 847
Language: English
```

---

## Alternative Flows

### Alternative Flow 1: File Size Exceeds API Limit (>25MB)

**Trigger**: Step 2 - File validation detects size >25MB

1. System displays warning: "File exceeds 25MB API limit. Chunking required."
2. System automatically chunks file into segments (or prompts user for confirmation)
3. System processes each chunk sequentially
4. System merges chunk transcripts into single output file
5. System ensures timestamps are continuous across chunks
6. Resume at Step 6 (save transcript)

**Notes**: See UC-004 for detailed large file handling workflow.

### Alternative Flow 2: API Rate Limit Exceeded

**Trigger**: Step 4 - API returns 429 (Too Many Requests) error

1. System detects rate limit error
2. System displays: "Rate limit reached. Retrying in X seconds..."
3. System implements exponential backoff (e.g., 5s, 10s, 20s)
4. System retries API request up to 3 times
5. If still failing, display error and save partial state
6. User can resume later with `--resume` flag

### Alternative Flow 3: Invalid Audio File

**Trigger**: Step 2 - File validation detects unsupported format or corrupted file

1. System attempts format detection via file extension and magic bytes
2. If format is unsupported, display error: "Unsupported format: [format]. Supported: MP3, AAC, FLAC, WAV, M4A"
3. If file is corrupted, display error: "File appears corrupted. Unable to read audio data."
4. System suggests conversion using FFmpeg: `ffmpeg -i [file] audio.mp3`
5. Exit with error code 1

### Alternative Flow 4: Missing API Key

**Trigger**: Step 3 - API key validation fails

1. System checks for `OPENAI_API_KEY` environment variable
2. If missing, check for `.env` file in current directory
3. If still missing, display error: "OpenAI API key not found. Set OPENAI_API_KEY environment variable or create .env file."
4. Provide documentation link for API key setup
5. Exit with error code 2

### Alternative Flow 5: Network Connectivity Issues

**Trigger**: Step 4 - API request fails due to network error

1. System detects network timeout or connection error
2. Display error: "Network error. Unable to reach OpenAI API. Check your connection."
3. Implement retry logic (2 retries with 3-second delay)
4. If persistent, suggest checking firewall/proxy settings
5. Exit with error code 3 if unrecoverable

---

## Success Criteria

### Functional Criteria
1. Transcript file is generated with >90% accuracy (Whisper API quality baseline)
2. Original audio content is preserved exactly in text form (no summarization)
3. Output file is valid UTF-8 plain text
4. File naming follows predictable pattern: `[input-filename].txt` or `[input-filename]-transcript.txt`

### Performance Criteria
1. Processing time <5 minutes for files up to 30 minutes duration (API-dependent)
2. User's active involvement time <30 seconds (command execution + output verification)
3. System memory usage <500MB for files <100MB
4. Disk I/O minimized (direct API upload, no unnecessary temp files)

### Usability Criteria
1. Single command execution with sensible defaults: `transcribe audio.mp3`
2. Clear progress indicator during processing (not silent)
3. Error messages are actionable (explain what went wrong and how to fix)
4. Success message includes next steps (file location, how to view)

### Quality Criteria
1. No data loss (transcript matches audio content)
2. Timestamps accurate if included in output format
3. Language detection automatic (no manual language specification required)
4. Graceful handling of silence/background noise (Whisper handles automatically)

---

## Non-Functional Requirements

### NFR-001: Reliability
- **Requirement**: 95%+ success rate for valid audio files under normal conditions
- **Validation**: Track success/failure ratio in logs, monitor API error rates

### NFR-002: Performance
- **Requirement**: Total workflow time (user action → transcript ready) <5 minutes for 30-minute audio file
- **Validation**: Measure end-to-end time in testing, track 95th percentile latency

### NFR-003: Usability
- **Requirement**: New users can complete first transcription within 10 minutes (including reading docs)
- **Validation**: User testing with 3-5 team members, measure time-to-first-success

### NFR-004: Security
- **Requirement**: API key never logged or displayed in plain text
- **Validation**: Code review, log inspection, ensure masking in verbose mode

### NFR-005: Error Handling
- **Requirement**: All error conditions produce clear, actionable messages with suggested fixes
- **Validation**: Test each error path, review message clarity with 2-3 users

---

## Assumptions

1. User has stable internet connection (required for API access)
2. Audio file is in English or common language supported by Whisper (100+ languages)
3. Audio quality is sufficient for transcription (not heavily distorted)
4. User has disk space for transcript output (typically <1% of audio file size)
5. OpenAI API uptime is >99.9% (per OpenAI SLA)

---

## Dependencies

### Technical Dependencies
- **OpenAI Whisper API**: Core transcription engine (external service)
- **Python `openai` library**: API client for making requests
- **Python `rich` library**: Progress bar and formatted output
- **Network connectivity**: Required for API communication

### Logical Dependencies
- **API Key Configuration**: Must be completed before first use
- **Output directory**: Created automatically if missing, but requires write permissions

---

## Related Use Cases

- **UC-002**: Extract and Transcribe Video File (extends this use case with audio extraction)
- **UC-003**: Batch Process Directory (repeats this use case for multiple files)
- **UC-004**: Handle Large File (extends this use case with chunking logic)
- **UC-005**: Generate Timestamped Output (variant with SRT format instead of TXT)

---

## Notes and Comments

### Design Considerations
- Default output format is plain text (.txt) for simplicity
- User can specify alternative formats via `--format` flag (SRT, JSON)
- Progress indicators are critical for user experience (long API wait times)
- Verbose mode (`--verbose`) should show detailed API request/response for debugging

### Future Enhancements (Post-MVP)
- Local Whisper model support (offline mode)
- Auto-retry with exponential backoff for transient failures
- Parallel processing of multiple files (see UC-003)
- Resume support for interrupted transcriptions (checkpoint/state file)

### Testing Guidance
- **Unit Tests**: File validation, API response parsing, error handling
- **Integration Tests**: End-to-end transcription with mock API
- **Manual Tests**: Real audio files (5 min, 15 min, 30 min), various formats
- **Error Tests**: Corrupted files, missing API key, network timeout, rate limits

---

## Traceability

### Requirements Mapping
- **Vision Document Section 6**: In-Scope Features → "Direct Audio Transcription"
- **Success Metrics**: Time Savings (70% reduction), Workflow Simplification (single command)
- **Personas**: Primary Persona → Engineering Team Member → Use Case 1 (Meeting Transcription)

### Test Coverage
- **Test Plan Reference**: Master Test Plan → Unit Tests → Audio Processing Module
- **Acceptance Tests**: Defined in test strategy document (TBD)

---

## Approvals

| Role | Name | Status | Date | Comments |
|------|------|--------|------|----------|
| Requirements Analyst | Claude (Requirements Analyst) | Draft | 2025-12-04 | Initial draft for review |
| Product Owner | TBD | Pending | - | - |
| Tech Lead | TBD | Pending | - | - |

---

**Document End**
