# Use Case Brief: UC-002 - Extract and Transcribe Video File

---

## Use Case Identification

**UC ID**: UC-002
**Title**: Extract and Transcribe Video File
**Version**: 1.0
**Priority**: P0 (Critical for MVP)
**Status**: Draft
**Date**: 2025-12-04

---

## Use Case Overview

### Actor(s)
- **Primary Actor**: Engineering Team Member
- **Supporting Actors**:
  - FFmpeg (local system utility)
  - OpenAI Whisper API (external system)

### Goal/Purpose
Extract audio from a video file (primarily MKV format from meeting recordings) and transcribe it to text in a single command, eliminating the manual multi-step workflow of separate extraction and transcription.

### Preconditions
1. CLI tool is installed on user's system
2. Python 3.9+ is available
3. FFmpeg is installed and accessible in system PATH
4. OpenAI API key is configured (environment variable `OPENAI_API_KEY` or `.env` file)
5. Valid video file exists in a supported format (primarily MKV, but also MP4, AVI, MOV with audio tracks)
6. User has network connectivity to OpenAI API
7. User has sufficient disk space for temporary extracted audio file
8. User has sufficient API quota/credits available

### Postconditions (Success)
1. Transcript file is saved to output directory (default: `./transcripts/`)
2. Intermediate audio file is cleaned up (unless `--keep-audio` flag specified)
3. User is notified of successful completion with file location
4. Original video file remains unmodified
5. API usage is logged/tracked (if telemetry enabled)

### Postconditions (Failure)
1. Clear error message displayed to user
2. Partial extraction/transcription artifacts cleaned up
3. Original video file remains unmodified
4. Error logged for troubleshooting

---

## Main Success Scenario

**User Action → System Response Pattern**

1. **User runs transcribe command on video file**
   - Command: `transcribe video.mkv`
   - User triggers integrated extraction + transcription workflow

2. **System validates input file**
   - Check file exists at specified path
   - Verify file format (detect video container: MKV, MP4, AVI, MOV)
   - Check file has audio track (FFmpeg probe)
   - Validate file is readable and not corrupted

3. **System validates environment**
   - Confirm FFmpeg is installed and in PATH
   - Confirm OpenAI API key is present
   - Test API connectivity (optional health check)
   - Verify output directory exists or create it
   - Check temp directory has adequate space for extracted audio

4. **System extracts audio from video**
   - Display progress: "Extracting audio from video.mkv..."
   - Run FFmpeg command: `ffmpeg -i video.mkv -vn -acodec libmp3lame audio_temp.mp3`
   - Show extraction progress bar (FFmpeg output parsing)
   - Validate extracted audio file is valid
   - Display: "Audio extraction complete: 5m 32s duration"

5. **System transcribes extracted audio**
   - Display progress: "Transcribing audio..."
   - Upload extracted audio to Whisper API (reuse UC-001 logic)
   - Wait for transcription response
   - Show estimated time remaining

6. **System receives transcription**
   - Parse Whisper API response
   - Extract transcript text
   - Extract metadata (duration, language detected, confidence)

7. **System saves transcript and cleans up**
   - Generate output filename (e.g., `video.txt` or `video-transcript.txt`)
   - Save plain text transcript to output directory
   - Delete temporary extracted audio file (unless `--keep-audio` specified)
   - Preserve file permissions and timestamps

8. **System notifies user**
   - Display success message with file location
   - Show transcript summary (word count, duration, language)
   - If `--keep-audio`, show location of extracted audio file
   - Display API cost (if available)

**Example Output**:
```
Extracting audio from video.mkv [====================================] 100%
Audio extracted: 5 minutes 32 seconds

Transcribing audio [=============================================] 100%
Success: Transcript saved to ./transcripts/video.txt
Duration: 5 minutes 32 seconds
Words: 847
Language: English

Cleaned up temporary audio file.
```

---

## Alternative Flows

### Alternative Flow 1: FFmpeg Not Installed

**Trigger**: Step 3 - FFmpeg validation fails (command not found)

1. System checks for FFmpeg in PATH
2. If not found, display error: "FFmpeg not found. FFmpeg is required for video processing."
3. Provide platform-specific installation guidance:
   - **Linux**: `sudo apt install ffmpeg` or `sudo yum install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Windows**: Link to FFmpeg download page and PATH configuration guide
4. Exit with error code 10 (FFmpeg missing)

**Notes**: This is a critical blocker. Consider creating a startup check with detailed troubleshooting link.

### Alternative Flow 2: Video File Has No Audio Track

**Trigger**: Step 2 - FFmpeg probe detects no audio streams

1. System runs: `ffmpeg -i video.mkv` and parses output
2. If no audio streams found, display error: "No audio track found in video.mkv"
3. Suggest checking file with: `ffmpeg -i video.mkv` for stream info
4. Exit with error code 11 (No audio track)

### Alternative Flow 3: Audio Extraction Fails

**Trigger**: Step 4 - FFmpeg extraction command fails

1. System runs FFmpeg extraction command
2. FFmpeg returns non-zero exit code
3. System captures FFmpeg error output
4. Display error: "Audio extraction failed: [FFmpeg error message]"
5. Common causes:
   - Unsupported codec (suggest re-encoding: `ffmpeg -i video.mkv -acodec libmp3lame output.mp3`)
   - Corrupted video file (suggest file repair tools)
   - Disk space insufficient (show required vs. available)
6. Exit with error code 12 (Extraction failed)

### Alternative Flow 4: Keep Extracted Audio (User Flag)

**Trigger**: User specifies `--keep-audio` flag

1. Execute Steps 1-6 normally
2. At Step 7, skip temporary file deletion
3. Save extracted audio to output directory: `./transcripts/video.mp3`
4. Notify user of both transcript and audio file locations:
   ```
   Success: Transcript saved to ./transcripts/video.txt
   Extracted audio saved to ./transcripts/video.mp3
   ```
5. User can reuse extracted audio without re-extraction

**Use Case**: User wants to archive audio separately or use for other purposes.

### Alternative Flow 5: Large Video File (>1GB)

**Trigger**: Step 2 - File validation detects size >1GB

1. System displays warning: "Large video file detected (1.2 GB). Extraction may take several minutes."
2. Provide time estimate based on file size (e.g., ~2 min per GB)
3. Show detailed FFmpeg progress during extraction
4. If extracted audio >25MB, automatically invoke chunking (see UC-004)
5. Resume at Step 6 (transcribe)

**Notes**: Video files are often much larger than extracted audio (e.g., 1GB video → 50MB audio).

### Alternative Flow 6: Multiple Audio Tracks in Video

**Trigger**: Step 2 - FFmpeg probe detects multiple audio streams

1. System detects 2+ audio tracks (e.g., English + Spanish, or stereo + mono)
2. Display prompt: "Multiple audio tracks found. Select track to extract:"
   - Track 1: English (stereo)
   - Track 2: Spanish (stereo)
3. User selects track number (default: Track 1)
4. FFmpeg extracts selected track: `ffmpeg -i video.mkv -map 0:a:[track] audio.mp3`
5. Resume at Step 5 (transcribe)

**Notes**: Defer to v2 if complex; MVP can default to first audio track.

---

## Success Criteria

### Functional Criteria
1. Audio extracted successfully from video file with no data loss
2. Extracted audio quality matches original (no transcoding artifacts)
3. Transcript file is generated with >90% accuracy (Whisper API quality baseline)
4. Temporary files cleaned up automatically (no disk clutter)
5. `--keep-audio` flag preserves extracted audio for reuse

### Performance Criteria
1. Audio extraction time <2 minutes for 1-hour video (depends on CPU/disk speed)
2. Total workflow time (extraction + transcription) <7 minutes for 30-minute video
3. User's active involvement time <30 seconds (command execution + output verification)
4. Disk space temporarily required: approximately video file size * 10% for audio

### Usability Criteria
1. Single command execution: `transcribe video.mkv` (no separate extraction step)
2. Clear progress indicators for both extraction and transcription phases
3. FFmpeg installation errors provide platform-specific guidance
4. Error messages distinguish between extraction failures vs. transcription failures

### Quality Criteria
1. Supports common video formats: MKV (primary), MP4, AVI, MOV
2. Handles various audio codecs embedded in video: AAC, MP3, FLAC, Opus
3. No audio quality degradation during extraction (lossless or high-quality transcode)
4. Graceful handling of edge cases (no audio track, corrupted file, unsupported codec)

---

## Non-Functional Requirements

### NFR-006: FFmpeg Dependency Transparency
- **Requirement**: User receives clear, actionable guidance if FFmpeg is missing
- **Validation**: Test on fresh system installations (Windows, macOS, Linux) without FFmpeg

### NFR-007: Disk Space Management
- **Requirement**: Temporary audio files cleaned up automatically, even on errors
- **Validation**: Test disk usage before/after failed transcriptions, verify cleanup

### NFR-008: Multi-Phase Progress Visibility
- **Requirement**: User sees distinct progress for extraction vs. transcription (not silent)
- **Validation**: User testing confirms clarity of two-phase workflow

### NFR-009: Format Compatibility
- **Requirement**: Supports 95%+ of team's video files without manual conversion
- **Validation**: Test with real meeting recordings (Zoom, Teams, Google Meet outputs)

### NFR-010: Error Isolation
- **Requirement**: Extraction failures vs. transcription failures have distinct error codes/messages
- **Validation**: Verify error message clarity for each failure mode

---

## Assumptions

1. FFmpeg is installable on user's system (not blocked by corporate policies)
2. Video files have at least one audio track (meeting recordings typically do)
3. User has sufficient disk space for temporary audio extraction (typically 10% of video size)
4. Audio quality in video is sufficient for transcription (not heavily distorted)
5. FFmpeg version is reasonably recent (v4.0+ recommended, v3.0+ minimum)

---

## Dependencies

### Technical Dependencies
- **FFmpeg**: Audio/video processing utility (external system dependency)
- **Python `subprocess` module**: For executing FFmpeg commands
- **OpenAI Whisper API**: Transcription engine (same as UC-001)
- **Python `openai` library**: API client
- **Python `rich` library**: Progress bars for multi-phase workflow
- **Disk I/O**: Temporary storage for extracted audio

### Logical Dependencies
- **UC-001 (Transcribe Single Audio File)**: Reuses transcription logic after extraction
- **FFmpeg Installation**: Must be completed before first video transcription
- **API Key Configuration**: Inherited from UC-001

---

## Related Use Cases

- **UC-001**: Transcribe Single Audio File (underlying transcription workflow)
- **UC-003**: Batch Process Directory (extends to batch video processing)
- **UC-004**: Handle Large File (extracted audio may exceed 25MB)
- **UC-005**: Generate Timestamped Output (applies to video transcription with SRT output)

---

## Notes and Comments

### Design Considerations
- **Two-Phase Workflow**: Extraction (FFmpeg) → Transcription (Whisper API)
- **Progress Indicators**: Critical for long extraction times (2+ minutes for large videos)
- **Cleanup Strategy**: Always delete temp audio unless `--keep-audio` specified
- **Error Handling**: Distinguish FFmpeg errors (extraction) from API errors (transcription)
- **Format Detection**: Use FFmpeg probe to auto-detect video format and audio codec

### Implementation Notes
- **FFmpeg Command Template**: `ffmpeg -i {input} -vn -acodec libmp3lame -q:a 2 {output}`
- **Progress Parsing**: Parse FFmpeg stderr output for time elapsed/remaining
- **Temp File Naming**: Use unique temp filenames to avoid collisions (UUID or hash-based)
- **Cleanup on Interrupt**: Register signal handlers to clean up temp files on Ctrl+C

### Future Enhancements (Post-MVP)
- Auto-detect and extract best quality audio track (if multiple tracks)
- Preserve original audio codec (lossless extraction) if already in supported format
- Parallel extraction + transcription (stream extraction to API, avoid disk I/O)
- Support for additional video formats (WEBM, FLV, OGV)

### Testing Guidance
- **Unit Tests**: FFmpeg command generation, path handling, cleanup logic
- **Integration Tests**: Mock FFmpeg execution, test error handling
- **Manual Tests**: Real MKV files from Zoom/Teams (30 min, 1 hour, 2 hours)
- **Error Tests**: Missing FFmpeg, corrupted video, no audio track, disk full
- **Platform Tests**: Validate on Windows, macOS, Linux (FFmpeg variations)

### Windows-Specific Guidance
- **Installation**: Link to https://ffmpeg.org/download.html#build-windows
- **PATH Configuration**: Guide users to add FFmpeg bin directory to system PATH
- **Common Issue**: Windows users may download FFmpeg but forget PATH setup
- **Troubleshooting**: Provide `where ffmpeg` command to verify PATH

---

## Traceability

### Requirements Mapping
- **Vision Document Section 6**: In-Scope Features → "Audio Extraction from MKV Video Files"
- **Success Metrics**: Time Savings (single command vs. multi-step), Workflow Simplification
- **Personas**: Primary Persona → Engineering Team Member → Use Case 1 (Meeting Transcription)
- **Constraints**: Technical Constraints → FFmpeg Requirement (documented mitigation)

### Test Coverage
- **Test Plan Reference**: Master Test Plan → Integration Tests → Audio Extraction Module
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
