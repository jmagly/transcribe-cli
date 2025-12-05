# Use Case Brief: UC-005 - Generate Timestamped Output (SRT)

---

## Use Case Identification

**UC ID**: UC-005
**Title**: Generate Timestamped Output (SRT)
**Version**: 1.0
**Priority**: P1 (Important for MVP - high user value)
**Status**: Draft
**Date**: 2025-12-04

---

## Use Case Overview

### Actor(s)
- **Primary Actor**: Content Creator needing subtitles (video editor, podcaster, educator)
- **Supporting Actors**:
  - FFmpeg (local system utility, if video input)
  - OpenAI Whisper API (external system, provides timestamps)

### Goal/Purpose
Generate timestamped subtitle files (SRT format) from audio or video files, enabling content creators to add accurate subtitles to videos for accessibility, SEO, and viewer engagement.

### Preconditions
1. CLI tool is installed on user's system
2. Python 3.9+ is available
3. FFmpeg is installed and accessible in PATH (if video input)
4. OpenAI API key is configured (environment variable `OPENAI_API_KEY` or `.env` file)
5. Valid media file exists (audio or video)
6. User has network connectivity to OpenAI API
7. User specifies SRT output format via `--format srt` flag

### Postconditions (Success)
1. Valid SRT subtitle file generated and saved to output directory
2. SRT file is compatible with video players and editing software (VLC, Premiere, Final Cut)
3. Timestamps are accurate (synchronized with audio/video)
4. Original media file remains unmodified
5. User can immediately use SRT file in video editing workflow

### Postconditions (Failure)
1. Clear error message displayed to user
2. Partial SRT file cleaned up (if transcription incomplete)
3. Original media file remains unmodified
4. Error logged for troubleshooting

---

## Main Success Scenario

**User Action → System Response Pattern**

1. **User runs transcribe command with SRT format**
   - Command: `transcribe video.mkv --format srt`
   - User triggers transcription workflow with SRT output

2. **System validates input file**
   - Check file exists and is readable
   - Verify file format is supported (audio or video)
   - If video: Validate audio track exists (FFmpeg probe)

3. **System validates environment**
   - Confirm OpenAI API key is present
   - If video: Confirm FFmpeg is installed
   - Verify output directory exists or create it

4. **System processes media file**
   - If video: Extract audio (UC-002 logic)
   - Display progress: "Transcribing with timestamps..."
   - Upload audio to Whisper API with `timestamp_granularity` parameter
   - Specify: Request word-level or segment-level timestamps (Whisper API supports both)

5. **System receives timestamped transcription**
   - Parse Whisper API response
   - Extract transcript segments with timestamps:
     ```json
     {
       "segments": [
         {"id": 1, "start": 0.0, "end": 3.5, "text": "Welcome to the tutorial."},
         {"id": 2, "start": 3.8, "end": 7.2, "text": "Today we'll cover Python basics."}
       ]
     }
     ```
   - Extract metadata (duration, language detected)

6. **System converts to SRT format**
   - Transform JSON segments to SRT structure:
     ```
     1
     00:00:00,000 --> 00:00:03,500
     Welcome to the tutorial.

     2
     00:00:03,800 --> 00:00:07,200
     Today we'll cover Python basics.
     ```
   - Format timestamps as SRT-compliant (HH:MM:SS,mmm)
   - Number segments sequentially (1, 2, 3, ...)
   - Ensure blank lines between segments (SRT spec requirement)

7. **System saves SRT file**
   - Generate output filename: `video.srt` (matches input basename)
   - Save SRT file to output directory
   - Validate SRT syntax (proper formatting, sequential numbering)

8. **System notifies user**
   - Display success message with file location
   - Show transcript summary (segment count, duration, language)
   - Provide usage tip: "Use with video: video.mkv + video.srt in video player"

**Example Output**:
```
Extracting audio from video.mkv [================================] 100%
Transcribing with timestamps [===================================] 100%

Success: SRT subtitles saved to ./transcripts/video.srt
Duration: 5 minutes 32 seconds
Segments: 87
Language: English

Usage: Load video.mkv and video.srt in VLC or video editor.
```

**Example SRT File (`video.srt`)**:
```
1
00:00:00,000 --> 00:00:03,500
Welcome to the tutorial.

2
00:00:03,800 --> 00:00:07,200
Today we'll cover Python basics.

3
00:00:07,500 --> 00:00:12,100
First, let's install Python on your system.

...
```

---

## Alternative Flows

### Alternative Flow 1: Generate Multiple Output Formats

**Trigger**: User specifies multiple formats: `--format srt,txt,json`

1. System processes media file once (single API call)
2. Generate outputs in parallel:
   - `video.srt`: SRT subtitles (timestamped segments)
   - `video.txt`: Plain text transcript (no timestamps)
   - `video.json`: JSON with full API response (segments, metadata, confidence scores)
3. Display:
   ```
   Success: Generated 3 output files:
   - ./transcripts/video.srt (SRT subtitles)
   - ./transcripts/video.txt (Plain text)
   - ./transcripts/video.json (JSON metadata)
   ```

**Use Case**: User wants both subtitles (SRT) and searchable text (TXT) from single transcription.

### Alternative Flow 2: Adjust Subtitle Segment Length

**Trigger**: User specifies `--max-segment-length N` (e.g., 42 characters for readability)

1. System receives full transcript with word-level timestamps
2. Group words into segments with maximum N characters per line
3. Split at natural phrase boundaries (commas, periods, pauses)
4. Ensure each SRT segment stays on screen long enough to read (e.g., min 2 seconds)
5. Example:
   ```
   Original: "Welcome to the tutorial. Today we'll cover Python basics and advanced features."
   Split into 2 segments:
   1. "Welcome to the tutorial."
   2. "Today we'll cover Python basics and advanced features."
   ```

**Use Case**: Optimize subtitle readability for different screen sizes or languages.

### Alternative Flow 3: Handle Large File with Chunk Timestamps

**Trigger**: File >25MB requires chunking (UC-004 logic)

1. System chunks file into segments (e.g., 3 chunks for 90-min file)
2. Transcribe each chunk with timestamps
3. Merge SRT segments with timestamp offsets:
   - Chunk 1: timestamps 00:00:00 - 00:40:00
   - Chunk 2: add 40:00 offset → timestamps 00:40:00 - 01:20:00
   - Chunk 3: add 80:00 offset → timestamps 01:20:00 - 01:30:00
4. Renumber SRT segments sequentially across chunks (1, 2, 3, ... 250)
5. Save merged SRT file

**Critical**: Timestamp offsets must be accurate to avoid sync issues.

### Alternative Flow 4: Whisper API Returns No Timestamps

**Trigger**: API response missing timestamps (rare edge case)

1. System receives transcript text but no segment timestamps
2. Display warning: "Timestamps not available. Falling back to plain text output."
3. Save as TXT instead of SRT
4. Suggest user retry or contact support

**Mitigation**: Ensure API request explicitly sets `timestamp_granularity=segment` parameter.

### Alternative Flow 5: Generate VTT Format (Future Enhancement)

**Trigger**: User specifies `--format vtt` (WebVTT for HTML5 video)

1. System transcribes with timestamps (same as SRT)
2. Convert to VTT format instead of SRT:
   ```
   WEBVTT

   00:00:00.000 --> 00:00:03.500
   Welcome to the tutorial.

   00:00:03.800 --> 00:00:07.200
   Today we'll cover Python basics.
   ```
3. VTT differences from SRT:
   - Header: `WEBVTT` (first line)
   - Timestamps: use `.` instead of `,` for milliseconds
   - No segment numbering required
4. Save as `video.vtt`

**Notes**: Defer VTT to v2 unless user feedback prioritizes it; SRT covers most use cases.

---

## Success Criteria

### Functional Criteria
1. SRT file is valid and parseable by standard video players (VLC, Windows Media Player)
2. Timestamps are accurate (sync with audio/video within ±0.5 seconds)
3. Segment text is readable (no mid-word breaks, proper punctuation)
4. SRT format adheres to specification (sequential numbering, blank lines, timestamp format)

### Performance Criteria
1. SRT generation adds negligible overhead vs. plain text (<1 second for formatting)
2. Large files (90+ min) produce accurate timestamps across all chunks
3. Total workflow time (extraction + transcription + SRT formatting) <5 minutes for 30-minute video

### Usability Criteria
1. Default behavior: If video input and no `--format` specified, prompt user: "Generate SRT subtitles? (y/n)"
2. SRT file is automatically named to match input file (e.g., `video.mkv` → `video.srt`)
3. Success message includes usage tip (how to use SRT with video player/editor)

### Quality Criteria
1. Transcription accuracy >90% (same as plain text baseline)
2. Timestamp synchronization tested with manual spot-checks (beginning, middle, end)
3. SRT file loads without errors in at least 3 video players (VLC, Premiere, Final Cut)
4. Subtitle readability meets best practices (max 2 lines, max 42 chars/line, min 2-second display time)

---

## Non-Functional Requirements

### NFR-021: SRT Format Compliance
- **Requirement**: Generated SRT files are 100% compliant with SRT specification and load correctly in all major video players
- **Validation**: Test SRT files in VLC, Windows Media Player, QuickTime, Premiere Pro, Final Cut Pro

### NFR-022: Timestamp Accuracy
- **Requirement**: Timestamps are accurate to within ±0.5 seconds of actual audio/video timing
- **Validation**: Manual spot-checks at 0%, 25%, 50%, 75%, 100% of file duration

### NFR-023: Subtitle Readability
- **Requirement**: SRT segments are formatted for readability (max 2 lines, 42 chars/line, min 2-second duration)
- **Validation**: Manual review of sample SRT files, user feedback on readability

### NFR-024: Large File Timestamp Continuity
- **Requirement**: Chunked files (UC-004) produce seamless timestamps with no gaps or overlaps at chunk boundaries
- **Validation**: Test 90-minute file chunked into 3 segments, verify timestamp continuity at 40:00 and 80:00 boundaries

### NFR-025: Multi-Format Output Efficiency
- **Requirement**: Generating multiple formats (SRT, TXT, JSON) from single transcription does not require multiple API calls
- **Validation**: Monitor API request count, verify single transcription produces all formats

---

## Assumptions

1. Whisper API consistently returns segment-level timestamps (current API behavior)
2. User's video editing software supports standard SRT format (most do)
3. User's content is in English or common language (Whisper supports 100+ languages with timestamps)
4. Segment durations (start/end times) are accurate enough for professional use (±0.5s tolerance)

---

## Dependencies

### Technical Dependencies
- **UC-001 (Transcribe Single Audio File)**: Core transcription logic with timestamp extraction
- **UC-002 (Extract and Transcribe Video File)**: Video extraction logic (if video input)
- **UC-004 (Handle Large File)**: Chunking logic with timestamp offset calculation
- **OpenAI Whisper API**: Must support `timestamp_granularity` parameter
- **Python `datetime` module**: Timestamp formatting (HH:MM:SS,mmm)

### Logical Dependencies
- **API Key Configuration**: Inherited from UC-001
- **FFmpeg Installation**: Required if video input (inherited from UC-002)

---

## Related Use Cases

- **UC-001**: Transcribe Single Audio File (underlying transcription with timestamps)
- **UC-002**: Extract and Transcribe Video File (common workflow: video → SRT subtitles)
- **UC-003**: Batch Process Directory (batch generation of SRT files for video series)
- **UC-004**: Handle Large File (timestamp merging across chunks)

---

## Notes and Comments

### Design Considerations
- **Timestamp Granularity**: Whisper API supports word-level and segment-level timestamps; segment-level is sufficient for SRT
- **SRT Formatting**: Strict specification requires sequential numbering, blank lines, HH:MM:SS,mmm format
- **Readability Optimization**: Split long segments to improve subtitle readability (max 2 lines, 42 chars/line)
- **Multiple Format Support**: Single API call should produce SRT, TXT, JSON outputs (efficient)

### Implementation Notes
- **Whisper API Request Parameter**: `timestamp_granularity="segment"` (or `word` for fine-grained control)
- **SRT Timestamp Format**: `HH:MM:SS,mmm` (e.g., `00:05:32,150`)
- **Timestamp Conversion**: Convert seconds to HH:MM:SS,mmm format:
  ```python
  def seconds_to_srt_time(seconds):
      hours = int(seconds // 3600)
      minutes = int((seconds % 3600) // 60)
      secs = int(seconds % 60)
      millis = int((seconds % 1) * 1000)
      return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
  ```
- **SRT Structure**:
  ```
  <segment_number>
  <start_time> --> <end_time>
  <subtitle_text>
  <blank_line>
  ```

### Future Enhancements (Post-MVP)
- VTT (WebVTT) format for HTML5 video (web-native subtitles)
- Word-level timestamp highlighting (karaoke-style captions)
- Automatic segment length optimization (readability algorithms)
- Burn-in subtitles directly into video file (FFmpeg integration)
- Multi-language subtitle generation (translate SRT via GPT API)

### Testing Guidance
- **Unit Tests**: Timestamp conversion, SRT formatting, segment numbering
- **Integration Tests**: End-to-end transcription with SRT output, validate format
- **Manual Tests**: Load SRT files in VLC, Premiere, Final Cut; verify sync
- **Error Tests**: Missing timestamps in API response, invalid timestamp format
- **Compatibility Tests**: Test SRT files across 5+ video players/editors

### SRT Specification Reference
- **Format**: https://www.matroska.org/technical/subtitles.html
- **Structure**: Numbered segments, HH:MM:SS,mmm timestamps, blank line separators
- **Encoding**: UTF-8 (ensure proper handling of special characters)

---

## Traceability

### Requirements Mapping
- **Vision Document Section 6**: In-Scope Features → "Essential Output Formats" → SRT format
- **Success Metrics**: Time Savings (single command for subtitles vs. manual timing)
- **Personas**: Secondary Persona → Content Creator/Researcher → Use Case 3 (Educational Content)
- **Constraints**: None specific (SRT generation adds minimal complexity)

### Test Coverage
- **Test Plan Reference**: Master Test Plan → Integration Tests → SRT Output Formatting Module
- **Acceptance Tests**: Defined in test strategy document (TBD)

---

## Approvals

| Role | Name | Status | Date | Comments |
|------|------|--------|------|----------|
| Requirements Analyst | Claude (Requirements Analyst) | Draft | 2025-12-04 | Initial draft for review |
| Product Owner | TBD | Pending | - | High value for content creators |
| Tech Lead | TBD | Pending | - | Review timestamp accuracy requirements |

---

**Document End**
