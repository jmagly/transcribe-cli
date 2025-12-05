# Use Case Brief: UC-004 - Handle Large File (>25MB)

---

## Use Case Identification

**UC ID**: UC-004
**Title**: Handle Large File (>25MB)
**Version**: 1.0
**Priority**: P1 (Important for MVP - deferred complexity acceptable)
**Status**: Draft
**Date**: 2025-12-04

---

## Use Case Overview

### Actor(s)
- **Primary Actor**: Researcher with 2-hour interview recording (or content creator with podcast episode)
- **Supporting Actors**:
  - FFmpeg (local system utility, for chunking)
  - OpenAI Whisper API (external system, with 25MB limit)

### Goal/Purpose
Transcribe audio/video files larger than the Whisper API's 25MB limit by automatically chunking the file into segments, transcribing each segment, and merging results into a seamless final transcript.

### Preconditions
1. CLI tool is installed on user's system
2. Python 3.9+ is available
3. FFmpeg is installed and accessible in PATH
4. OpenAI API key is configured (environment variable `OPENAI_API_KEY` or `.env` file)
5. Valid media file exists with size >25MB (typically 60+ minutes at standard bitrates)
6. User has network connectivity to OpenAI API
7. User has adequate API quota for multiple chunk transcriptions
8. User has sufficient disk space for temporary chunk files

### Postconditions (Success)
1. Complete transcript file generated spanning all chunks
2. Timestamps are continuous across chunk boundaries (no gaps/overlaps)
3. Temporary chunk files cleaned up
4. User receives seamless transcript as if file was processed in one pass
5. Original large file remains unmodified

### Postconditions (Failure)
1. Partial transcript saved for successfully processed chunks
2. Checkpoint/state file saved to enable resume
3. Clear error message indicates which chunk failed and why
4. User can resume processing from last successful chunk
5. Temporary chunk files cleaned up (or preserved for debugging)

---

## Main Success Scenario

**User Action → System Response Pattern**

1. **User runs transcribe command on large file**
   - Command: `transcribe large-interview.mp3` (e.g., 45MB, 90 minutes)
   - User triggers transcription workflow

2. **System validates input file**
   - Check file exists and is readable
   - Verify file format is supported
   - Detect file size: 45MB
   - Display: "Large file detected (45MB). Will process in chunks."

3. **System validates environment**
   - Confirm FFmpeg is installed (required for chunking)
   - Confirm OpenAI API key is present
   - Verify sufficient disk space for chunks (~2x file size temp space)
   - Verify API quota is adequate for multiple requests

4. **System determines chunking strategy**
   - Calculate chunk size based on API limit (target: 20MB per chunk, 5MB buffer)
   - Estimate chunk duration: ~40 minutes per chunk for standard bitrate
   - Calculate number of chunks needed: 45MB / 20MB = 3 chunks
   - Display:
     ```
     File size: 45MB (90 minutes)
     Chunk size: 20MB (~40 minutes)
     Estimated chunks: 3
     ```

5. **System chunks audio file**
   - Use FFmpeg to split file into segments by time:
     - Chunk 1: 0:00 - 40:00
     - Chunk 2: 40:00 - 80:00
     - Chunk 3: 80:00 - 90:00
   - FFmpeg command: `ffmpeg -i large-interview.mp3 -f segment -segment_time 2400 -c copy chunk_%03d.mp3`
   - Display progress: "Chunking file [====================] 100%"
   - Validate each chunk is <25MB

6. **System transcribes chunks sequentially**
   - For each chunk (in order):
     - Display: "Transcribing chunk 1/3 (0:00-40:00)..."
     - Upload chunk to Whisper API (UC-001 logic)
     - Receive transcript with timestamps
     - Save chunk transcript to temp file
     - Save checkpoint: `.transcribe-checkpoint.json` (track progress)
     - Display: "Chunk 1/3 complete ✓"
   - Show overall progress:
     ```
     [============================                ] 66% (2/3 chunks)
     Estimated time remaining: 3 minutes
     ```

7. **System merges chunk transcripts**
   - Load all chunk transcripts in order
   - Adjust timestamps for chunk 2+ (add offset based on chunk start time):
     - Chunk 1: timestamps as-is (0:00 - 40:00)
     - Chunk 2: add 40:00 offset (40:00 - 80:00)
     - Chunk 3: add 80:00 offset (80:00 - 90:00)
   - Concatenate transcript text seamlessly
   - Display: "Merging chunks..."

8. **System saves final transcript**
   - Generate output filename: `large-interview.txt` or `large-interview-transcript.txt`
   - Save merged transcript to output directory
   - Include metadata: total duration, word count, chunk count
   - Display:
     ```
     Success: Transcript saved to ./transcripts/large-interview.txt
     Duration: 90 minutes
     Words: 13,500
     Chunks processed: 3
     ```

9. **System cleans up temporary files**
   - Delete chunk audio files: `chunk_001.mp3`, `chunk_002.mp3`, `chunk_003.mp3`
   - Delete chunk transcript files
   - Delete checkpoint file (successful completion)
   - Display: "Cleaned up 3 temporary chunk files."

10. **System notifies user**
    - Display completion message with file location
    - Show total processing time
    - Display estimated API cost (sum of chunk costs)

**Example Output**:
```
Large file detected (45MB). Processing in 3 chunks...
Chunking file [========================================] 100%
Transcribing chunk 1/3 (0:00-40:00) [=================] 100% ✓
Transcribing chunk 2/3 (40:00-80:00) [=================] 100% ✓
Transcribing chunk 3/3 (80:00-90:00) [=================] 100% ✓
Merging chunks...

Success: Transcript saved to ./transcripts/large-interview.txt
Duration: 90 minutes
Words: 13,500
Chunks processed: 3
Estimated API cost: $0.54

Cleaned up temporary files.
```

---

## Alternative Flows

### Alternative Flow 1: Resume Interrupted Chunked Transcription

**Trigger**: User interrupts processing (Ctrl+C) or chunk transcription fails partway through

1. System catches interrupt or error during chunk N processing
2. Save checkpoint: `.transcribe-checkpoint.json` with:
   ```json
   {
     "file": "large-interview.mp3",
     "total_chunks": 3,
     "completed_chunks": [1],
     "failed_chunks": [],
     "pending_chunks": [2, 3]
   }
   ```
3. Display:
   ```
   Interrupted. Processed 1/3 chunks.
   Resume with: transcribe large-interview.mp3 --resume
   ```
4. User runs: `transcribe large-interview.mp3 --resume`
5. System loads checkpoint, skips completed chunks, resumes from chunk 2
6. Display: "Resuming from chunk 2/3..."
7. Resume at Step 6 (transcribe remaining chunks)

### Alternative Flow 2: Smart Chunking by Silence Detection

**Trigger**: User specifies `--smart-chunk` flag (advanced feature, v2 candidate)

1. Instead of fixed-time chunks, use FFmpeg silence detection to split at natural pauses
2. FFmpeg command: `ffmpeg -i input.mp3 -af silencedetect=noise=-30dB:d=1 -f null -`
3. Parse silence timestamps, split at silence boundaries (avoid mid-sentence splits)
4. Pros: Better transcript quality (no awkward mid-word splits)
5. Cons: More complex, slower chunking, not guaranteed to stay under 25MB
6. Resume at Step 6 (transcribe chunks)

**Notes**: Defer to v2 unless user feedback indicates mid-sentence splits are problematic.

### Alternative Flow 3: Chunk Transcription Fails (API Error)

**Trigger**: Step 6 - API returns error for chunk N transcription

1. System attempts to transcribe chunk N
2. API returns error (e.g., 500 Internal Server Error, rate limit)
3. System retries chunk N up to 3 times with exponential backoff
4. If still failing, save checkpoint and display:
   ```
   Chunk 2/3 failed after 3 retries: [API error message]
   Completed chunks: 1
   Resume with: transcribe large-interview.mp3 --resume
   ```
5. User can troubleshoot (check API status, quota) and resume later

### Alternative Flow 4: Chunk Size Optimization (Very Large Files)

**Trigger**: File is extremely large (e.g., 500MB, 8 hours)

1. System detects file >100MB
2. Calculate optimal chunk size to minimize API requests while staying under 25MB
3. Example: 500MB → 25 chunks of 20MB each
4. Display warning: "Very large file (500MB). This will take approximately 2 hours and cost ~$3.00. Continue? (y/n)"
5. User confirms or cancels
6. If confirmed, resume at Step 5 (chunk file)

**Use Case**: Podcasts, conference recordings, long lectures.

### Alternative Flow 5: Parallel Chunk Transcription (v2 Feature)

**Trigger**: User specifies `--parallel-chunks` flag (future enhancement)

1. Instead of sequential chunk transcription, process chunks in parallel
2. Limit concurrency to respect API rate limits (e.g., 3 parallel chunks)
3. Pros: Faster total processing time (e.g., 90 min file in 30 min vs. 60 min)
4. Cons: Higher API rate limit risk, more complex error handling
5. Display:
   ```
   Transcribing 3 chunks in parallel...
   Chunk 1/3 [======================] 100% ✓
   Chunk 2/3 [================      ] 70%
   Chunk 3/3 [===========           ] 50%
   ```
6. Resume at Step 7 (merge chunks)

**Notes**: Defer to v2; sequential processing is simpler and more reliable for MVP.

---

## Success Criteria

### Functional Criteria
1. Files >25MB are automatically detected and chunked without user intervention
2. Chunks are transcribed successfully with <5% failure rate
3. Merged transcript is seamless (no missing content at chunk boundaries)
4. Timestamps are accurate and continuous across chunks
5. Resume functionality allows recovery from interruptions without re-processing completed chunks

### Performance Criteria
1. Chunking time <5% of total processing time (e.g., <2 min for 90 min file)
2. Total processing time scales linearly with file size (e.g., 90 min file in ~12-15 min)
3. Temporary disk usage does not exceed 2x original file size
4. Memory usage remains constant regardless of file size (streaming/chunking approach)

### Usability Criteria
1. User is notified upfront about chunking (not silent background process)
2. Progress indicators show both chunk-level and overall progress
3. Resume functionality is automatic (detects checkpoint, no manual intervention)
4. Error messages specify which chunk failed and why

### Quality Criteria
1. Transcription accuracy >90% across all chunks (same as single-file baseline)
2. No data loss at chunk boundaries (e.g., sentence fragments, missing words)
3. Timestamp offsets are calculated correctly (no overlapping or gap timestamps)
4. Temporary files cleaned up reliably, even on errors

---

## Non-Functional Requirements

### NFR-016: Seamless Large File Handling
- **Requirement**: Files >25MB are handled transparently; user experience is indistinguishable from single-file processing
- **Validation**: User testing with 45MB, 100MB, 200MB files; confirm no manual intervention needed

### NFR-017: Resume Reliability
- **Requirement**: Interrupted chunked transcriptions can be resumed without data loss or duplicate processing
- **Validation**: Interrupt processing at 50%, resume, verify completion without re-processing

### NFR-018: Timestamp Accuracy
- **Requirement**: Timestamps in merged transcript are accurate to within ±1 second
- **Validation**: Manual spot-check timestamps against original audio at chunk boundaries

### NFR-019: Chunk Boundary Quality
- **Requirement**: Chunk boundaries do not introduce transcription artifacts (e.g., repeated words, missing segments)
- **Validation**: Manual review of transcript text at chunk boundaries, compare to original audio

### NFR-020: Disk Space Efficiency
- **Requirement**: Temporary disk usage does not exceed 2x original file size
- **Validation**: Monitor disk usage during chunking/transcription, verify cleanup

---

## Assumptions

1. User has adequate disk space for temporary chunk files (typically 2x original file size)
2. User has sufficient API quota for multiple chunk requests (e.g., 10-20 chunks for very large files)
3. Network connectivity remains stable during multi-chunk processing (or user can resume)
4. FFmpeg can chunk files accurately by time (timestamp-based splitting is reliable)
5. Whisper API transcription quality is consistent across chunks (no quality degradation for segments)

---

## Dependencies

### Technical Dependencies
- **UC-001 (Transcribe Single Audio File)**: Reuses transcription logic per chunk
- **FFmpeg**: Required for file chunking (time-based or silence-based splitting)
- **Python `subprocess` module**: Execute FFmpeg chunking commands
- **OpenAI Whisper API**: Must support multiple sequential requests (typical for paid tier)
- **Checkpoint/State Persistence**: JSON file for resume functionality

### Logical Dependencies
- **API Key Configuration**: Inherited from UC-001
- **FFmpeg Installation**: Required for chunking (same as UC-002 dependency)

---

## Related Use Cases

- **UC-001**: Transcribe Single Audio File (underlying transcription logic per chunk)
- **UC-002**: Extract and Transcribe Video File (may produce large extracted audio >25MB)
- **UC-003**: Batch Process Directory (batch may include large files requiring chunking)
- **UC-005**: Generate Timestamped Output (chunk timestamps must merge correctly in SRT format)

---

## Notes and Comments

### Design Considerations
- **Chunking Strategy**: Fixed-time chunking (e.g., 40-minute segments) is simpler than silence detection
- **Sequential vs. Parallel**: Sequential processing is more reliable for MVP (simpler error handling)
- **Checkpoint Persistence**: JSON file tracks progress, enables resume after interruption or error
- **Timestamp Offsetting**: Critical for SRT/VTT output; plain text transcripts may not need timestamps

### Implementation Notes
- **FFmpeg Chunking Command**: `ffmpeg -i input.mp3 -f segment -segment_time 2400 -c copy chunk_%03d.mp3`
  - `-segment_time 2400`: 40-minute chunks (2400 seconds)
  - `-c copy`: Stream copy (fast, lossless)
- **Chunk Validation**: Check each chunk size <25MB after splitting; re-split if needed
- **Timestamp Calculation**: Chunk N offset = sum(durations of chunks 1...N-1)
- **Cleanup Logic**: Always clean up temp files, even on errors (use try/finally)

### Future Enhancements (Post-MVP)
- Smart chunking by silence detection (avoid mid-sentence splits)
- Parallel chunk transcription (faster total time, respects rate limits)
- Adaptive chunk sizing (optimize based on API rate limits and file size)
- Chunk overlap (e.g., 5-second overlap to ensure no word loss at boundaries)
- Progress estimation (ETA based on completed chunks and API latency)

### Testing Guidance
- **Unit Tests**: Chunking logic, timestamp offset calculation, checkpoint persistence
- **Integration Tests**: Mock FFmpeg chunking, test merge logic
- **Manual Tests**: 45MB file (3 chunks), 100MB file (5 chunks), 500MB file (25 chunks)
- **Error Tests**: Interrupt during chunk 2, API failure on chunk 3, disk full during chunking
- **Edge Cases**: Exact 25MB file (should not chunk), 25.1MB file (minimal chunking)

### Performance Expectations
- **45MB file (90 min)**: ~12-15 min total (2 min chunk + 10 min transcribe 3 chunks + 1 min merge)
- **100MB file (200 min)**: ~30-35 min total (sequential processing)
- **500MB file (1000 min)**: ~2-3 hours (25 chunks × 5-7 min per chunk)

---

## Traceability

### Requirements Mapping
- **Vision Document Section 6**: In-Scope Features → "Large File Handling"
- **Success Metrics**: Time Savings (automated chunking vs. manual splitting)
- **Personas**: Secondary Persona → Content Creator/Researcher → Use Case 2 (Podcast Processing)
- **Constraints**: Technical Constraints → API File Size Limits (documented mitigation)

### Test Coverage
- **Test Plan Reference**: Master Test Plan → Integration Tests → Large File Chunking Module
- **Acceptance Tests**: Defined in test strategy document (TBD)

---

## Approvals

| Role | Name | Status | Date | Comments |
|------|------|--------|------|----------|
| Requirements Analyst | Claude (Requirements Analyst) | Draft | 2025-12-04 | Initial draft for review |
| Product Owner | TBD | Pending | - | Consider deferring to v1.1 if timeline tight |
| Tech Lead | TBD | Pending | - | Review chunking strategy complexity |

---

**Document End**
