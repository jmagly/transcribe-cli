# Use Case Brief: UC-003 - Batch Process Directory

---

## Use Case Identification

**UC ID**: UC-003
**Title**: Batch Process Directory
**Version**: 1.0
**Priority**: P0 (Critical for MVP)
**Status**: Draft
**Date**: 2025-12-04

---

## Use Case Overview

### Actor(s)
- **Primary Actor**: Engineering Team Member (with backlog of recordings)
- **Supporting Actors**:
  - FFmpeg (local system utility, for video files)
  - OpenAI Whisper API (external system)

### Goal/Purpose
Transcribe all audio and video files in a directory with a single command, enabling efficient processing of conference recordings, interview series, or meeting archives without manual per-file execution.

### Preconditions
1. CLI tool is installed on user's system
2. Python 3.9+ is available
3. FFmpeg is installed and accessible in PATH (if directory contains video files)
4. OpenAI API key is configured (environment variable `OPENAI_API_KEY` or `.env` file)
5. Target directory exists and contains valid media files (audio/video)
6. User has network connectivity to OpenAI API
7. User has sufficient API quota/credits for multiple transcriptions
8. User has sufficient disk space for output transcripts (and temp audio if videos present)

### Postconditions (Success)
1. Transcript files generated for all valid media files in directory
2. Summary report displayed showing success/failure counts
3. Failed files logged with error reasons
4. Output directory contains transcripts with predictable naming
5. User receives completion notification with statistics

### Postconditions (Failure)
1. Partial transcripts saved for successfully processed files
2. Clear error summary displayed for failed files
3. Error log file generated for troubleshooting
4. User can identify which files succeeded vs. failed
5. User can rerun command to retry failed files only

---

## Main Success Scenario

**User Action → System Response Pattern**

1. **User runs batch transcribe command**
   - Command: `transcribe ./recordings/` or `transcribe --dir ./recordings/`
   - User triggers batch processing workflow

2. **System discovers media files**
   - Scan target directory for supported file extensions
   - Supported audio: `.mp3`, `.aac`, `.flac`, `.wav`, `.m4a`
   - Supported video: `.mkv`, `.mp4`, `.avi`, `.mov`
   - Recursively scan subdirectories (optional: `--recursive` flag)
   - Display discovered files: "Found 12 media files (8 audio, 4 video)"

3. **System validates environment**
   - Confirm OpenAI API key is present
   - Check FFmpeg availability (if video files detected)
   - Verify output directory exists or create it
   - Estimate total API cost and display: "Estimated cost: $0.72 for 120 minutes"
   - Prompt user for confirmation (optional: `--yes` flag to skip)

4. **System initializes batch processing**
   - Create job queue with all discovered files
   - Initialize progress tracking (file count, time elapsed, estimated remaining)
   - Set concurrency limit (default: 5 parallel jobs, configurable via `--parallel N`)
   - Display progress overview:
     ```
     Processing 12 files (5 parallel workers)...
     [============                                    ] 25% (3/12)
     ```

5. **System processes files in parallel**
   - For each file in queue:
     - If video: Extract audio (UC-002 logic)
     - Transcribe audio (UC-001 logic)
     - Save transcript to output directory
     - Update progress bar
   - Handle rate limits: Queue jobs if API limit hit, retry with exponential backoff
   - Handle errors: Log error, mark file as failed, continue with next file

6. **System monitors progress**
   - Display real-time progress updates:
     ```
     [========================================] 100% (12/12)
     Completed: 10 | Failed: 2 | Time: 8m 32s
     ```
   - Show individual file status (optional: `--verbose` mode):
     ```
     ✓ meeting-2024-11-15.mkv → meeting-2024-11-15.txt
     ✓ interview-john.mp3 → interview-john.txt
     ✗ corrupted-file.mp3 (ERROR: Invalid audio format)
     ```

7. **System generates summary report**
   - Count successful vs. failed transcriptions
   - Calculate total processing time
   - Estimate total API cost (if available)
   - Display summary:
     ```
     Batch Processing Complete
     -------------------------
     Total files: 12
     Successful: 10
     Failed: 2
     Total duration: 8 minutes 32 seconds
     Estimated API cost: $0.68

     Failed files:
     - corrupted-file.mp3: Invalid audio format
     - large-video.mkv: Rate limit exceeded (retry recommended)

     Transcripts saved to: ./transcripts/
     ```

8. **System saves error log**
   - Create error log file: `./transcripts/batch-errors-2024-12-04.log`
   - Include file paths, error messages, timestamps
   - User can review errors for troubleshooting

9. **System notifies user**
   - Display completion message
   - Provide path to transcripts directory
   - Suggest next steps (review error log, retry failed files)

---

## Alternative Flows

### Alternative Flow 1: No Media Files Found in Directory

**Trigger**: Step 2 - Directory scan finds no supported files

1. System scans directory and subdirectories
2. No files match supported extensions
3. Display message: "No media files found in ./recordings/"
4. Suggest: "Supported formats: MP3, AAC, FLAC, WAV, M4A, MKV, MP4, AVI, MOV"
5. Exit with error code 20 (No files to process)

### Alternative Flow 2: API Rate Limit Hit During Batch

**Trigger**: Step 5 - API returns 429 (Too Many Requests) during batch processing

1. System detects rate limit error for current file
2. Pause processing queue
3. Display: "Rate limit reached. Pausing for 60 seconds..."
4. Implement exponential backoff (60s, 120s, 240s)
5. Resume processing after backoff period
6. If persistent, save state and allow user to resume later:
   ```
   Rate limit exhausted. Processed 5/12 files.
   Resume with: transcribe ./recordings/ --resume
   ```

### Alternative Flow 3: Disk Space Insufficient

**Trigger**: Step 5 - Disk space check fails during processing

1. System monitors available disk space during extraction/transcription
2. If space falls below threshold (e.g., <100MB), pause processing
3. Display error: "Insufficient disk space. Need ~500MB for remaining files."
4. Suggest cleanup or change output directory: `--output-dir /other/path`
5. User can free space and resume with `--resume`

### Alternative Flow 4: User Interrupts Processing (Ctrl+C)

**Trigger**: User presses Ctrl+C during batch processing

1. System catches interrupt signal (SIGINT)
2. Gracefully stop current jobs
3. Clean up temporary files (extracted audio)
4. Save progress state: `.transcribe-state.json`
5. Display:
   ```
   Interrupted. Processed 7/12 files.
   Resume with: transcribe ./recordings/ --resume
   ```
6. User can resume later without re-processing completed files

### Alternative Flow 5: Selective File Processing (Filtering)

**Trigger**: User specifies file pattern filter

**Command**: `transcribe ./recordings/ --pattern "meeting-*.mkv"`

1. System scans directory for all files
2. Apply regex/glob pattern filter
3. Display: "Found 4 files matching pattern 'meeting-*.mkv'"
4. Process only matching files
5. Resume at Step 4 (batch processing)

**Use Case**: User wants to transcribe only specific file types or date ranges.

### Alternative Flow 6: Resume Previous Batch

**Trigger**: User runs with `--resume` flag

**Command**: `transcribe ./recordings/ --resume`

1. System checks for state file: `.transcribe-state.json`
2. Load previous job queue and completed files list
3. Skip already-processed files
4. Display: "Resuming batch. 5 files remaining (7 already completed)."
5. Resume at Step 5 (process remaining files)

**Use Case**: Recover from interruption, rate limit exhaustion, or network failure.

---

## Success Criteria

### Functional Criteria
1. All valid media files in directory are discovered and processed
2. Transcripts generated for successful files with >90% accuracy
3. Failed files clearly reported with actionable error messages
4. Summary report provides complete batch statistics
5. Resume functionality allows recovery from interruptions

### Performance Criteria
1. Parallel processing reduces total time by ~5x (5 parallel jobs vs. sequential)
2. Total batch time scales linearly with number of files (e.g., 12 files in ~15 minutes)
3. Progress updates refresh at least every 5 seconds
4. Memory usage scales with concurrency limit (not total file count)

### Usability Criteria
1. Single command execution: `transcribe ./recordings/`
2. Clear progress indicator showing completion percentage and ETA
3. Summary report highlights successes and failures at a glance
4. Error log provides detailed troubleshooting information
5. Resume feature requires minimal user effort (single flag)

### Quality Criteria
1. No data loss (all completed transcripts saved, even on interruption)
2. Error isolation (one file's failure doesn't stop entire batch)
3. Rate limit handling prevents API quota exhaustion
4. Temporary files cleaned up even on errors

---

## Non-Functional Requirements

### NFR-011: Batch Processing Efficiency
- **Requirement**: Parallel processing reduces total time by >=80% vs. sequential (5 parallel jobs)
- **Validation**: Benchmark 12-file batch sequential vs. parallel, measure time savings

### NFR-012: Progress Transparency
- **Requirement**: User receives progress updates at least every 5 seconds (not silent for long periods)
- **Validation**: User testing confirms clarity and frequency of progress updates

### NFR-013: Error Resilience
- **Requirement**: Single file failure does not halt entire batch; processing continues
- **Validation**: Test with intentionally corrupted files in batch, verify others succeed

### NFR-014: Resumability
- **Requirement**: User can resume interrupted batch without re-processing completed files
- **Validation**: Interrupt batch at 50%, resume, verify no duplicate processing

### NFR-015: API Cost Transparency
- **Requirement**: User sees estimated API cost before batch starts
- **Validation**: Display cost estimate in pre-processing summary, compare to actual cost

---

## Assumptions

1. User has sufficient API quota for batch size (e.g., 10-20 files typical, not 1000s)
2. Network connectivity remains stable during batch (or user can resume on reconnection)
3. Directory contains mix of audio/video files (not all video, which would require more disk space)
4. User's system can handle 5 parallel API requests without performance degradation
5. OpenAI API rate limits allow for batch processing (e.g., 50 requests/minute for paid tier)

---

## Dependencies

### Technical Dependencies
- **UC-001 (Transcribe Single Audio File)**: Core transcription logic reused per file
- **UC-002 (Extract and Transcribe Video File)**: Extraction logic reused for video files
- **Python `concurrent.futures` or `asyncio`**: Parallel processing framework
- **Python `rich` library**: Progress bars and batch status display
- **OpenAI Whisper API**: Must support concurrent requests (typical for paid tier)
- **FFmpeg**: Required if batch includes video files

### Logical Dependencies
- **API Key Configuration**: Inherited from UC-001
- **FFmpeg Installation**: Required if video files detected in batch

---

## Related Use Cases

- **UC-001**: Transcribe Single Audio File (underlying transcription logic per file)
- **UC-002**: Extract and Transcribe Video File (underlying extraction logic per video)
- **UC-004**: Handle Large File (individual files in batch may exceed 25MB)
- **UC-005**: Generate Timestamped Output (batch can output SRT for all files)

---

## Notes and Comments

### Design Considerations
- **Concurrency Strategy**: Default 5 parallel jobs balances speed and rate limits
- **Rate Limit Handling**: Exponential backoff prevents quota exhaustion
- **Progress Tracking**: Multi-level (overall batch + individual file)
- **State Persistence**: Save `.transcribe-state.json` for resume functionality
- **Error Isolation**: Use try/except per file, continue batch on individual failures

### Implementation Notes
- **File Discovery**: Use `pathlib.Path.glob()` or `os.walk()` for recursive scanning
- **Job Queue**: `concurrent.futures.ThreadPoolExecutor` or `asyncio.Queue`
- **Progress Display**: `rich.progress.Progress` with multiple tasks (overall + per-file)
- **State File Format**: JSON with `{"completed": [...], "failed": [...], "pending": [...]}`

### Future Enhancements (Post-MVP)
- Dry-run mode: `--dry-run` to preview files without processing
- Cost limit: `--max-cost 5.00` to stop batch if estimated cost exceeds limit
- Priority queue: Process smaller files first for faster initial results
- Notification on completion: Email/Slack notification for long batches
- Batch statistics: Generate CSV report with per-file duration, word count, cost

### Testing Guidance
- **Unit Tests**: File discovery, job queue management, state persistence
- **Integration Tests**: Mock parallel API calls, test rate limit handling
- **Manual Tests**: 5-file batch, 10-file batch, mixed audio/video batch
- **Error Tests**: Interruption (Ctrl+C), rate limit, disk full, corrupted files
- **Stress Tests**: 50-file batch, 100-file batch (with low concurrency to avoid rate limits)

### Performance Expectations
- **Sequential Processing**: ~2 min per file → 20 min for 10 files
- **Parallel Processing (5 workers)**: ~4 min for 10 files (5x speedup)
- **Rate Limit Impact**: May slow to sequential if API limits hit

---

## Traceability

### Requirements Mapping
- **Vision Document Section 6**: In-Scope Features → "Batch Processing"
- **Success Metrics**: Time Savings (multiplicative for batch), Workflow Simplification
- **Personas**: Primary Persona → Engineering Team Member with multiple recordings
- **Constraints**: Technical Constraints → API Rate Limits (documented mitigation)

### Test Coverage
- **Test Plan Reference**: Master Test Plan → Integration Tests → Batch Processing Module
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
