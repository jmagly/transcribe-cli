# Technical and Architectural Risks

**Project**: Audio Transcription CLI Tool
**Author**: Architecture Designer
**Date**: 2025-12-04
**Status**: Draft v0.1

---

## Risk Summary Matrix

| Risk ID | Category | Risk Title | Likelihood | Impact | Risk Score | Priority |
|---------|----------|------------|------------|--------|------------|----------|
| TECH-001 | Integration | FFmpeg Subprocess Handling | Medium | High | 12 | High |
| TECH-002 | Integration | Whisper API Dependency | Low | High | 8 | Medium |
| TECH-003 | Performance | Large File Memory Consumption | Medium | Medium | 9 | Medium |
| TECH-004 | Scalability | Batch Processing Concurrency | Medium | Medium | 9 | Medium |
| TECH-005 | Compatibility | Audio Codec Support Gaps | Medium | Medium | 9 | Medium |
| TECH-006 | Integration | FFmpeg Installation Complexity | High | Medium | 12 | High |
| TECH-007 | Architecture | File Chunking Synchronization | Medium | High | 12 | High |
| TECH-008 | Security | API Key Exposure | Low | High | 8 | Medium |
| TECH-009 | Performance | Temporary File Storage Exhaustion | Low | Medium | 6 | Low |
| TECH-010 | Compatibility | Cross-Platform Path Handling | Medium | Low | 6 | Low |

**Risk Scoring**: Likelihood (1-5) x Impact (1-5) = Risk Score
- Low: 1-6, Medium: 7-12, High: 13-19, Critical: 20-25

---

## Detailed Risk Analysis

### TECH-001: FFmpeg Subprocess Handling

**Category**: Integration Risk

**Description**: The application relies on FFmpeg for audio extraction from MKV video files and audio format conversion. Using ffmpeg-python library (recommended approach) or direct subprocess calls introduces risks around command construction errors, FFmpeg version incompatibilities across platforms, and unexpected behavior with edge-case file formats.

**Root Causes**:
- FFmpeg command syntax is complex with hundreds of options
- Different FFmpeg versions may have different codec support or option names
- Platform differences (Linux/macOS/Windows) in FFmpeg behavior
- ffmpeg-python library may not abstract all edge cases

**Likelihood**: Medium (3/5)
- Team has limited FFmpeg experience (identified in option-matrix)
- Edge cases with unusual codecs are expected (noted in intake)
- Library abstraction reduces but does not eliminate risk

**Impact**: High (4/5)
- Core functionality blocked if FFmpeg extraction fails
- User trust and adoption directly affected
- Manual workarounds complex for non-technical users

**Risk Score**: 12 (Medium-High)

**Mitigation Approach**:
1. **Use ffmpeg-python library**: Provides Pythonic interface, handles common patterns, tested in production systems
2. **FFmpeg version validation**: Check FFmpeg version at startup, warn if below minimum (4.0+)
3. **Comprehensive format testing**: Create test suite with diverse audio/video samples (MKV with AAC, MP3, FLAC, multi-track audio)
4. **Hybrid fallback**: ffmpeg-python allows dropping to direct FFmpeg calls via `.run()` for edge cases
5. **Structured error handling**: Parse FFmpeg stderr output for user-friendly error messages

**Technical Validation Method**:
```python
# Version check implementation
import subprocess
import re

def validate_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                                capture_output=True, text=True)
        version_match = re.search(r'ffmpeg version (\d+\.\d+)', result.stdout)
        if version_match:
            version = float(version_match.group(1))
            if version < 4.0:
                raise RuntimeError(f"FFmpeg {version} found, requires 4.0+")
        return True
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found in PATH")
```

**Validation Criteria**:
- [ ] FFmpeg version check implemented and tested on Linux, macOS, Windows
- [ ] Test suite includes 10+ audio/video format variations
- [ ] Error messages provide actionable guidance (install instructions, format suggestions)
- [ ] Integration tests verify extraction from MKV with common codecs (AAC, MP3, FLAC)

**Owner**: Development Team
**Due Date**: Sprint 2 (PoC validation)
**Status**: Open

---

### TECH-002: Whisper API Dependency

**Category**: Integration Risk

**Description**: The application's core transcription functionality depends on OpenAI Whisper API. This external dependency introduces risks of API changes (pricing, rate limits, endpoint modifications), service outages, and unexpected behavior changes that could break functionality or increase costs.

**Root Causes**:
- Third-party API is outside team's control
- API versioning and deprecation policies may change
- Rate limits may tighten during high usage periods
- Pricing model changes could affect team budget

**Likelihood**: Low (2/5)
- OpenAI has stable API versioning practices
- Whisper API is established (not beta)
- SDK handles most API complexity

**Impact**: High (4/5)
- Core functionality completely blocked during outages
- Breaking changes require immediate code updates
- Price increases could affect adoption (budget constraint)

**Risk Score**: 8 (Medium)

**Mitigation Approach**:
1. **Abstract API client**: Create `WhisperClient` abstraction layer to isolate API calls
2. **Version pinning**: Pin openai SDK version in requirements.txt (test before upgrading)
3. **Monitor changelog**: Subscribe to OpenAI API changelog and status updates
4. **Graceful degradation**: Detect API unavailability, provide clear status message with retry guidance
5. **Cost tracking**: Log API usage (minutes transcribed) for budget monitoring
6. **Future: Local fallback**: Plan for local Whisper model (whisper.cpp) as offline alternative (post-MVP)

**Technical Validation Method**:
```python
# API client abstraction
from abc import ABC, abstractmethod

class TranscriptionClient(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, language: str = None) -> dict:
        """Transcribe audio file, returns transcript with metadata."""
        pass

class WhisperAPIClient(TranscriptionClient):
    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def transcribe(self, audio_path: str, language: str = None) -> dict:
        try:
            with open(audio_path, "rb") as f:
                result = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=f,
                    language=language,
                    response_format="verbose_json"
                )
            return {"text": result.text, "segments": result.segments}
        except openai.RateLimitError as e:
            raise TranscriptionRateLimitError(str(e))
        except openai.APIStatusError as e:
            raise TranscriptionAPIError(str(e))

# Future: LocalWhisperClient implementing same interface
```

**Validation Criteria**:
- [ ] API client abstraction implemented with interface
- [ ] Retry logic with exponential backoff tested
- [ ] Rate limit errors caught and user-friendly message displayed
- [ ] API usage logged (minutes, cost estimate)
- [ ] Integration test mocks API for offline testing

**Owner**: Development Team
**Due Date**: Sprint 3-4 (API integration)
**Status**: Open

---

### TECH-003: Large File Memory Consumption

**Category**: Performance Risk

**Description**: The application must handle large audio/video files (1GB+, 2+ hour recordings). Loading entire files into memory for processing could exhaust available RAM, causing crashes or degraded performance. This affects both audio extraction (FFmpeg processing) and API submission (chunking for 25MB API limit).

**Root Causes**:
- Naive file loading reads entire file into memory
- Audio extraction may buffer full uncompressed audio
- Chunk preparation for API requires temporary storage
- Multiple concurrent files compound memory usage

**Likelihood**: Medium (3/5)
- 2+ hour recordings are common use case (explicitly noted in intake)
- Batch processing could trigger simultaneous large file handling
- Python memory management less efficient than native code

**Impact**: Medium (3/5)
- Specific use case fails (large files), not all functionality
- User frustration with crashes on important recordings
- Workarounds exist (process smaller files)

**Risk Score**: 9 (Medium)

**Mitigation Approach**:
1. **Stream processing for extraction**: Configure FFmpeg to stream output, not buffer in memory
2. **Chunk-based API submission**: Split audio files into segments before API call (25MB limit)
3. **Memory profiling during development**: Use memory_profiler to track consumption
4. **Configurable chunk size**: Allow users to adjust chunk size for memory-constrained systems
5. **Progress checkpointing**: Save intermediate results for resume capability
6. **Resource monitoring**: Log memory usage during processing, warn if approaching limits

**Technical Validation Method**:
```python
# Memory-efficient chunk processing
import os
import tempfile
from pathlib import Path

MAX_CHUNK_SIZE_BYTES = 24 * 1024 * 1024  # 24MB (under 25MB API limit)

def chunk_audio_file(audio_path: Path, chunk_dir: Path) -> list[Path]:
    """Split audio file into chunks for API submission."""
    file_size = os.path.getsize(audio_path)

    if file_size <= MAX_CHUNK_SIZE_BYTES:
        return [audio_path]  # No chunking needed

    # Use FFmpeg to split by time segments (avoid memory loading)
    # Example: 10-minute segments
    chunk_duration = 600  # seconds
    chunks = []

    # FFmpeg segment command (streams, no full memory load)
    # ffmpeg -i input.mp3 -f segment -segment_time 600 -c copy chunk_%03d.mp3

    return chunks

# Memory profiling decorator
from functools import wraps
import tracemalloc

def profile_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        logger.debug(f"{func.__name__}: Current={current/1024/1024:.1f}MB, Peak={peak/1024/1024:.1f}MB")
        return result
    return wrapper
```

**Validation Criteria**:
- [ ] Process 2GB test file without exceeding 512MB RAM usage
- [ ] Chunk splitting uses streaming (FFmpeg segment), not memory loading
- [ ] Memory profiling integrated in CI for regression detection
- [ ] User documentation includes memory recommendations for large files

**Owner**: Development Team
**Due Date**: Sprint 4 (large file handling)
**Status**: Open

---

### TECH-004: Batch Processing Concurrency

**Category**: Scalability Risk

**Description**: Batch processing feature allows transcribing multiple files (e.g., entire folder of meeting recordings). Excessive concurrent API calls could trigger rate limits, cause cost overruns, or degrade performance. Finding the right concurrency balance is critical for reliability.

**Root Causes**:
- Users may submit hundreds of files in batch mode
- OpenAI API has rate limits (tokens per minute, requests per minute)
- No inherent throttling in asyncio concurrency
- Cost accumulates quickly at scale ($0.006/minute)

**Likelihood**: Medium (3/5)
- Batch processing is core feature (explicitly in scope)
- Team may process backlogs of recordings
- Concurrency optimization requires experimentation

**Impact**: Medium (3/5)
- Batch processing degraded (slower, failed jobs)
- Rate limit errors frustrate users
- Unexpected cost bills affect budget

**Risk Score**: 9 (Medium)

**Mitigation Approach**:
1. **Configurable concurrency limit**: Default to 5 concurrent API calls, allow user override
2. **Rate limiter implementation**: Use token bucket or semaphore for request throttling
3. **Cost tracking and warnings**: Estimate cost before batch start, warn if exceeds threshold
4. **Backoff on rate limits**: Detect rate limit responses, exponentially back off
5. **Progress persistence**: Save batch state for resume after failures
6. **Dry-run mode**: Show batch plan (file count, estimated cost, time) before execution

**Technical Validation Method**:
```python
import asyncio
from asyncio import Semaphore
from dataclasses import dataclass

@dataclass
class BatchConfig:
    max_concurrent: int = 5
    cost_warning_threshold: float = 10.0  # dollars
    retry_max: int = 3
    backoff_base: float = 2.0

class BatchProcessor:
    def __init__(self, transcriber: TranscriptionClient, config: BatchConfig):
        self.transcriber = transcriber
        self.config = config
        self.semaphore = Semaphore(config.max_concurrent)
        self.total_cost = 0.0

    async def process_file(self, file_path: Path) -> dict:
        async with self.semaphore:
            # Retry with exponential backoff
            for attempt in range(self.config.retry_max):
                try:
                    result = await self._transcribe_async(file_path)
                    self._track_cost(result)
                    return result
                except TranscriptionRateLimitError:
                    wait_time = self.config.backoff_base ** attempt
                    await asyncio.sleep(wait_time)
            raise TranscriptionMaxRetriesError(f"Failed after {self.config.retry_max} attempts")

    def estimate_batch_cost(self, files: list[Path]) -> float:
        """Estimate cost before processing."""
        total_duration = sum(get_audio_duration(f) for f in files)
        return total_duration * 0.006 / 60  # $0.006/minute
```

**Validation Criteria**:
- [ ] Batch of 20 files completes without rate limit errors at default concurrency (5)
- [ ] Cost estimate displayed before batch execution
- [ ] User warning if estimated cost exceeds threshold
- [ ] Resume capability after partial batch failure
- [ ] Load test: 100 files batch completes in reasonable time (<30 min for short files)

**Owner**: Development Team
**Due Date**: Sprint 5-6 (batch processing)
**Status**: Open

---

### TECH-005: Audio Codec Support Gaps

**Category**: Compatibility Risk

**Description**: Real-world audio/video files use diverse codecs and container formats. Unusual codecs, corrupted files, or non-standard format variations may fail extraction or transcription silently or with cryptic errors, degrading user experience.

**Root Causes**:
- Audio codec landscape is fragmented (hundreds of codecs exist)
- User files may have unusual sources (screen recorders, niche devices)
- Corrupted files may pass initial validation but fail during processing
- FFmpeg support varies by build configuration

**Likelihood**: Medium (3/5)
- Real-world files have format quirks (noted in intake)
- MKV container can hold many codec types
- Team cannot test all possible formats

**Impact**: Medium (3/5)
- Specific files fail, not all functionality
- User frustration if their important file fails
- Workarounds may exist (convert with other tools)

**Risk Score**: 9 (Medium)

**Mitigation Approach**:
1. **Format detection with ffprobe**: Probe files before processing, identify codec and validate support
2. **Conversion fallback**: Convert unsupported formats to WAV (universal format) before transcription
3. **Clear error messages**: Identify format issues specifically (codec X not supported, try converting)
4. **Supported formats documentation**: List explicitly supported codecs, known limitations
5. **Graceful corruption handling**: Detect corrupted files early, report specific issue
6. **Format test suite**: Build comprehensive test suite with edge-case formats

**Technical Validation Method**:
```python
import subprocess
import json

SUPPORTED_CODECS = {'aac', 'mp3', 'flac', 'opus', 'vorbis', 'pcm_s16le', 'pcm_s24le'}
WHISPER_SUPPORTED_FORMATS = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}

def probe_audio(file_path: Path) -> dict:
    """Probe audio file for format information."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioProbeError(f"Cannot probe file: {result.stderr}")

    probe_data = json.loads(result.stdout)
    audio_streams = [s for s in probe_data.get('streams', []) if s['codec_type'] == 'audio']

    if not audio_streams:
        raise AudioProbeError("No audio stream found in file")

    codec = audio_streams[0].get('codec_name')
    return {
        'codec': codec,
        'sample_rate': audio_streams[0].get('sample_rate'),
        'channels': audio_streams[0].get('channels'),
        'duration': float(probe_data['format'].get('duration', 0)),
        'supported': codec in SUPPORTED_CODECS
    }

def ensure_compatible_format(file_path: Path, output_dir: Path) -> Path:
    """Convert to compatible format if needed."""
    probe = probe_audio(file_path)

    if probe['supported'] and file_path.suffix.lower() in WHISPER_SUPPORTED_FORMATS:
        return file_path  # Already compatible

    # Convert to WAV (universal compatibility)
    output_path = output_dir / f"{file_path.stem}.wav"
    convert_to_wav(file_path, output_path)
    return output_path
```

**Validation Criteria**:
- [ ] Format detection identifies codec, sample rate, channels before processing
- [ ] Unsupported codecs trigger automatic WAV conversion
- [ ] Test suite includes 15+ format variations (MKV/AAC, MKV/FLAC, MP4/AAC, OGG/Vorbis, etc.)
- [ ] Corrupted file detection with user-friendly error message
- [ ] Documentation lists supported formats and conversion behavior

**Owner**: Development Team
**Due Date**: Sprint 2-3 (format handling)
**Status**: Open

---

### TECH-006: FFmpeg Installation Complexity

**Category**: Integration Risk

**Description**: FFmpeg is an external system dependency that must be installed separately from the Python package. Users (especially on Windows) may struggle with installation, PATH configuration, or version management, blocking the tool's core functionality.

**Root Causes**:
- FFmpeg is not a Python package (cannot pip install)
- Installation differs by platform (apt, brew, manual on Windows)
- PATH configuration varies by OS and shell
- Multiple FFmpeg versions/distributions exist (static builds, package managers)

**Likelihood**: High (4/5)
- Explicitly identified as high likelihood in intake
- Windows users particularly affected (no package manager default)
- Team support burden for installation issues

**Impact**: Medium (3/5)
- Blocks core functionality until resolved
- User frustration during setup
- Workaround exists (follow installation guide)

**Risk Score**: 12 (Medium-High)

**Mitigation Approach**:
1. **Startup validation**: Check FFmpeg availability on first command, provide platform-specific installation instructions
2. **Comprehensive installation docs**: README section with Linux (apt/yum), macOS (brew), Windows (chocolatey/manual) instructions
3. **Version check**: Validate minimum FFmpeg version (4.0+), warn if outdated
4. **Helpful error messages**: Direct users to specific installation resources based on detected OS
5. **Consider bundling (future)**: Explore static FFmpeg binaries in package (licensing complexity, large package size)
6. **Docker option (future)**: Provide Docker image with FFmpeg pre-installed for containerized usage

**Technical Validation Method**:
```python
import platform
import shutil
import sys

def check_ffmpeg_installation() -> tuple[bool, str]:
    """Check if FFmpeg is installed and provide guidance if not."""

    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        os_name = platform.system()

        install_instructions = {
            'Linux': """
FFmpeg not found. Install with your package manager:
  Ubuntu/Debian: sudo apt install ffmpeg
  Fedora/RHEL:   sudo dnf install ffmpeg
  Arch:          sudo pacman -S ffmpeg
""",
            'Darwin': """
FFmpeg not found. Install with Homebrew:
  brew install ffmpeg

If Homebrew not installed: https://brew.sh
""",
            'Windows': """
FFmpeg not found. Install options:
  1. Chocolatey: choco install ffmpeg
  2. Winget:     winget install ffmpeg
  3. Manual:     https://www.gyan.dev/ffmpeg/builds/
     - Download "ffmpeg-release-essentials.zip"
     - Extract and add bin folder to PATH

Verify: Open new terminal and run "ffmpeg -version"
"""
        }

        guidance = install_instructions.get(os_name,
            "FFmpeg not found. Visit https://ffmpeg.org/download.html")

        return False, guidance

    return True, f"FFmpeg found at: {ffmpeg_path}"
```

**Validation Criteria**:
- [ ] Startup check runs on first command, blocks with helpful message if FFmpeg missing
- [ ] Installation docs tested on fresh Linux, macOS, Windows systems
- [ ] Error message includes platform-specific instructions
- [ ] FFmpeg version check warns if below 4.0
- [ ] User feedback collection on installation friction (post-MVP)

**Owner**: Development Team
**Due Date**: Sprint 1 (initial setup)
**Status**: Open

---

### TECH-007: File Chunking Synchronization

**Category**: Architecture Risk

**Description**: Large audio files must be split into chunks (<25MB) for API submission, then transcription results must be reassembled in correct order with accurate timestamps. Chunk boundary handling (mid-word splits, timestamp alignment) is complex and errors could produce garbled or discontinuous transcripts.

**Root Causes**:
- Audio splitting at arbitrary byte boundaries may cut words
- Timestamp synchronization across chunks requires careful offset calculation
- Concurrent chunk processing may return results out of order
- Edge cases: silence at boundaries, overlapping speech

**Likelihood**: Medium (3/5)
- Large file handling is explicit requirement
- Chunking logic is non-trivial
- Edge cases in audio content are unpredictable

**Impact**: High (4/5)
- Garbled transcripts unusable for user's purpose
- Timestamp misalignment breaks SRT/VTT output
- Rework required if discovered late

**Risk Score**: 12 (Medium-High)

**Mitigation Approach**:
1. **Time-based splitting**: Split by duration (10 minutes) not bytes, ensures natural boundaries
2. **Overlap with deduplication**: Add small overlap (5 seconds) at chunk boundaries, deduplicate in merge
3. **Ordered reassembly**: Use chunk index for ordering, not async completion order
4. **Timestamp offset calculation**: Track cumulative offset, apply to each chunk's timestamps
5. **Validation pass**: After merge, validate transcript continuity and timestamp progression
6. **Test with long files**: Specific test suite for 1+ hour files with various content types

**Technical Validation Method**:
```python
from dataclasses import dataclass
from typing import List

@dataclass
class ChunkResult:
    index: int
    start_time: float  # seconds from original file start
    duration: float
    text: str
    segments: list  # [{start, end, text}, ...]

class TranscriptMerger:
    OVERLAP_SECONDS = 5.0

    def merge_chunks(self, chunks: List[ChunkResult]) -> dict:
        """Merge chunked transcription results into single transcript."""

        # Sort by index to ensure order
        sorted_chunks = sorted(chunks, key=lambda c: c.index)

        merged_text = []
        merged_segments = []

        for i, chunk in enumerate(sorted_chunks):
            # Calculate absolute timestamps
            chunk_offset = chunk.start_time

            # Process segments with offset
            for seg in chunk.segments:
                absolute_start = seg['start'] + chunk_offset
                absolute_end = seg['end'] + chunk_offset

                # Skip if in overlap region of previous chunk
                if merged_segments and absolute_start < merged_segments[-1]['end']:
                    continue

                merged_segments.append({
                    'start': absolute_start,
                    'end': absolute_end,
                    'text': seg['text']
                })

            merged_text.append(chunk.text)

        return {
            'text': ' '.join(merged_text),
            'segments': merged_segments,
            'chunk_count': len(chunks)
        }

    def validate_transcript(self, transcript: dict) -> list[str]:
        """Validate merged transcript for issues."""
        issues = []
        segments = transcript['segments']

        for i in range(1, len(segments)):
            prev_end = segments[i-1]['end']
            curr_start = segments[i]['start']

            # Check for gaps > 5 seconds
            if curr_start - prev_end > 5.0:
                issues.append(f"Large gap at {prev_end:.1f}s - {curr_start:.1f}s")

            # Check for overlaps
            if curr_start < prev_end:
                issues.append(f"Overlap at {curr_start:.1f}s (prev ended {prev_end:.1f}s)")

        return issues
```

**Validation Criteria**:
- [ ] 2-hour test file produces continuous transcript with correct timestamps
- [ ] SRT output from chunked file plays correctly with video
- [ ] No duplicate text at chunk boundaries (overlap deduplication works)
- [ ] Timestamp gaps logged, maximum gap < 1 second for continuous speech
- [ ] Integration test compares chunked vs. single-file transcription for short file

**Owner**: Development Team
**Due Date**: Sprint 4-5 (chunking implementation)
**Status**: Open

---

### TECH-008: API Key Exposure

**Category**: Security Risk

**Description**: OpenAI API key is required for Whisper API access. Improper key management (hardcoding, logging, insecure storage) could expose the key, leading to unauthorized API usage, cost overruns, or key revocation.

**Root Causes**:
- Developers may hardcode keys during development
- Keys could appear in logs, error messages, or stack traces
- Insecure config file permissions expose keys
- Keys in version control (accidentally committed)

**Likelihood**: Low (2/5)
- Identified as concern, team aware of risks
- Standard secure practices well-documented
- Internal tool limits exposure scope

**Impact**: High (4/5)
- Unauthorized API usage charges to team account
- Key may need emergency revocation
- Trust and security posture damaged

**Risk Score**: 8 (Medium)

**Mitigation Approach**:
1. **Environment variable default**: Primary method is OPENAI_API_KEY environment variable
2. **Config file with permissions**: If file-based, require 0600 permissions, validate on load
3. **Never log keys**: Sanitize all logs and error messages, mask key values
4. **Gitignore enforcement**: Include .env, *.key, config files in .gitignore
5. **Pre-commit hook**: Scan for potential key patterns before commit
6. **Documentation**: Clear instructions on secure key management in README
7. **Key rotation guidance**: Document how to rotate keys if exposure suspected

**Technical Validation Method**:
```python
import os
import re
import stat
from pathlib import Path

class ConfigSecurityError(Exception):
    pass

def load_api_key(config_path: Path = None) -> str:
    """Load API key securely from environment or config file."""

    # Prefer environment variable
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        return api_key

    # Fallback to config file
    if config_path and config_path.exists():
        # Check file permissions (Unix only)
        if os.name == 'posix':
            mode = config_path.stat().st_mode
            if mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise ConfigSecurityError(
                    f"Config file {config_path} has insecure permissions. "
                    f"Run: chmod 600 {config_path}"
                )

        # Load and validate
        with open(config_path) as f:
            config = yaml.safe_load(f)
            api_key = config.get('api_key')

    if not api_key:
        raise ConfigSecurityError(
            "API key not found. Set OPENAI_API_KEY environment variable or "
            "create config file with api_key field."
        )

    # Validate key format (basic pattern check)
    if not re.match(r'^sk-[a-zA-Z0-9]{40,}$', api_key):
        raise ConfigSecurityError("Invalid API key format")

    return api_key

def mask_api_key(key: str) -> str:
    """Mask API key for logging."""
    if len(key) > 8:
        return f"{key[:4]}...{key[-4:]}"
    return "****"
```

**Validation Criteria**:
- [ ] API key loaded from environment variable (primary method)
- [ ] Config file with insecure permissions rejected with clear error
- [ ] All log output verified to not contain API key
- [ ] .gitignore includes .env, *.key, config.yaml
- [ ] README documents secure key management practices
- [ ] Security scan (grep for sk-) in CI pipeline

**Owner**: Development Team
**Due Date**: Sprint 2 (configuration setup)
**Status**: Open

---

### TECH-009: Temporary File Storage Exhaustion

**Category**: Performance Risk

**Description**: Audio extraction, format conversion, and chunking create temporary files. Processing multiple large files or incomplete cleanup could exhaust disk space, particularly on systems with limited temp storage or SSDs with constrained capacity.

**Root Causes**:
- Extracted audio may be larger than source (uncompressed)
- Chunking creates multiple copies of audio data
- Failed processing may leave orphan temp files
- Concurrent batch processing compounds storage usage

**Likelihood**: Low (2/5)
- Modern systems typically have adequate temp storage
- Proper cleanup should prevent accumulation
- Issue more likely with batch processing of large files

**Impact**: Medium (3/5)
- System instability if disk full
- Failed processing for current and other applications
- Data loss if running out of space during writes

**Risk Score**: 6 (Low)

**Mitigation Approach**:
1. **Temp directory management**: Use tempfile module with explicit cleanup
2. **Context managers**: Ensure cleanup in finally blocks / with statements
3. **Disk space check**: Before large operations, check available space
4. **Streaming where possible**: Pipe FFmpeg output directly when feasible
5. **Configurable temp directory**: Allow users to specify temp location
6. **Cleanup on startup**: Option to clean orphaned temp files from previous runs

**Technical Validation Method**:
```python
import tempfile
import shutil
from contextlib import contextmanager

@contextmanager
def temp_audio_workspace(prefix: str = "transcribe_"):
    """Create temporary workspace with guaranteed cleanup."""
    workspace = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

def check_disk_space(path: Path, required_bytes: int) -> bool:
    """Check if sufficient disk space available."""
    usage = shutil.disk_usage(path)
    available = usage.free

    # Require 2x for safety margin
    if available < required_bytes * 2:
        raise DiskSpaceError(
            f"Insufficient disk space. Need {required_bytes * 2 / 1024 / 1024:.0f}MB, "
            f"have {available / 1024 / 1024:.0f}MB available in {path}"
        )
    return True
```

**Validation Criteria**:
- [ ] All temp files created with context managers for cleanup
- [ ] Disk space checked before large file processing
- [ ] Interrupted processing does not leave orphan files (test with SIGINT)
- [ ] Batch processing monitors cumulative temp usage
- [ ] Configurable temp directory documented

**Owner**: Development Team
**Due Date**: Sprint 3 (core processing)
**Status**: Open

---

### TECH-010: Cross-Platform Path Handling

**Category**: Compatibility Risk

**Description**: File path handling differs between operating systems (forward slash vs. backslash, case sensitivity, special characters, Unicode paths). Improper path handling could cause file-not-found errors or unexpected behavior on different platforms.

**Root Causes**:
- Windows uses backslash, Unix uses forward slash
- Windows paths may have drive letters (C:\)
- Windows is case-insensitive, Unix is case-sensitive
- Unicode characters in paths require proper encoding
- Spaces in paths require quoting for subprocesses

**Likelihood**: Medium (3/5)
- Team uses multiple platforms (Linux, macOS, Windows noted)
- User files may have varied naming conventions
- Subprocess calls to FFmpeg require proper escaping

**Impact**: Low (2/5)
- Specific files or platforms affected
- Workarounds exist (rename files, use simple paths)
- Well-understood problem with standard solutions

**Risk Score**: 6 (Low)

**Mitigation Approach**:
1. **Use pathlib consistently**: Path objects handle cross-platform differences
2. **Subprocess with lists**: Pass arguments as list, not shell string
3. **Test on all platforms**: CI matrix includes Linux, macOS, Windows
4. **Unicode path testing**: Include test files with non-ASCII names
5. **Space handling**: Verify paths with spaces work correctly

**Technical Validation Method**:
```python
from pathlib import Path
import subprocess

def safe_ffmpeg_call(input_path: Path, output_path: Path, args: list) -> subprocess.CompletedProcess:
    """Execute FFmpeg with proper path handling."""

    # Use Path objects, convert to strings for subprocess
    cmd = [
        'ffmpeg',
        '-i', str(input_path.resolve()),  # Absolute path
        *args,
        str(output_path.resolve())
    ]

    # Pass as list (not shell=True) for proper escaping
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
    )

    return result

# Test paths
TEST_PATHS = [
    "simple.mp3",
    "path with spaces/file.mp3",
    "unicode-\u00e9\u00e8\u00ea/audio.mp3",  # French accents
    "very/deep/nested/directory/structure/file.mp3",
]
```

**Validation Criteria**:
- [ ] All file operations use pathlib.Path
- [ ] CI tests run on Linux, macOS, Windows
- [ ] Test files with spaces in path process correctly
- [ ] Test files with Unicode names process correctly
- [ ] FFmpeg subprocess calls use list arguments (not shell string)

**Owner**: Development Team
**Due Date**: Sprint 1-2 (foundation)
**Status**: Open

---

## Risk Monitoring and Review

### Risk Review Schedule

| Review Type | Frequency | Participants | Focus |
|-------------|-----------|--------------|-------|
| Sprint Risk Review | Every 2 weeks | Dev Team | Active risks, new discoveries |
| Phase Gate Review | Phase transitions | Team + Stakeholders | Risk retirement, escalations |
| Post-Incident Review | As needed | Dev Team | Lessons learned, new risks |

### Risk Status Tracking

| Status | Definition |
|--------|------------|
| Open | Risk identified, mitigation planned |
| In Progress | Mitigation actively being implemented |
| Mitigated | Controls in place, residual risk acceptable |
| Retired | Risk no longer applies |
| Realized | Risk occurred, managing impact |

### Escalation Criteria

Escalate to stakeholders when:
- Risk score increases to Critical (20+)
- New High-priority risk identified
- Mitigation blocked or ineffective
- Risk realized with significant impact

---

## Summary

This document identifies 10 technical and architectural risks for the Audio Transcription CLI Tool project. Key findings:

**High Priority Risks (Score 12)**:
1. **TECH-001**: FFmpeg Subprocess Handling - Core dependency with version/platform variability
2. **TECH-006**: FFmpeg Installation Complexity - User onboarding friction, especially Windows
3. **TECH-007**: File Chunking Synchronization - Complex logic for large file handling

**Medium Priority Risks (Score 8-9)**:
4. **TECH-002**: Whisper API Dependency - External service reliability
5. **TECH-003**: Large File Memory Consumption - Performance for 1GB+ files
6. **TECH-004**: Batch Concurrency - Rate limits and cost management
7. **TECH-005**: Audio Codec Support - Format compatibility edge cases
8. **TECH-008**: API Key Exposure - Security for credential management

**Low Priority Risks (Score 6)**:
9. **TECH-009**: Temp File Storage - Disk space management
10. **TECH-010**: Cross-Platform Paths - OS compatibility

**Recommended Actions**:
1. Sprint 1-2 PoC should specifically validate FFmpeg integration (TECH-001, TECH-006)
2. Allocate additional time for chunking implementation and testing (TECH-007)
3. Implement cost tracking early for batch processing visibility (TECH-004)
4. Create comprehensive format test suite with edge cases (TECH-005)
5. Establish security practices from day one (TECH-008)

---

*Document will be reviewed and updated during Sprint Reviews and Phase Gate checks.*
