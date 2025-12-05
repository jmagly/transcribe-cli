# ADR-001: FFmpeg Integration Approach

## Status

**Accepted**

## Date

2025-12-04

## Context

The Audio Transcription CLI Tool needs to extract audio from MKV video files and convert audio between various formats (MP3, AAC, FLAC, etc.) before sending to the OpenAI Whisper API for transcription.

The team has strong Python skills but limited FFmpeg/audio processing experience. The project has a 1-3 month MVP timeline with 80% team adoption as the primary success metric.

Three integration approaches were evaluated:

1. **ffmpeg-python library** - Python wrapper providing Pythonic interface to FFmpeg
2. **Direct subprocess calls** - Use Python subprocess module to call FFmpeg binary directly
3. **pydub library** - High-level audio processing library that wraps FFmpeg

## Decision

Use **ffmpeg-python library** as the primary FFmpeg integration approach, with the ability to fall back to direct subprocess calls for edge cases.

## Rationale

### Weighted Scoring Analysis

Based on project priorities (Quality/Reliability: 0.30 each, Delivery Speed: 0.25, Cost: 0.15):

| Option | Delivery | Cost | Quality | Reliability | Weighted Score |
|--------|----------|------|---------|-------------|----------------|
| ffmpeg-python | 4/5 | 5/5 | 4/5 | 4/5 | **4.15** |
| Direct subprocess | 3/5 | 5/5 | 3/5 | 4/5 | 3.60 |
| pydub | 5/5 | 5/5 | 3/5 | 3/5 | 3.80 |

### Key Decision Factors

1. **Pythonic Interface**: Team strength is Python. Library provides familiar patterns (method chaining, Python exceptions) rather than shell command string construction.

2. **Abstracts Command Construction**: Reduces bugs from FFmpeg command syntax errors, argument escaping issues, and platform-specific path handling.

3. **Allows Fallback**: Unlike pydub, ffmpeg-python allows dropping down to direct FFmpeg calls via `.run()` method when the high-level API is insufficient.

4. **MKV Support**: Better than pydub for complex container formats like MKV with multiple audio streams and various codecs.

5. **Error Handling**: Library provides structured Python exceptions rather than parsing FFmpeg stderr output.

### Why Not Direct Subprocess?

- Slower development (manual command construction, error parsing)
- More security risk (command injection if inputs not carefully sanitized)
- More code to maintain (retry logic, platform-specific handling)
- Steeper learning curve for team without FFmpeg experience

### Why Not pydub?

- Limited control for edge cases (unusual codecs, complex operations)
- High-level abstraction may hit ceiling with MKV container complexities
- Risk of needing to switch approaches mid-project if limitations encountered
- Less flexible for the audio stream selection (MKV files often have multiple audio tracks)

## Consequences

### Positive

- Faster development with Pythonic API
- Reduced bug surface from command construction
- Team can learn FFmpeg concepts incrementally through library abstraction
- Flexibility to drop to direct calls for edge cases
- Proven library with production usage

### Negative

- Added dependency introduces library maintenance risk
- Team needs to learn library API in addition to FFmpeg concepts
- Possible FFmpeg version incompatibilities (library may not support newest FFmpeg features)
- May need direct FFmpeg calls for unusual codecs or complex filters

### Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Library limitations for edge cases | Medium | Medium | Use `.run()` method for direct FFmpeg calls when needed |
| FFmpeg version incompatibilities | Low | Medium | Document minimum FFmpeg version (4.0+), pin library version |
| Library maintenance/abandonment | Low | High | Code structure allows migration to direct subprocess if needed |
| Team learning curve | Medium | Low | Allocate Sprint 1-2 for PoC and learning |

## Alternatives Rejected

### Direct Subprocess Calls

**Rejected because**: Slower development, more security risk (command injection), steeper learning curve. Team lacks FFmpeg experience to efficiently construct and debug commands.

**When to reconsider**: If ffmpeg-python library becomes unmaintained or major bugs are discovered. If edge cases dominate and library abstraction becomes limiting.

### pydub Library

**Rejected because**: Limited control for MKV edge cases. Risk of hitting abstraction ceiling with complex container formats. Simpler API is appealing but flexibility concerns outweigh speed benefit.

**When to reconsider**: If project scope narrows to only common audio formats (MP3, WAV, AAC) without MKV extraction requirement.

## Implementation Guidance

### PoC Validation (Sprint 1-2)

```python
# Test extraction from MKV
import ffmpeg

# Extract first audio stream to MP3
(
    ffmpeg
    .input('sample.mkv')
    .output('output.mp3', acodec='libmp3lame', audio_bitrate='192k')
    .run(overwrite_output=True)
)
```

### Fallback Pattern for Edge Cases

```python
import subprocess

def extract_with_direct_call(input_path, output_path, codec_options):
    """Fallback to direct FFmpeg for edge cases."""
    cmd = ['ffmpeg', '-i', input_path, *codec_options, output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error: {result.stderr}")
```

### Minimum Version Requirements

- FFmpeg: 4.0+
- ffmpeg-python: Pin to latest stable in requirements.txt
- Python: 3.9+

## Related Decisions

- ADR-002: Batch Processing Concurrency Model (async architecture)
- ADR-003: Output Format Support (txt/SRT for MVP)

## References

- [ffmpeg-python GitHub](https://github.com/kkroening/ffmpeg-python)
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
- Option Matrix: `.aiwg/intake/option-matrix.md`
