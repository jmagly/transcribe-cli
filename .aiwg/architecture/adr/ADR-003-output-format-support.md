# ADR-003: Output Format Support

## Status

**Accepted**

## Date

2025-12-04

## Context

Users need transcripts in various formats depending on their use case:
- **Plain text (txt)**: Simple transcription for documentation, notes, searchable archives
- **SRT (SubRip Subtitle)**: Timestamped subtitles for video content
- **VTT (WebVTT)**: Web video captions for HTML5 video players
- **JSON**: Structured data for programmatic processing, integrations

The MVP timeline is 1-3 months with 80% team adoption as the primary success metric. Need to balance feature scope with timeline risk.

Key considerations:
- Whisper API returns timestamps in response (enables subtitle formats)
- Team use cases: meeting transcripts (txt), interview recordings (txt/SRT), lecture videos (SRT for subtitles)
- VTT is structurally similar to SRT (low incremental effort)
- JSON is power-user feature (programmatic access, less common)

## Decision

Support **txt and SRT** formats in MVP. Defer VTT and JSON to v2.

## Rationale

### Use Case Analysis

| Format | Use Cases | Team Priority | MVP Value |
|--------|-----------|---------------|-----------|
| txt | Meeting notes, searchable archives, documentation | **High** | 80% of use cases |
| SRT | Video subtitles, interview recordings, lectures | **High** | Enables subtitle use case |
| VTT | Web video captions | Medium | Similar to SRT, defer |
| JSON | Programmatic processing, integrations | Low | Power-user feature |

### Why txt and SRT for MVP?

1. **txt covers 80% of use cases**: Simple transcription for meetings, notes, documentation. Most users just need readable text.

2. **SRT enables subtitle use case**: Team has interview recordings and lectures that benefit from timestamped subtitles. SRT is widely supported by video players.

3. **Whisper API provides timestamps**: The API response includes word-level timestamps, making SRT generation straightforward with no additional API cost.

4. **Minimal incremental effort**: SRT formatter is ~50-100 lines of Python. Low cost to include in MVP.

### Why Defer VTT?

- **Similar to SRT**: VTT format is structurally similar (different header, slightly different timestamp format)
- **Low incremental effort for v2**: Once SRT is implemented, VTT is ~30 minutes of work
- **Less urgent**: Web video captions are less common use case for internal team
- **Scope control**: Including VTT doesn't add user value proportional to testing/documentation effort

### Why Defer JSON?

- **Power-user feature**: Most users don't need programmatic access to transcription data
- **Scope creep risk**: JSON format invites feature requests (custom fields, schema changes)
- **Integration complexity**: JSON consumers have expectations about schema stability
- **v2 validation**: See if users actually request JSON before implementing

## Consequences

### Positive

- Reduced MVP scope (timeline benefit)
- Focus on most valuable formats (txt, SRT)
- Output formatter design can be extensible for v2 formats
- Simpler testing matrix (2 formats vs. 4)
- Clearer documentation (fewer options to explain)

### Negative

- Users wanting VTT/JSON must wait for v2
- Some web video workflows require VTT specifically (manual SRT->VTT conversion as workaround)
- Users wanting programmatic access must parse txt/SRT themselves

### Design for Extensibility

The output formatter should be designed for easy addition of new formats:

```python
from abc import ABC, abstractmethod

class OutputFormatter(ABC):
    @abstractmethod
    def format(self, transcription: TranscriptionResult) -> str:
        pass

    @abstractmethod
    def file_extension(self) -> str:
        pass

class TxtFormatter(OutputFormatter):
    def format(self, transcription: TranscriptionResult) -> str:
        return transcription.text

    def file_extension(self) -> str:
        return ".txt"

class SrtFormatter(OutputFormatter):
    def format(self, transcription: TranscriptionResult) -> str:
        # Format segments with timestamps
        ...

    def file_extension(self) -> str:
        return ".srt"

# v2: Add VttFormatter, JsonFormatter
```

## Implementation Guidance

### txt Format

```
This is the transcribed text from the audio file.
Multiple sentences appear on consecutive lines.
No timestamps or special formatting.
```

### SRT Format

```srt
1
00:00:00,000 --> 00:00:04,500
Hello and welcome to the meeting.

2
00:00:04,500 --> 00:00:08,200
Today we'll discuss the project roadmap.

3
00:00:08,200 --> 00:00:12,800
Let's start with a quick status update.
```

### CLI Interface

```bash
# Default: txt output
tnf transcribe audio.mp3
# Output: audio.txt

# Specify format
tnf transcribe audio.mp3 --format srt
# Output: audio.srt

# Batch with format
tnf batch ./recordings/ --format srt
# Output: ./recordings/*.srt
```

### Formatter Implementation

```python
def format_srt(segments: list[Segment]) -> str:
    """Format transcription segments as SRT subtitles."""
    lines = []
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(segment.text.strip())
        lines.append("")  # Blank line between entries
    return "\n".join(lines)

def format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
```

## v2 Format Roadmap

### VTT (WebVTT)

```vtt
WEBVTT

00:00:00.000 --> 00:00:04.500
Hello and welcome to the meeting.

00:00:04.500 --> 00:00:08.200
Today we'll discuss the project roadmap.
```

**Effort**: ~30 minutes (header change, timestamp format change from SRT)

### JSON

```json
{
  "text": "Full transcription text...",
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 4.5,
      "text": "Hello and welcome to the meeting."
    }
  ],
  "metadata": {
    "duration": 3600.0,
    "language": "en",
    "model": "whisper-1"
  }
}
```

**Effort**: ~2 hours (schema design, metadata inclusion, documentation)

### v2 Trigger Criteria

Add VTT/JSON when:
1. Users explicitly request (GitHub issues, feedback)
2. Integration use case emerges (web player, API consumer)
3. MVP is stable (80% adoption achieved)
4. Team has bandwidth (post-MVP polish phase)

## Alternatives Rejected

### All Formats in MVP

**Rejected because**: Scope creep, timeline risk. Four formats require 4x testing, documentation, edge case handling. Delays MVP without proportional user value.

### txt Only

**Rejected because**: Misses important SRT use case. Team has video content (interviews, lectures) that benefits from timestamped subtitles. SRT is low incremental effort.

### SRT Only

**Rejected because**: txt is simpler default for most users. Many use cases (meeting notes, documentation) don't need timestamps. Forcing SRT adds friction.

## Related Decisions

- ADR-001: FFmpeg Integration Approach (extraction provides input for transcription)
- ADR-002: Batch Processing Concurrency Model (batch output in specified format)

## References

- [SRT Subtitle Format](https://en.wikipedia.org/wiki/SubRip#SubRip_file_format)
- [WebVTT Standard](https://www.w3.org/TR/webvtt1/)
- [Whisper API Response Format](https://platform.openai.com/docs/guides/speech-to-text)
- Option Matrix: `.aiwg/intake/option-matrix.md`
