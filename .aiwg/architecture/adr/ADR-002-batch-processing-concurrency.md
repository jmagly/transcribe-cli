# ADR-002: Batch Processing Concurrency Model

## Status

**Accepted**

## Date

2025-12-04

## Context

Batch processing is a core feature for team workflows (transcribe entire folders of meeting recordings). The tool must efficiently process multiple files while respecting OpenAI Whisper API rate limits and providing good user experience with progress tracking.

Key constraints:
- **I/O-bound workload**: API calls are the bottleneck, not CPU computation
- **Rate limits**: OpenAI API has request limits that must be respected
- **Cost awareness**: Concurrent processing affects API usage patterns
- **Team experience**: Moderate async Python experience
- **Progress tracking**: Users need visibility into batch progress

Options considered:
1. **Python asyncio** - Native async/await with configurable concurrency
2. **ThreadPoolExecutor** - Thread-based concurrency from concurrent.futures
3. **Sequential processing** - Process files one at a time
4. **multiprocessing** - Process-based parallelism

## Decision

Use **Python asyncio** with a configurable concurrency limit (default: 5 concurrent requests) for batch processing.

## Rationale

### Workload Analysis

The transcription workflow is I/O-bound:
1. Read audio file from disk (fast)
2. Upload to Whisper API (network I/O)
3. Wait for transcription response (network I/O)
4. Write output file to disk (fast)

Steps 2-3 dominate execution time. Async I/O is ideal for this pattern.

### Why asyncio?

1. **Optimal for I/O-bound workloads**: Cooperative multitasking without thread overhead
2. **Standard library**: No additional dependencies (unlike aiohttp alternatives)
3. **Configurable concurrency**: `asyncio.Semaphore` provides fine-grained control
4. **OpenAI SDK support**: Official SDK (`openai>=1.0`) supports async client
5. **Progress integration**: Works well with rich library for progress bars

### Why Not ThreadPoolExecutor?

- **Less efficient**: Thread context switching overhead for I/O-bound work
- **GIL limitations**: Python GIL can create contention with many threads
- **Simpler but limiting**: Harder to implement fine-grained rate limiting
- **Note**: ThreadPoolExecutor would work and is simpler; asyncio chosen for efficiency and team growth

### Why Not Sequential Processing?

- **Too slow**: A folder of 20 meeting recordings at 5 minutes each would take 100+ minutes sequentially vs. 20-30 minutes with concurrency
- **Poor UX**: Users expect batch processing to be faster than processing files individually

### Why Not multiprocessing?

- **Overkill**: I/O-bound workload doesn't benefit from multiple CPU cores
- **Overhead**: Process creation and IPC overhead not justified
- **Complexity**: Harder to share state (progress tracking, configuration)

## Consequences

### Positive

- Efficient processing of batch files (5x+ speedup vs. sequential)
- Configurable concurrency respects API rate limits
- Standard library (no additional dependencies)
- Team can learn async patterns incrementally
- Good integration with progress tracking (rich library)

### Negative

- Learning curve for team with moderate async experience
- Debugging async code can be challenging (stack traces, race conditions)
- Need proper async exception handling
- Testing async code requires async test fixtures

### Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Team unfamiliarity with async | Medium | Medium | Code examples, pair programming, allocate learning time in Sprint 3-4 |
| Race conditions in progress tracking | Low | Low | Use `asyncio.Lock` for shared state, or thread-safe data structures |
| Async exception handling errors | Medium | Medium | Comprehensive error handling patterns, integration tests |
| Debugging complexity | Medium | Low | Good logging, structured error reporting, async-aware debugger |

## Implementation Guidance

### Core Pattern

```python
import asyncio
from openai import AsyncOpenAI

class BatchProcessor:
    def __init__(self, concurrency_limit: int = 5):
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.client = AsyncOpenAI()

    async def transcribe_file(self, file_path: str) -> str:
        """Transcribe a single file with rate limiting."""
        async with self.semaphore:
            with open(file_path, 'rb') as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return response.text

    async def process_batch(self, files: list[str]) -> list[str]:
        """Process multiple files concurrently."""
        tasks = [self.transcribe_file(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
```

### Configuration

```python
# Default concurrency (conservative for rate limits)
DEFAULT_CONCURRENCY = 5

# User can override via CLI flag or config
# tnf batch --concurrency 10 ./recordings/
```

### Progress Tracking Integration

```python
from rich.progress import Progress, TaskID

async def process_with_progress(files: list[str], progress: Progress, task: TaskID):
    semaphore = asyncio.Semaphore(5)

    async def process_one(file_path: str):
        async with semaphore:
            result = await transcribe_file(file_path)
            progress.advance(task)
            return result

    tasks = [process_one(f) for f in files]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### Error Handling Pattern

```python
async def transcribe_with_retry(file_path: str, max_retries: int = 3) -> str:
    """Transcribe with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await transcribe_file(file_path)
        except RateLimitError:
            wait_time = 2 ** attempt  # 1, 2, 4 seconds
            await asyncio.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
    raise RuntimeError(f"Failed after {max_retries} attempts")
```

### Testing Pattern

```python
import pytest

@pytest.mark.asyncio
async def test_batch_processing():
    """Test batch processing with mock API."""
    processor = BatchProcessor(concurrency_limit=2)

    # Mock the API client
    with patch.object(processor.client.audio.transcriptions, 'create') as mock:
        mock.return_value = Mock(text="Transcribed text")

        results = await processor.process_batch(['file1.mp3', 'file2.mp3'])

        assert len(results) == 2
        assert mock.call_count == 2
```

## Alternatives Rejected

### ThreadPoolExecutor

```python
# Simpler but less efficient for I/O-bound workload
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(transcribe_file, files))
```

**Rejected because**: Less efficient for I/O-bound work, harder to implement fine-grained rate limiting, team should learn async patterns.

**When to reconsider**: If team struggles with async patterns and timeline is at risk, ThreadPoolExecutor is a valid fallback.

### Sequential Processing

**Rejected because**: Too slow for batch use case. Batch processing is a core feature for team adoption.

### multiprocessing

**Rejected because**: I/O-bound workload doesn't benefit from multiple CPU cores. Process overhead and IPC complexity not justified.

## Performance Expectations

| Scenario | Sequential | Concurrent (5) | Speedup |
|----------|------------|----------------|---------|
| 10 files (1 min each) | ~15 min | ~3 min | 5x |
| 20 files (5 min each) | ~120 min | ~25 min | 4.8x |
| 50 files (2 min each) | ~75 min | ~15 min | 5x |

*Note: Actual speedup depends on API response times and file sizes. Diminishing returns above 5-10 concurrent requests due to rate limits.*

## Related Decisions

- ADR-001: FFmpeg Integration Approach (extraction before transcription)
- ADR-003: Output Format Support (output formatting after transcription)

## References

- [Python asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [OpenAI Python SDK Async](https://github.com/openai/openai-python#async-usage)
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- Option Matrix: `.aiwg/intake/option-matrix.md`
