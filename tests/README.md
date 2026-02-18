# Test Environment Notes

## Batch Executor-Path Tests

`tests/unit/test_batch.py` includes deterministic tests for `process_batch_async`.

To keep CI/local behavior consistent across constrained runners:
- Tests patch `asyncio.to_thread` to execute inline.
- Tests mock `transcribe_file` and `save_formatted_transcript`.
- No network calls or model downloads are performed.

This validates async orchestration and error handling without depending on thread-pool scheduling behavior.
