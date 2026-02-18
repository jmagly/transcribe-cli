#!/bin/sh
set -eu

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <media-file> [whisperx args...]"
  exit 1
fi

if command -v whisperx >/dev/null 2>&1; then
  exec whisperx "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 -m whisperx "$@"
fi

echo "WhisperX is not installed. Install with: pip install whisperx"
exit 1
