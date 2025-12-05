"""Output formatting modules for transcribe-cli."""

from .formatters import (
    format_as_srt,
    format_as_txt,
    format_transcript,
    get_output_extension,
    save_formatted_transcript,
)

__all__ = [
    "format_as_txt",
    "format_as_srt",
    "format_transcript",
    "save_formatted_transcript",
    "get_output_extension",
]
