"""Core processing modules for transcribe-cli."""

from .batch import (
    BatchResult,
    BatchSummary,
    process_batch,
    process_directory,
    scan_directory,
)
from .extractor import (
    AUDIO_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ExtractionError,
    ExtractionResult,
    MediaInfo,
    NoAudioStreamError,
    UnsupportedFormatError,
    extract_audio,
    get_media_info,
    is_audio_file,
    is_supported_file,
    is_video_file,
)
from .ffmpeg import (
    FFmpegInfo,
    FFmpegNotFoundError,
    FFmpegVersionError,
    check_ffmpeg_available,
    validate_ffmpeg,
)
from .transcriber import (
    APIKeyMissingError,
    FileTooLargeError,
    TranscriptionError,
    TranscriptionResult,
    TranscriptionSegment,
    save_transcript,
    transcribe_file,
)
from .local_transcriber import (
    LocalTranscriberConfig,
    LocalTranscriptionError,
    ModelNotFoundError,
    get_available_models,
    get_gpu_info,
    transcribe_file_local,
)
from .diarizer import (
    DiarizationConfig,
    DiarizationError,
    DiarizationNotAvailable,
    DiarizationResult,
    DiarizedSegment,
    DiarizedTranscript,
    SpeakerSegment,
    diarize_audio,
    merge_transcription_with_diarization,
    transcribe_and_diarize,
)
from .postprocess import (
    DiarizedSegmentSimple,
    apply_speaker_mapping,
    build_diarized_json,
    extract_speaker_labels,
    generate_reports,
    generate_speaker_config,
    load_speaker_config,
    load_whisperx_json,
    segments_from_whisperx_json,
    write_diarized_outputs,
)

__all__ = [
    # FFmpeg
    "FFmpegInfo",
    "FFmpegNotFoundError",
    "FFmpegVersionError",
    "validate_ffmpeg",
    "check_ffmpeg_available",
    # Extractor
    "ExtractionError",
    "ExtractionResult",
    "MediaInfo",
    "NoAudioStreamError",
    "UnsupportedFormatError",
    "extract_audio",
    "get_media_info",
    "is_audio_file",
    "is_video_file",
    "is_supported_file",
    # Transcriber (API)
    "APIKeyMissingError",
    "FileTooLargeError",
    "TranscriptionError",
    "TranscriptionResult",
    "TranscriptionSegment",
    "transcribe_file",
    "save_transcript",
    # Local Transcriber
    "LocalTranscriberConfig",
    "LocalTranscriptionError",
    "ModelNotFoundError",
    "get_available_models",
    "get_gpu_info",
    "transcribe_file_local",
    # Diarization
    "DiarizationConfig",
    "DiarizationError",
    "DiarizationNotAvailable",
    "DiarizationResult",
    "DiarizedSegment",
    "DiarizedTranscript",
    "SpeakerSegment",
    "diarize_audio",
    "merge_transcription_with_diarization",
    "transcribe_and_diarize",
    # Postprocess
    "DiarizedSegmentSimple",
    "apply_speaker_mapping",
    "build_diarized_json",
    "extract_speaker_labels",
    "generate_reports",
    "generate_speaker_config",
    "load_speaker_config",
    "load_whisperx_json",
    "segments_from_whisperx_json",
    "write_diarized_outputs",
    # Batch
    "BatchResult",
    "BatchSummary",
    "process_batch",
    "process_directory",
    "scan_directory",
    # Constants
    "VIDEO_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "SUPPORTED_EXTENSIONS",
]
