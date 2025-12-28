"""Local transcription using faster-whisper.

Implements local GPU inference as alternative to OpenAI API.
Supports NVIDIA GPUs with CUDA for accelerated transcription.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from .extractor import extract_audio, is_video_file
from .transcriber import TranscriptionResult, TranscriptionSegment


class LocalTranscriptionError(Exception):
    """Raised when local transcription fails."""

    pass


class ModelNotFoundError(Exception):
    """Raised when whisper model cannot be loaded."""

    pass


@dataclass
class LocalTranscriberConfig:
    """Configuration for local transcription."""

    model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = "medium"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    compute_type: Literal["int8", "float16", "float32", "auto"] = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    vad_threshold: float = 0.5

    def get_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def get_compute_type(self) -> str:
        """Resolve 'auto' compute type based on device."""
        if self.compute_type != "auto":
            return self.compute_type
        device = self.get_device()
        if device == "cuda":
            # int8 is fastest on older GPUs (Pascal), float16 on newer (Ampere+)
            return "int8"
        return "int8"  # CPU also benefits from int8


def _load_model(config: LocalTranscriberConfig):
    """Load faster-whisper model.

    Args:
        config: Transcriber configuration.

    Returns:
        WhisperModel instance.

    Raises:
        ModelNotFoundError: If model cannot be loaded.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ModelNotFoundError(
            "faster-whisper is not installed.\n"
            "Install with: pip install faster-whisper\n"
            "Or: pip install transcribe-cli[local]"
        ) from e

    device = config.get_device()
    compute_type = config.get_compute_type()

    try:
        model = WhisperModel(
            config.model_size,
            device=device,
            compute_type=compute_type,
        )
        return model
    except Exception as e:
        raise ModelNotFoundError(f"Failed to load model: {e}") from e


def transcribe_file_local(
    input_path: Path,
    output_path: Optional[Path] = None,
    language: str = "auto",
    config: Optional[LocalTranscriberConfig] = None,
) -> TranscriptionResult:
    """Transcribe an audio or video file using local inference.

    For video files, audio is automatically extracted first.

    Args:
        input_path: Path to audio or video file.
        output_path: Optional path for output text file.
        language: Language code or "auto" for detection.
        config: Local transcriber configuration.

    Returns:
        TranscriptionResult with transcribed text and metadata.

    Raises:
        ModelNotFoundError: If faster-whisper not installed.
        LocalTranscriptionError: If transcription fails.
    """
    input_path = Path(input_path).resolve()
    config = config or LocalTranscriberConfig()

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    # Load model
    model = _load_model(config)

    # Handle video files - extract audio first
    audio_path = input_path
    temp_audio = None

    try:
        if is_video_file(input_path):
            # Extract audio to temporary file
            temp_dir = tempfile.mkdtemp(prefix="transcribe_")
            temp_audio = Path(temp_dir) / f"{input_path.stem}.mp3"
            extraction_result = extract_audio(
                input_path=input_path,
                output_path=temp_audio,
                output_format="mp3",
            )
            audio_path = extraction_result.output_path

        # Transcribe with faster-whisper
        try:
            transcribe_kwargs = {
                "beam_size": config.beam_size,
                "vad_filter": config.vad_filter,
                "vad_parameters": {"threshold": config.vad_threshold},
            }
            if language and language != "auto":
                transcribe_kwargs["language"] = language

            segments_gen, info = model.transcribe(
                str(audio_path),
                **transcribe_kwargs,
            )

            # Collect segments
            segments = []
            text_parts = []
            for i, seg in enumerate(segments_gen):
                segments.append(
                    TranscriptionSegment(
                        id=i,
                        start=seg.start,
                        end=seg.end,
                        text=seg.text.strip(),
                    )
                )
                text_parts.append(seg.text)

            full_text = " ".join(text_parts).strip()
            detected_language = info.language or "unknown"
            duration = info.duration

        except Exception as e:
            raise LocalTranscriptionError(f"Transcription failed: {e}") from e

        # Determine output path
        if output_path is None:
            output_path = input_path.with_suffix(".txt")

        return TranscriptionResult(
            input_path=input_path,
            output_path=output_path,
            text=full_text,
            segments=segments,
            language=detected_language,
            duration=duration,
        )

    finally:
        # Clean up temporary audio file
        if temp_audio and temp_audio.exists():
            temp_audio.unlink()
            temp_audio.parent.rmdir()


def get_available_models() -> list[str]:
    """Get list of available model sizes.

    Returns:
        List of model size names.
    """
    return ["tiny", "base", "small", "medium", "large-v3"]


def get_gpu_info() -> dict:
    """Get GPU information for diagnostics.

    Returns:
        Dictionary with GPU info or error message.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory // (1024**2),
            }
        return {"available": False, "reason": "CUDA not available"}
    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}
    except Exception as e:
        return {"available": False, "reason": str(e)}
