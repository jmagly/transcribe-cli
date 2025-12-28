"""Speaker diarization using pyannote-audio.

Identifies and labels different speakers in audio.
Requires HuggingFace token for pyannote model access.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .transcriber import TranscriptionResult, TranscriptionSegment


class DiarizationError(Exception):
    """Raised when diarization fails."""

    pass


class DiarizationNotAvailable(Exception):
    """Raised when pyannote is not installed or configured."""

    pass


@dataclass
class SpeakerSegment:
    """A segment of audio attributed to a speaker."""

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        """Duration of segment in seconds."""
        return self.end - self.start


@dataclass
class DiarizedSegment:
    """A transcription segment with speaker attribution."""

    id: int
    speaker: str
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        """Duration of segment in seconds."""
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Result of diarization operation."""

    input_path: Path
    speakers: list[str]
    speaker_segments: list[SpeakerSegment]
    total_duration: float

    @property
    def speaker_count(self) -> int:
        """Number of unique speakers."""
        return len(self.speakers)


@dataclass
class DiarizedTranscript:
    """Transcription with speaker labels."""

    input_path: Path
    output_path: Optional[Path]
    segments: list[DiarizedSegment]
    speakers: list[str]
    language: str
    duration: float

    @property
    def full_text(self) -> str:
        """Full transcript with speaker labels."""
        lines = []
        current_speaker = None
        for seg in self.segments:
            if seg.speaker != current_speaker:
                current_speaker = seg.speaker
                lines.append(f"\n[{seg.speaker}]")
            lines.append(seg.text)
        return " ".join(lines).strip()


@dataclass
class DiarizationConfig:
    """Configuration for diarization."""

    hf_token: Optional[str] = None
    device: str = "auto"
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    def get_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


def _load_diarization_pipeline(config: DiarizationConfig):
    """Load pyannote diarization pipeline.

    Args:
        config: Diarization configuration.

    Returns:
        Pyannote Pipeline instance.

    Raises:
        DiarizationNotAvailable: If pyannote not installed or token missing.
    """
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError as e:
        raise DiarizationNotAvailable(
            "pyannote-audio is not installed.\n"
            "Install with: pip install pyannote.audio\n"
            "Or: pip install transcribe-cli[diarization]"
        ) from e

    if not config.hf_token:
        import os
        config.hf_token = os.environ.get("HF_TOKEN")

    if not config.hf_token:
        raise DiarizationNotAvailable(
            "HuggingFace token required for pyannote models.\n\n"
            "1. Create account at https://huggingface.co\n"
            "2. Accept terms at https://hf.co/pyannote/speaker-diarization-3.1\n"
            "3. Create token at https://hf.co/settings/tokens\n"
            "4. Set: export HF_TOKEN=hf_..."
        )

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.hf_token,
        )
        device = config.get_device()
        pipeline.to(torch.device(device))
        return pipeline
    except Exception as e:
        raise DiarizationNotAvailable(f"Failed to load diarization model: {e}") from e


def diarize_audio(
    audio_path: Path,
    config: Optional[DiarizationConfig] = None,
) -> DiarizationResult:
    """Perform speaker diarization on audio file.

    Args:
        audio_path: Path to audio file.
        config: Diarization configuration.

    Returns:
        DiarizationResult with speaker segments.

    Raises:
        DiarizationNotAvailable: If pyannote not available.
        DiarizationError: If diarization fails.
    """
    audio_path = Path(audio_path).resolve()
    config = config or DiarizationConfig()

    if not audio_path.exists():
        raise FileNotFoundError(f"File not found: {audio_path}")

    pipeline = _load_diarization_pipeline(config)

    try:
        # Run diarization
        diarization_kwargs = {}
        if config.min_speakers:
            diarization_kwargs["min_speakers"] = config.min_speakers
        if config.max_speakers:
            diarization_kwargs["max_speakers"] = config.max_speakers

        diarization = pipeline(str(audio_path), **diarization_kwargs)

        # Extract speaker segments
        speaker_segments = []
        speakers_set = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers_set.add(speaker)
            speaker_segments.append(
                SpeakerSegment(
                    speaker=speaker,
                    start=turn.start,
                    end=turn.end,
                )
            )

        # Get total duration
        total_duration = max(seg.end for seg in speaker_segments) if speaker_segments else 0.0

        return DiarizationResult(
            input_path=audio_path,
            speakers=sorted(speakers_set),
            speaker_segments=speaker_segments,
            total_duration=total_duration,
        )

    except Exception as e:
        raise DiarizationError(f"Diarization failed: {e}") from e


def merge_transcription_with_diarization(
    transcription: TranscriptionResult,
    diarization: DiarizationResult,
) -> DiarizedTranscript:
    """Merge transcription segments with speaker diarization.

    Assigns speaker labels to transcription segments based on overlap.

    Args:
        transcription: Transcription result with segments.
        diarization: Diarization result with speaker segments.

    Returns:
        DiarizedTranscript with speaker-labeled segments.
    """
    diarized_segments = []

    for seg in transcription.segments:
        # Find the speaker with most overlap for this segment
        seg_mid = (seg.start + seg.end) / 2
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for spk_seg in diarization.speaker_segments:
            # Calculate overlap
            overlap_start = max(seg.start, spk_seg.start)
            overlap_end = min(seg.end, spk_seg.end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = spk_seg.speaker

        diarized_segments.append(
            DiarizedSegment(
                id=seg.id,
                speaker=best_speaker,
                start=seg.start,
                end=seg.end,
                text=seg.text,
            )
        )

    return DiarizedTranscript(
        input_path=transcription.input_path,
        output_path=transcription.output_path,
        segments=diarized_segments,
        speakers=diarization.speakers,
        language=transcription.language,
        duration=transcription.duration or diarization.total_duration,
    )


def transcribe_and_diarize(
    input_path: Path,
    output_path: Optional[Path] = None,
    language: str = "auto",
    transcriber_config=None,
    diarization_config: Optional[DiarizationConfig] = None,
) -> DiarizedTranscript:
    """Transcribe and diarize audio file in one operation.

    Args:
        input_path: Path to audio or video file.
        output_path: Optional output path.
        language: Language code or "auto".
        transcriber_config: LocalTranscriberConfig for transcription.
        diarization_config: DiarizationConfig for diarization.

    Returns:
        DiarizedTranscript with speaker-labeled segments.
    """
    from .local_transcriber import transcribe_file_local, LocalTranscriberConfig

    input_path = Path(input_path).resolve()
    transcriber_config = transcriber_config or LocalTranscriberConfig()
    diarization_config = diarization_config or DiarizationConfig()

    # Transcribe
    transcription = transcribe_file_local(
        input_path=input_path,
        output_path=output_path,
        language=language,
        config=transcriber_config,
    )

    # Diarize (use the same audio path)
    # For video files, we need to use extracted audio
    audio_path = input_path
    if hasattr(transcription, '_audio_path'):
        audio_path = transcription._audio_path

    diarization = diarize_audio(
        audio_path=input_path,  # pyannote handles video extraction
        config=diarization_config,
    )

    # Merge results
    return merge_transcription_with_diarization(transcription, diarization)
