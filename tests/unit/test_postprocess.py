from pathlib import Path

from transcribe_cli.core.postprocess import (
    apply_speaker_mapping,
    build_diarized_json,
    extract_speaker_labels,
    generate_reports,
    generate_speaker_config,
    load_speaker_config,
    segments_from_whisperx_json,
)


def _sample_whisperx_json() -> dict:
    return {
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello there", "speaker": "SPEAKER_01"},
            {"start": 1.0, "end": 2.0, "text": "the network time", "speaker": "SPEAKER_02"},
            {"start": 2.0, "end": 3.0, "text": "we need to test", "speaker": "SPEAKER_01"},
        ],
        "word_segments": [],
        "language": "en",
    }


def test_generate_and_apply_speaker_config(tmp_path: Path) -> None:
    data = _sample_whisperx_json()
    labels = extract_speaker_labels(data)
    assert labels == ["SPEAKER_01", "SPEAKER_02"]

    config_path = tmp_path / "speakers.json"
    generate_speaker_config(config_path, tmp_path / "source.json", labels)

    mapping = load_speaker_config(config_path)
    assert mapping["SPEAKER_01"].startswith("Speaker")

    updated = apply_speaker_mapping(data, mapping)
    assert updated["segments"][0]["speaker"].startswith("Speaker")
    assert updated["segments"][0]["speaker_raw"] == "SPEAKER_01"


def test_speaker_mapping_can_be_reapplied() -> None:
    data = _sample_whisperx_json()
    first = apply_speaker_mapping(data, {"SPEAKER_01": "Speaker 1"})
    assert extract_speaker_labels(first) == ["SPEAKER_01", "SPEAKER_02"]

    second = apply_speaker_mapping(first, {"SPEAKER_01": "Sam", "SPEAKER_02": "Joe"})
    assert [seg["speaker"] for seg in second["segments"]] == ["Sam", "Joe", "Sam"]
    assert second["segments"][0]["speaker_raw"] == "SPEAKER_01"


def test_build_diarized_json_roundtrip() -> None:
    data = _sample_whisperx_json()
    segments = segments_from_whisperx_json(data)
    rebuilt = build_diarized_json(segments, language="en")
    assert rebuilt["language"] == "en"
    assert len(rebuilt["segments"]) == len(data["segments"])


def test_generate_reports_filters_stopwords(tmp_path: Path) -> None:
    data = _sample_whisperx_json()
    segments = segments_from_whisperx_json(data)
    reports_dir = tmp_path / "reports"
    reports = generate_reports(segments, reports_dir, tmp_path / "source.json", top_n=5)

    analysis = reports["analysis"].read_text(encoding="utf-8")
    assert "the:" not in analysis
    assert "network:" in analysis
