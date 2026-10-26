"""Post-processing utilities for diarized transcripts and reports."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterable

import srt


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "must", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
}

QUESTION_STARTS = {
    "what", "why", "how", "when", "who", "which", "where", "can", "could", "should",
    "would", "is", "are", "do", "did", "does", "will", "may", "might",
}

ACTION_PATTERNS = [
    r"\bwe need to\b",
    r"\bneed to\b",
    r"\bwe should\b",
    r"\bwe must\b",
    r"\blet's\b",
    r"\bwe will\b",
    r"\bwe're going to\b",
    r"\bgoing to\b",
    r"\bi will\b",
    r"\bi'll\b",
    r"\bwe can\b",
]


@dataclass
class DiarizedSegmentSimple:
    start: float
    end: float
    text: str
    speaker: str | None = None


def load_whisperx_json(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def build_diarized_json(
    segments: list[DiarizedSegmentSimple],
    language: str | None = None,
) -> dict:
    data_segments = []
    for seg in segments:
        entry = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        }
        if seg.speaker:
            entry["speaker"] = seg.speaker
        data_segments.append(entry)
    return {
        "segments": data_segments,
        "word_segments": [],
        "language": language or "unknown",
    }


def _speaker_sort_key(label: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", label)
    if match:
        return (int(match.group(1)), label)
    return (10_000, label)


def extract_speaker_labels(data: dict) -> list[str]:
    labels = set()
    for seg in data.get("segments", []):
        speaker = seg.get("speaker_raw") or seg.get("speaker")
        if speaker:
            labels.add(speaker)
    return sorted(labels, key=_speaker_sort_key)


def generate_speaker_config(config_path: Path, source_path: Path, labels: list[str]) -> Path:
    config = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(Path(source_path).expanduser().resolve()),
        "speakers": [
            {"id": label, "name": f"Speaker {idx + 1}"} for idx, label in enumerate(labels)
        ],
    }
    return write_json(config_path, config)


def load_speaker_config(config_path: Path) -> dict[str, str]:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    mapping = {}
    for entry in config.get("speakers", []):
        speaker_id = entry.get("id")
        name = entry.get("name")
        if speaker_id and name:
            mapping[speaker_id] = name
    return mapping


def apply_speaker_mapping(data: dict, mapping: dict[str, str]) -> dict:
    updated = json.loads(json.dumps(data))
    for seg in updated.get("segments", []):
        # Key on the original diarizer label so mappings can be re-applied
        speaker = seg.get("speaker_raw") or seg.get("speaker")
        if speaker and speaker in mapping:
            seg["speaker_raw"] = speaker
            seg["speaker"] = mapping[speaker]
        for word in seg.get("words", []) or []:
            word_speaker = word.get("speaker_raw") or word.get("speaker")
            if word_speaker and word_speaker in mapping:
                word["speaker_raw"] = word_speaker
                word["speaker"] = mapping[word_speaker]
    return updated


def segments_from_whisperx_json(data: dict) -> list[DiarizedSegmentSimple]:
    segments = []
    for seg in data.get("segments", []):
        segments.append(
            DiarizedSegmentSimple(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
                speaker=seg.get("speaker"),
            )
        )
    return segments


def segments_from_srt(path: Path) -> list[DiarizedSegmentSimple]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))
    segments = []
    for sub in subtitles:
        segments.append(
            DiarizedSegmentSimple(
                start=sub.start.total_seconds(),
                end=sub.end.total_seconds(),
                text=sub.content.strip(),
                speaker=None,
            )
        )
    return segments


def _format_timestamp_srt(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_ms = int(td.total_seconds() * 1000)
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    return _format_timestamp_srt(seconds).replace(",", ".")


def render_diarized_txt(segments: list[DiarizedSegmentSimple]) -> str:
    lines = []
    current_speaker = None
    for seg in segments:
        speaker = seg.speaker or "Speaker"
        if speaker != current_speaker:
            current_speaker = speaker
            lines.append(f"\n[{speaker}]")
        lines.append(seg.text)
    return " ".join(lines).strip()


def render_diarized_srt(segments: list[DiarizedSegmentSimple]) -> str:
    subtitles = []
    for idx, seg in enumerate(segments, start=1):
        speaker = seg.speaker or "Speaker"
        content = f"{speaker}: {seg.text}" if seg.text else speaker
        subtitles.append(
            srt.Subtitle(
                index=idx,
                start=timedelta(seconds=seg.start),
                end=timedelta(seconds=seg.end),
                content=content.strip(),
            )
        )
    return srt.compose(subtitles)


def render_diarized_vtt(segments: list[DiarizedSegmentSimple]) -> str:
    srt_text = render_diarized_srt(segments)
    lines = ["WEBVTT", ""]
    for line in srt_text.splitlines():
        if "-->" in line:
            lines.append(line.replace(",", "."))
        else:
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def render_diarized_tsv(segments: list[DiarizedSegmentSimple]) -> str:
    rows = [("speaker", "start", "end", "text")]
    for seg in segments:
        speaker = seg.speaker or "Speaker"
        rows.append((speaker, int(seg.start * 1000), int(seg.end * 1000), seg.text))
    output = []
    for row in rows:
        output.append("\t".join(str(item) for item in row))
    return "\n".join(output) + "\n"


def write_diarized_outputs(
    base_name: str,
    output_dir: Path,
    updated_json: dict,
    segments: list[DiarizedSegmentSimple],
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    json_path = output_dir / f"{base_name}.json"
    outputs["json"] = write_json(json_path, updated_json)

    txt_path = output_dir / f"{base_name}.txt"
    txt_path.write_text(render_diarized_txt(segments), encoding="utf-8")
    outputs["txt"] = txt_path

    srt_path = output_dir / f"{base_name}.srt"
    srt_path.write_text(render_diarized_srt(segments), encoding="utf-8")
    outputs["srt"] = srt_path

    vtt_path = output_dir / f"{base_name}.vtt"
    vtt_path.write_text(render_diarized_vtt(segments), encoding="utf-8")
    outputs["vtt"] = vtt_path

    tsv_path = output_dir / f"{base_name}.tsv"
    tsv_path.write_text(render_diarized_tsv(segments), encoding="utf-8")
    outputs["tsv"] = tsv_path

    return outputs


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [t.strip("'") for t in tokens if t.strip("'")]


def _filter_terms(tokens: Iterable[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 3]


def _count_ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _filter_ngrams(ngrams: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    filtered = {}
    for gram, count in ngrams.items():
        content_tokens = [t for t in gram if t not in STOPWORDS and len(t) >= 3]
        if len(gram) == 2 and len(content_tokens) == 2:
            filtered[gram] = count
        elif len(gram) == 3 and len(content_tokens) >= 2:
            filtered[gram] = count
    return filtered


def _top_items(counts: dict, limit: int) -> list[tuple[str, int]]:
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [(" ".join(item) if isinstance(item, tuple) else item, count) for item, count in sorted_items[:limit]]


def _is_question(text: str) -> bool:
    if "?" in text:
        return True
    tokens = _tokenize(text)
    return bool(tokens and tokens[0] in QUESTION_STARTS)


def _is_action_candidate(text: str) -> bool:
    lower = text.lower()
    if "?" in lower:
        return False
    return any(re.search(pattern, lower) for pattern in ACTION_PATTERNS)


def generate_reports(
    segments: list[DiarizedSegmentSimple],
    output_dir: Path,
    source_path: Path,
    top_n: int = 20,
    highlights: int = 12,
) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    full_text = " ".join(seg.text for seg in segments if seg.text)
    tokens = _tokenize(full_text)
    content_tokens = _filter_terms(tokens)

    term_counts = {}
    for token in content_tokens:
        term_counts[token] = term_counts.get(token, 0) + 1

    bigram_counts = _filter_ngrams(_count_ngrams(tokens, 2))
    trigram_counts = _filter_ngrams(_count_ngrams(tokens, 3))

    top_terms = _top_items(term_counts, top_n)
    top_bigrams = _top_items(bigram_counts, min(15, top_n))
    top_trigrams = _top_items(trigram_counts, min(15, top_n))

    question_segments = [seg for seg in segments if _is_question(seg.text)]
    action_segments = [seg for seg in segments if _is_action_candidate(seg.text)]

    duration = segments[-1].end if segments else 0.0
    word_count = len(content_tokens)

    analysis_path = output_dir / "analysis-summary.md"
    analysis_lines = [
        "# Transcript Analysis Summary",
        "",
        f"- Source: `{Path(source_path).expanduser().resolve()}`",
        f"- Subtitle entries: **{len(segments)}**",
        f"- Estimated duration: **{duration / 60:.1f} minutes**",
        f"- Approximate words: **{word_count}**",
        f"- Question lines detected: **{len(question_segments)}**",
        f"- Action-candidate lines detected: **{len(action_segments)}**",
        "",
        "## Top Terms",
    ]
    analysis_lines += [f"- {term}: {count}" for term, count in top_terms]
    analysis_lines += ["", "## Top Bigrams"]
    analysis_lines += [f"- {term}: {count}" for term, count in top_bigrams]
    analysis_lines += ["", "## Top Trigrams"]
    analysis_lines += [f"- {term}: {count}" for term, count in top_trigrams]
    analysis_lines += ["", "## Timeline Highlights"]

    highlight_indices = []
    if segments:
        step = max(1, len(segments) // highlights)
        highlight_indices = list(range(0, len(segments), step))[:highlights]

    for idx in highlight_indices:
        seg = segments[idx]
        analysis_lines.append(f"- `{_format_timestamp_srt(seg.start)}` {seg.text[:120]}")
    analysis_path.write_text("\n".join(analysis_lines).strip() + "\n", encoding="utf-8")

    actions_path = output_dir / "action-candidates.md"
    action_lines = ["# Action Candidates", ""]
    for seg in action_segments:
        action_lines.append(f"- `{_format_timestamp_srt(seg.start)}` {seg.text}")
    actions_path.write_text("\n".join(action_lines).strip() + "\n", encoding="utf-8")

    questions_path = output_dir / "question-candidates.md"
    question_lines = ["# Question Candidates", ""]
    for seg in question_segments:
        question_lines.append(f"- `{_format_timestamp_srt(seg.start)}` {seg.text}")
    questions_path.write_text("\n".join(question_lines).strip() + "\n", encoding="utf-8")

    timeline_path = output_dir / "timeline-highlights.csv"
    with timeline_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "text"])
        for idx in highlight_indices:
            seg = segments[idx]
            writer.writerow([_format_timestamp_srt(seg.start), seg.text])

    metadata_path = output_dir / "metadata.json"
    metadata = {
        "source": str(Path(source_path).expanduser().resolve()),
        "segments": len(segments),
        "duration_seconds": duration,
        "word_count": word_count,
        "question_candidates": len(question_segments),
        "action_candidates": len(action_segments),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(metadata_path, metadata)

    final_summary_path = output_dir / "final-summary.md"
    final_summary = [
        "# Final Summary (Auto-Generated)",
        "",
        "## Context",
        "Auto-generated summary based on transcript frequency signals and heuristics.",
        "",
        "## Core Themes",
        "- Frequent terms: " + ", ".join(term for term, _ in top_terms[:8]),
        "- Common phrases: " + ", ".join(term for term, _ in top_bigrams[:5]),
        "",
        "## Action Signals Observed",
    ]
    if action_segments:
        final_summary += [f"- {_format_timestamp_srt(seg.start)} {seg.text}" for seg in action_segments[:5]]
    else:
        final_summary.append("- No strong action signals detected by heuristics.")
    final_summary += [
        "",
        "## Risks / Uncertainties",
        "- Transcript quality may introduce ambiguity in intent and ownership.",
        "- Heuristic extraction may miss nuanced decisions.",
        "",
        "## Recommended Next Steps",
        "1. Review speaker mapping and confirm ownership of action items.",
        "2. Promote action candidates into a tracked task list.",
        "3. Validate key decisions against meeting context.",
    ]
    final_summary_path.write_text("\n".join(final_summary).strip() + "\n", encoding="utf-8")

    engineering_path = output_dir / "final-summary-v2-engineering.md"
    engineering = [
        "# Final Summary V2 (Auto-Generated)",
        "",
        "## Session Signal",
        "Auto-generated engineering summary from transcript signals.",
        "",
        "## Engineering Readout",
        "- Frequent terms: " + ", ".join(term for term, _ in top_terms[:8]),
        "",
        "## Proposed Workstreams",
    ]
    workstreams = []
    terms_set = {term for term, _ in top_terms}
    if {"model", "orchestration", "selector", "agent", "llm"} & terms_set:
        workstreams.append("### 1) Model Orchestration Runtime")
    if {"gpu", "compute", "local", "inference"} & terms_set:
        workstreams.append("### 2) Local Inference Platform")
    if {"time", "network", "sync", "precision"} & terms_set:
        workstreams.append("### 3) Time/Network Reliability Layer")
    if not workstreams:
        workstreams.append("### 1) Core Platform Stabilization")
    engineering += workstreams
    engineering += [
        "",
        "## Concrete Next Actions",
    ]
    if action_segments:
        engineering += [f"1. {_format_timestamp_srt(seg.start)} {seg.text}" for seg in action_segments[:5]]
    else:
        engineering.append("1. Review transcript for actionable items.")
    engineering += [
        "",
        "## Risks to Track",
        "- Heuristic extraction may omit implicit decisions.",
        "- Ownership assignment is not captured in transcript text.",
    ]
    engineering_path.write_text("\n".join(engineering).strip() + "\n", encoding="utf-8")

    return {
        "analysis": analysis_path,
        "actions": actions_path,
        "questions": questions_path,
        "timeline": timeline_path,
        "metadata": metadata_path,
        "final_summary": final_summary_path,
        "final_summary_v2": engineering_path,
    }
