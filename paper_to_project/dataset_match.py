"""Match a user-uploaded dataset against what the paper claims."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field


@dataclass
class DatasetMatch:
    score: float  # 0..1
    summary: str
    overlaps: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


def _paper_tokens(dataset_text: str) -> set[str]:
    words = set(re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", dataset_text.lower()))
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
        "dataset", "data", "set", "used", "using", "public", "available", "images",
        "image", "video", "videos", "train", "test", "samples", "containing",
    }
    return {w for w in words if w not in stop}


def match_tabular_dataset(
    file_name: str,
    file_bytes: bytes,
    paper_dataset: str,
    paper_metrics: str = "",
) -> DatasetMatch:
    """Inspect CSV/TSV headers + row count and compare to paper dataset description."""
    name = file_name.lower()
    overlaps: list[str] = []
    gaps: list[str] = []
    suggestions: list[str] = []
    paper_toks = _paper_tokens(paper_dataset + " " + paper_metrics)

    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return DatasetMatch(0.0, "Could not decode dataset file as text.")

    dialect = csv.excel_tab if name.endswith(".tsv") else csv.excel
    reader = csv.reader(io.StringIO(text), dialect)
    rows = list(reader)
    if not rows:
        return DatasetMatch(0.0, "Uploaded file is empty.")

    headers = [h.strip() for h in rows[0]]
    n = max(0, len(rows) - 1)
    header_l = {h.lower() for h in headers}

    # Overlaps: header tokens vs paper tokens
    for h in header_l:
        for t in paper_toks:
            if t in h or h in t:
                overlaps.append(f"column `{h}` ~ paper term `{t}`")
                break

    # Modality hints
    if any(k in paper_dataset.lower() for k in ("image", "video", "frame", "bbox", "bounding")):
        if name.endswith((".csv", ".tsv")) and not any(
            k in " ".join(header_l) for k in ("path", "image", "file", "bbox", "xmin", "label")
        ):
            gaps.append(
                "Paper looks vision-oriented, but upload looks like a plain table without image/bbox columns."
            )
            suggestions.append(
                "Use a folder of images + labels, or a CSV with image paths and class/bbox fields."
            )

    if any(k in paper_dataset.lower() for k in ("review", "text", "tweet", "sentiment", "nlp")):
        if not any(k in " ".join(header_l) for k in ("text", "review", "sentence", "content", "tweet")):
            gaps.append("Paper looks text/NLP, but no obvious text column found.")
            suggestions.append("Need a text column plus a label column.")

    # Size honesty
    if re.search(r"\b(\d{4,})\b", paper_dataset):
        m = re.search(r"\b(\d{4,})\b", paper_dataset)
        paper_n = int(m.group(1)) if m else 0
        if paper_n and n and n < paper_n * 0.2:
            gaps.append(
                f"Paper mentions ~{paper_n} samples; your file has {n} rows (much smaller)."
            )
            suggestions.append(
                "Fine for a mini project — say you used a subset and do not claim paper-scale results."
            )
        elif paper_n and n:
            overlaps.append(f"row count {n} vs paper mention ~{paper_n}")

    score = 0.2
    score += min(0.5, 0.15 * len(overlaps))
    score -= min(0.4, 0.15 * len(gaps))
    score = max(0.0, min(1.0, score))

    summary = (
        f"Compared `{file_name}` ({n} rows, columns: {', '.join(headers[:8])}{'…' if len(headers) > 8 else ''}) "
        f"to the paper dataset description. Match score: {score:.0%}."
    )
    if not overlaps and not gaps:
        suggestions.append(
            "Could not strongly match fields — manually justify why this data is a valid proxy."
        )
    return DatasetMatch(score, summary, overlaps[:8], gaps[:8], suggestions[:6])


def match_dataset_description(user_desc: str, paper_dataset: str) -> DatasetMatch:
    """Free-text dataset note from the student vs paper dataset section."""
    ut = _paper_tokens(user_desc)
    pt = _paper_tokens(paper_dataset)
    if not ut or not pt:
        return DatasetMatch(
            0.0,
            "Need both a paper dataset section and your dataset description.",
            suggestions=["Paste what data you will use (name, size, labels)."],
        )
    inter = sorted(ut & pt)
    score = len(inter) / max(1, len(pt))
    overlaps = [f"shared term `{t}`" for t in inter[:8]]
    missing = sorted(pt - ut)[:6]
    gaps = [f"paper term not mentioned: `{t}`" for t in missing]
    suggestions = []
    if score < 0.25:
        suggestions.append(
            "Your dataset may be a weak proxy — explicitly list what differs (classes, size, domain)."
        )
    return DatasetMatch(
        score=max(0.0, min(1.0, score)),
        summary=f"Text overlap score {score:.0%} between your dataset note and the paper dataset section.",
        overlaps=overlaps,
        gaps=gaps,
        suggestions=suggestions,
    )
