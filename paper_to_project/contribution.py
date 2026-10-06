"""Contribution axis: what the paper delivers (orthogonal to task domain)."""

from __future__ import annotations

import re
from dataclasses import dataclass


CONTRIBUTIONS = (
    "new_metric",
    "new_dataset_benchmark",
    "training_technique",
    "survey",
    "new_model",
)


@dataclass
class ContributionScore:
    contribution: str
    score: int
    matched: list[str]


CUES: dict[str, list[tuple[str, int]]] = {
    "new_metric": [
        (r"\bevaluation metric\b", 6),
        (r"\bautomatic (?:evaluation )?metric\b", 5),
        (r"\bcorrelat(?:es|ion) with human\b", 7),
        (r"\bhuman judgments?\b", 5),
        (r"\bexisting metrics?\b", 4),
        (r"\bBERTScore\b", 8),
        (r"\bwe propose .{0,40}metric\b", 6),
        (r"\binstead of exact matches\b", 4),
        (r"\bPearson|Spearman|Kendall\b", 4),
        (r"\bBLEU|METEOR|ROUGE\b", 2),
    ],
    "new_dataset_benchmark": [
        (r"\bwe (?:introduce|release|present) .{0,40}(?:dataset|benchmark|corpus)\b", 7),
        (r"\bnew (?:dataset|benchmark)\b", 6),
        (r"\bannotated .{0,20}dataset\b", 4),
    ],
    "training_technique": [
        (r"\btraining (?:technique|method|recipe|trick)\b", 5),
        (r"\boptimization method\b", 4),
        (r"\bregularization (?:method|technique)\b", 4),
        (r"\blearning rate schedule\b", 3),
        (r"\bdata augmentation (?:strategy|method)\b", 3),
    ],
    "survey": [
        (r"\bthis survey\b", 6),
        (r"\bliterature review\b", 6),
        (r"\bwe (?:will )?review\b", 4),
        (r"\btaxonomy\b", 3),
    ],
    "new_model": [
        (r"\bwe propose .{0,80}(?:model|architecture|network|framework|method)\b", 5),
        (r"\bthis paper proposes .{0,80}(?:model|architecture|network|framework|method)\b", 5),
        (r"\bnovel (?:model|architecture|network)\b", 4),
        (r"\bour (?:proposed )?model\b", 3),
        (r"\binnovative .{0,40}(?:YOLO|model|network)\b", 4),
        (r"\bimproved YOLO\b", 5),
        (r"\bMSN-?YOLO\b", 6),
        (r"\bYOLOv?\d\w*.{0,40}ByteTrack|ByteTrack.{0,40}YOLOv?\d", 5),
    ],
}


def score_contributions(text: str) -> list[ContributionScore]:
    out: list[ContributionScore] = []
    for name, rules in CUES.items():
        total = 0
        matched: list[str] = []
        for pat, w in rules:
            if re.search(pat, text, flags=re.I):
                total += w
                matched.append(pat)
        out.append(ContributionScore(name, total, matched))
    out.sort(key=lambda x: (-x.score, x.contribution))
    return out


def detect_contribution(text: str) -> str:
    scores = score_contributions(text)
    by_name = {s.contribution: s for s in scores}
    if not scores or scores[0].score < 4:
        if re.search(r"(?i)\bsurvey|literature review|taxonomy\b", text):
            return "survey"
        if re.search(r"(?i)\btheorem|lemma|we prove|theoretical analysis\b", text):
            return "survey"  # study-style plan; not a new model claim
        if re.search(
            r"(?i)\bno (?:machine learning|ml|deep learning)|instrumentation|optics notes\b",
            text,
        ):
            return "survey"
        return "new_model"
    top = scores[0]
    # Prefer metric over model when both fire (BERTScore case)
    if top.contribution == "new_model":
        for s in scores[1:3]:
            if s.contribution == "new_metric" and s.score >= 4 and s.score >= top.score - 2:
                return "new_metric"
            if s.contribution == "new_dataset_benchmark" and s.score >= 5 and s.score >= top.score - 1:
                return "new_dataset_benchmark"
    # Detector/tracker architecture papers: do not call them training_technique
    if top.contribution == "training_technique":
        model = by_name.get("new_model")
        if re.search(r"(?i)\bYOLOv?\d|ByteTrack|object detection|multi-object tracking\b", text):
            if model and model.score >= 3:
                return "new_model"
            if top.score <= 5:
                return "new_model"
    # Dataset papers: prefer benchmark contribution when strong
    ds = by_name.get("new_dataset_benchmark")
    if ds and ds.score >= 6 and ds.score >= top.score - 1:
        return "new_dataset_benchmark"
    return top.contribution
