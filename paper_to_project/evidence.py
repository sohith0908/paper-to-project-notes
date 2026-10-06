"""Claim tags and evidence spans — from_paper only when a key-term span exists."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

CLAIM_TAGS = ("from_paper", "suggested", "not_found", "plan", "inferred", "unverified")


@dataclass
class FieldEvidence:
    field: str
    value: str
    source_sentence: str = ""
    confidence: str = "low"
    note: str = ""
    provenance: str = "not_found"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_span(s: str, title: str = "") -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    s = re.sub(r"(?i)^\s*Abstract\s*[:\-]?\s*", "", s)
    if title:
        t = re.sub(r"\s+", " ", title.strip())
        if t and s.lower().startswith(t.lower()):
            s = s[len(t) :].lstrip(" :-–—")
            s = re.sub(r"(?i)^\s*Abstract\s*[:\-]?\s*", "", s)
    if title and s.lower().strip() == title.lower().strip():
        return ""
    return s.strip()[:320]


def whole_word_in(span: str, term: str) -> bool:
    if not span or not term:
        return False
    return bool(re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", span))


def key_terms_for(field: str, value: str) -> list[str]:
    v = (value or "").strip()
    if field == "title":
        return [v] if v else []
    if field == "contribution":
        return ["evaluation metric", "human judgments", "we propose", "this survey"]
    if field == "component":
        return [v] if v else []
    if field == "metrics":
        terms = []
        for name in (
            "Pearson", "Spearman", "Kendall", "BLEU", "METEOR", "ROUGE",
            "BERTScore", "correlation", "accuracy", "mAP", "F1", "WER",
        ):
            if re.search(rf"(?i)\b{re.escape(name)}\b", v):
                terms.append(name)
        if re.search(r"(?i)correlat", v) and not any(t.lower() == "correlation" for t in terms):
            terms.append("correlation")
        return terms
    if field == "dataset":
        if re.search(r"(?i)^not stated", v):
            return []
        terms = []
        for name in (
            "WMT16", "WMT17", "WMT18", "WMT", "PAWS", "COCO Captions", "COCO",
            "ImageNet", "CIFAR", "GLUE", "SQuAD", "STS-B", "LibriSpeech", "Cora",
        ):
            if re.search(rf"(?i)\b{re.escape(name)}\b", v):
                terms.append(name)
        return terms
    if field == "problem":
        if re.search(r"(?i)^problem not stated", v):
            return []
        return [w for w in re.findall(r"[A-Za-z]{5,}", v) if w.lower() not in {
            "paper", "stated", "provided", "text", "approaches", "argues"
        }][:4]
    return [w for w in re.findall(r"[A-Za-z]{5,}", v) if w.lower() not in {
        "paper", "stated", "provided", "text"
    }][:4]


def span_proves(span: str, key_terms: list[str], *, whole_word: bool = False) -> bool:
    if not span or not key_terms:
        return False
    for t in key_terms:
        if len(t) < 3:
            continue
        if whole_word or len(t) <= 6:
            if whole_word_in(span, t):
                return True
        elif t.lower() in span.lower():
            return True
    return False


def find_proving_span(
    text: str,
    key_terms: list[str],
    title: str = "",
    *,
    whole_word: bool = False,
    field: str = "",
) -> str:
    if not text or not key_terms:
        return ""
    if field == "title" and title:
        return title.strip()[:320]

    parts = re.split(r"(?<=[.!?])\s+", text)
    # Try terms in order so callers can put preferred terms first
    for term in key_terms:
        for s in parts:
            sl = s.strip()
            if len(sl) < 10:
                continue
            if re.match(r"(?i)^(abstract|introduction|keywords?|references)\b", sl):
                continue
            if title and sl.lower() == title.lower() and field != "title":
                continue
            cleaned = _clean_span(sl, title) if field != "title" else sl
            if span_proves(cleaned, [term], whole_word=whole_word or field == "component"):
                return cleaned[:320]
    return ""


def is_body_span(span: str, abstract: str) -> bool:
    if not span or not abstract:
        return bool(span)
    return span.strip().lower() not in abstract.lower() and span[:40].lower() not in abstract.lower()


def confidence_from_score(score: int, runner_up: int = 0) -> str:
    margin = score - runner_up
    if score >= 8 and margin >= 3:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def attach_evidence(
    field: str,
    value: str,
    text: str,
    *,
    confidence: str | None = None,
    note: str = "",
    title: str = "",
    provenance: str | None = None,
    key_terms: list[str] | None = None,
    suggested: bool = False,
) -> FieldEvidence:
    if not value or re.search(r"(?i)^(?:not (?:stated|found)|problem not stated)", value or ""):
        return FieldEvidence(field, value or "", "", "low", note or "empty", "not_found")

    if provenance == "plan" or (value or "").startswith("[Plan]"):
        return FieldEvidence(field, value, "", confidence or "medium", note or "plan", "plan")

    terms = key_terms if key_terms is not None else key_terms_for(field, value)
    if not terms:
        return FieldEvidence(field, value, "", "low", note or "no key terms", "not_found")

    # Metrics: prefer correlation / baseline names over the proposed metric name
    if field == "metrics":
        priority = [
            t for t in terms
            if t.lower() in {
                "pearson", "kendall", "spearman", "correlation",
                "bleu", "meteor", "rouge", "accuracy", "map", "f1", "wer",
            }
        ]
        rest = [t for t in terms if t not in priority]
        terms = priority + rest

    ww = field == "component"
    src = find_proving_span(text, terms, title=title, whole_word=ww, field=field)

    if suggested and not src:
        return FieldEvidence(field, value, "", "low", note or "suggested, not stated", "suggested")

    if provenance == "inferred":
        return FieldEvidence(field, value, src, confidence or "medium", note or "inferred", "inferred")

    if src and span_proves(src, terms, whole_word=ww):
        return FieldEvidence(
            field, value, src,
            confidence or ("high" if len(src) > 40 else "medium"),
            note or "from_paper",
            "from_paper",
        )

    return FieldEvidence(field, value, "", "low", note or "no proving span", "not_found")
