"""Abstract sentence role tagging + problem contrast derivation."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TaggedSentence:
    text: str
    role: str


_DISCOURSE = re.compile(
    r"(?i)^(however|finally|moreover|furthermore|also|thus|therefore)\s*,?\s+"
)

RESULT_VERBS = re.compile(
    r"(?i)\b(correlates?\s+better|outperform(?:s|ed)?|achieves?|improves?|"
    r"gains?|state-of-the-art|more robust|better than)\b"
)

PROBLEM = re.compile(
    r"(?i)\b(existing|limited|fail|challenge|lack|problem|struggle|"
    r"insufficient|poor|cannot|unable|open question|not fully|"
    r"rely on exact|do not capture|depend on exact)\b"
)
METHOD = re.compile(
    r"(?i)\b(we (?:propose|introduce|present|compute|use|develop|design|construct)|"
    r"our (?:method|metric|approach|model)|instead of)\b"
)
RESULT = re.compile(
    r"(?i)\b(outperform|better|achieves?|correlat(?:es|ion)|improves?|gains?|"
    r"state-of-the-art|more robust)\b"
)
EVAL = re.compile(
    r"(?i)\b(we (?:evaluate|compare|benchmark)|to show that|on (?:WMT|COCO|GLUE|ImageNet)|"
    r"human judgments?|shared tasks?)\b"
)
CONTRAST = re.compile(
    r"(?i)\b(instead of|rather than|unlike existing|in contrast to existing|"
    r"as opposed to|existing metrics? (?:rely|depend|use|fail))\b"
)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def strip_title_abstract_leak(sentence: str, title: str = "") -> str:
    s = (sentence or "").strip()
    s = re.sub(r"(?i)^\s*Abstract\s*[:\-]?\s*", "", s)
    if title:
        t = re.sub(r"\s+", " ", title.strip())
        if t and s.lower().startswith(t.lower()):
            s = s[len(t) :].lstrip(" :-–—\n")
            s = re.sub(r"(?i)^\s*Abstract\s*[:\-]?\s*", "", s)
    return s.strip()


def is_result_sentence(s: str) -> bool:
    return bool(RESULT_VERBS.search(s))


# PDF / equation noise — never use as problem/method prose
# LNWD-style tokens stay case-sensitive so "large" does not match.
_EQ_NOISE_CS = re.compile(
    r"("
    r"\\[a-zA-Z]+|"
    r"[=≈≤≥]\s*\d|"
    r"\bL_?[A-Z]{2,}|"
    r"µ[ab]|"
    r"^\s*\d+\s*\(\d+\)|"
    r"\(\d+\)\s*$|"
    r"^\s*\d+\.\d+\.\s*|"
    r"[]|"
    r"W\s*2\s*2\s*\("
    r")"
)
_EQ_NOISE_CI = re.compile(
    r"(?i)(\bNWD\s*\(|\bthe above equation\b|\bdistance measure\b|\bcannot be used directly\b)"
)


def is_noisy_sentence(s: str) -> bool:
    """True for equation fragments, section crumbs, or non-prose PDF junk."""
    if not s or len(s.strip()) < 25:
        return True
    if _EQ_NOISE_CS.search(s) or _EQ_NOISE_CI.search(s):
        return True
    letters = len(re.findall(r"[A-Za-z]", s))
    digits = len(re.findall(r"\d", s))
    if letters < 20 and digits >= 4:
        return True
    if digits > letters:
        return True
    return False


def clean_prose_sentence(s: str, title: str = "") -> str:
    s = strip_title_abstract_leak(s, title)
    s = re.sub(r"\s+", " ", s).strip()
    # Drop trailing section crumbs
    s = re.sub(r"\s+\d+\.\d+\.?\s*$", "", s).strip()
    return s


def tag_sentence(s: str) -> str:
    if is_result_sentence(s) and not re.search(r"(?i)\bwe (?:propose|compute|introduce)\b", s):
        return "result"
    scores = {
        "problem": 1 if PROBLEM.search(s) else 0,
        "method": 2 if METHOD.search(s) else 0,
        "result": 2 if RESULT.search(s) else 0,
        "eval": 1 if EVAL.search(s) else 0,
    }
    if re.search(r"(?i)\bwe (?:propose|introduce|present|compute|develop)\b", s):
        scores["method"] += 4
        scores["problem"] = 0
    if re.search(r"(?i)\bto show that\b|\bwe evaluate\b|\bwe compare\b", s):
        scores["eval"] += 4
        scores["method"] = max(0, scores["method"] - 2)
    if re.search(r"(?i)\bcorrelat", s) and re.search(r"(?i)\bhuman\b", s):
        scores["result"] += 4
        scores["method"] = 0
        scores["problem"] = 0
    if re.search(r"(?i)\bmore robust\b|\boutperform", s):
        scores["result"] += 2
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def _strip_discourse(s: str) -> str:
    cleaned = _DISCOURSE.sub("", s).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or s


def tagged_abstract(abstract: str, title: str = "") -> list[TaggedSentence]:
    raw = [strip_title_abstract_leak(s, title) for s in split_sentences(abstract)]
    raw = [s for s in raw if len(s) > 20]
    tagged: list[TaggedSentence] = []
    for s in raw:
        role = tag_sentence(s)
        if _DISCOURSE.match(s):
            # Keep problem/method after stripping However/Finally; drop empty glue
            if role in {"problem", "method", "result", "eval"}:
                tagged.append(TaggedSentence(_strip_discourse(s), role))
            continue
        tagged.append(TaggedSentence(s, role))
    return tagged


def sentences_for_role(abstract: str, role: str, limit: int = 3, title: str = "") -> list[str]:
    items = [t.text for t in tagged_abstract(abstract, title=title) if t.role == role]
    out: list[str] = []
    seen = set()
    for s in items:
        s = clean_prose_sentence(s, title)
        if is_noisy_sentence(s):
            continue
        if role == "problem" and is_result_sentence(s):
            continue
        key = re.sub(r"\s+", " ", s.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def derive_problem_from_contrast(abstract: str, title: str = "") -> str:
    """Return an existing problem sentence from the text only — never invent wording."""
    for s in split_sentences(strip_title_abstract_leak(abstract, title)):
        s = clean_prose_sentence(s, title)
        if is_result_sentence(s) or is_noisy_sentence(s):
            continue
        # Prefer an explicit existing-metrics / gap sentence already in the paper
        if re.search(r"(?i)existing metrics?\s+(?:rely|depend|use|fail)", s):
            return s
        if CONTRAST.search(s) and PROBLEM.search(s) and not re.search(r"(?i)\bwe (?:propose|compute)\b", s):
            return s
        if PROBLEM.search(s) and re.search(r"(?i)struggle|challenge|poor|fail", s):
            return s
    return ""


def exclude_used(candidates: list[str], used: set[str]) -> list[str]:
    out = []
    for s in candidates:
        key = re.sub(r"\s+", " ", s.lower())
        if key in used:
            continue
        out.append(s)
    return out
