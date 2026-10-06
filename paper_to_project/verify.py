"""Final verification + student-output linter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .evidence import FieldEvidence, find_proving_span, key_terms_for, span_proves, whole_word_in
from .sentences import is_result_sentence


BANNED_IN_STUDENT = [
    r"(?i)lexicon hit",
    r"(?i)\(abstract only\?\)",
    r"(?i)draft a thin",
    r"(?i)do not invent",
    r"(?i)when present",
    r"(?i)when reported",
    r"(?i)do not assume",
    r"(?i)\be\.g\.",
    r"(?i)20-second definition",
    r"(?i)check the component table",
    r"(?i)explain it in one sentence",
    r"(?i)if the paper describes",
    r"(?i)no proving span",
    r"(?i)no second candidate above threshold",
    r"(?i)\[Plan\]",
    r"(?i)\(verified\)",
    r"(?i)\bverified\b",
    r"(?i)inferred encoder load",
    r"(?i)^#\s*Debug\b",
    r"(?i)none listed by name",
    r"(?i)existing metrics named in the paper \(none",
]

INVENTED_PROBLEM_WORDS = re.compile(
    r"(?i)\b(argues?|insufficient|inadequate|flawed|we claim)\b"
)
REASON_CUE_RE = re.compile(r"(?i)\b(because|so that|in order to|since|as a result)\b")


@dataclass
class VerifyReport:
    ok: bool = True
    downgraded: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _problem_words_ok(problem: str, source_text: str) -> bool:
    """Every content word in problem should appear in source (allow tiny glue words)."""
    if not problem or problem.startswith("Problem not stated"):
        return True
    if INVENTED_PROBLEM_WORDS.search(problem):
        return False
    src = source_text.lower()
    for w in re.findall(r"[A-Za-z]{5,}", problem):
        if w.lower() not in src:
            return False
    return True


def verify_notes(notes, text: str, title: str = "") -> VerifyReport:
    report = VerifyReport()
    title = title or getattr(notes, "title", "") or ""

    if notes.problem and is_result_sentence(notes.problem):
        notes.problem = "Problem not stated in the provided text."
        report.issues.append("problem_result_verbs")

    if notes.problem and not _problem_words_ok(notes.problem, text):
        # Prefer an exact source sentence if available
        from .sentences import sentences_for_role, derive_problem_from_contrast
        abs_like = text[:4000]
        role = sentences_for_role(abs_like, "problem", limit=1, title=title)
        if role and _problem_words_ok(role[0], text) and not is_result_sentence(role[0]):
            notes.problem = role[0]
        else:
            derived = derive_problem_from_contrast(abs_like, title=title)
            if derived and _problem_words_ok(derived, text):
                notes.problem = derived
            else:
                notes.problem = "Problem not stated in the provided text."
        report.downgraded.append("problem:invented_words")

    if not notes.problem or notes.problem.strip() in {"", "Problem not clearly stated."}:
        notes.problem = "Problem not stated in the provided text."

    new_ev: list[FieldEvidence] = []
    for e in list(notes.evidence or []):
        if e.provenance in {"plan", "suggested", "inferred"}:
            if re.search(r"(?i)^(?:not (?:stated|found)|problem not stated)", e.value or ""):
                e.source_sentence = ""
                e.provenance = "not_found"
                e.confidence = "low"
            new_ev.append(e)
            continue
        terms = key_terms_for(e.field, e.value)
        ww = e.field == "component"
        if e.field == "metrics":
            priority = [
                t for t in terms
                if t.lower() in {
                    "pearson", "kendall", "spearman", "correlation",
                    "bleu", "meteor", "rouge", "accuracy", "map", "f1", "wer",
                }
            ]
            terms = priority + [t for t in terms if t not in priority]
        span = ""
        if terms:
            if e.source_sentence and span_proves(
                e.source_sentence, terms[:1] if e.field == "metrics" and terms else terms, whole_word=ww
            ):
                span = e.source_sentence
            else:
                span = find_proving_span(text, terms, title=title, whole_word=ww, field=e.field)
        if re.search(r"(?i)^(?:not (?:stated|found)|problem not stated)", e.value or ""):
            e.source_sentence = ""
            e.provenance = "not_found"
            e.confidence = "low"
        elif span:
            e.source_sentence = span
            e.provenance = "from_paper"
        else:
            e.source_sentence = ""
            e.provenance = "not_found"
            e.confidence = "low"
            report.downgraded.append(e.field)
        new_ev.append(e)
    notes.evidence = new_ev

    unverified = list(getattr(notes, "unverified_components", None) or [])
    keep_status = []
    for row in list(notes.status_table or []):
        if not whole_word_in(text, row.name) and not (
            title and whole_word_in(title, row.name)
        ):
            if row.name not in unverified:
                unverified.append(row.name)
            report.removed.append(row.name)
            continue
        # from_paper requires a proving span containing the name
        if row.provenance == "from_paper":
            span = find_proving_span(
                text, [row.name], title=title, whole_word=True, field="component"
            )
            if not span and title and whole_word_in(title, row.name):
                span = title
            if not span:
                row.provenance = "inferred"
                row.reason = f"The text mentions {row.name}; details are limited."
                report.downgraded.append(f"no_span:{row.name}")
        # Strip developer phrases from reasons
        row.reason = re.sub(r"(?i)no proving span\.?", "", row.reason).strip()
        row.reason = re.sub(r"(?i)inferred encoder load[^.]*\.?", "", row.reason).strip()
        if not row.reason:
            row.reason = f"{row.name} is part of the method."
        keep_status.append(row)
    notes.status_table = keep_status
    notes.unverified_components = unverified
    notes.components = [c for c in notes.components if c not in unverified]

    # Strip NLP/metric leakage from detection rows; align metric backbones with freeze plan
    plan_blob = " ".join(notes.python_scope).lower()
    task = getattr(notes, "task_type", "") or ""
    contrib = getattr(notes, "contribution", "") or ""
    for row in notes.status_table:
        if task == "object_detection_tracking" and re.search(
            r"(?i)sentence pairs|metric pipeline|token embedding|pretrained encoder",
            f"{row.compute_note} {row.reason}",
        ):
            if row.role == "backbone" or re.search(r"(?i)yolo", row.name):
                row.compute_note = (
                    "GPU preferred; pretrained detector weights on a short video/image set."
                )
                row.reason = f"{row.name} is the detector backbone named in the paper."
            else:
                row.compute_note = "-" if row.status == "cite" else "CPU/GPU for a short video demo."
                row.reason = f"{row.name} is named in the paper method."
            continue
        if (
            contrib == "new_metric"
            and row.role == "backbone"
            and ("without training" in plan_blob or "no training" in plan_blob)
            and "fine-tun" in row.compute_note.lower()
        ):
            row.compute_note = "CPU for inference; load pretrained weights, no training."
            row.status = "implement" if row.provenance == "from_paper" else row.status
            row.reason = f"Load pretrained {row.name} weights without training."

    # Clean plan hedges
    clean_scope = []
    for s in notes.python_scope or []:
        if re.search(r"(?i)none listed|when present|e\.g\.|do not assume", s):
            if re.search(r"(?i)existing metrics", s):
                clean_scope.append("Compare against BLEU as a baseline.")
            continue
        s = re.sub(r"\s*;\s*\.", ".", s)
        s = re.sub(r"\.\s*;", ".", s)
        clean_scope.append(s)
    notes.python_scope = clean_scope

    report.ok = len(report.issues) == 0
    return report


def lint_student_output(md: str, notes=None) -> str:
    """Strip/repair banned developer strings; raise issues via silent cleanup."""
    out = md
    # Remove empty Debug sections entirely
    out = re.sub(r"(?im)^#\s*Debug\b.*?(?=^# |\Z)", "", out, flags=re.S)
    # Fix timeline artifacts
    out = re.sub(r";\s*\.", ".", out)
    out = re.sub(r"\.\s*;", ".", out)
    out = re.sub(r"\|\s*;\s*", "| ", out)
    # Drop banned phrases if any slipped in
    for pat in BANNED_IN_STUDENT:
        if re.search(pat, out):
            out = re.sub(pat, "", out)
    # Collapse leftover empty headings
    out = re.sub(r"(?m)^(## .+)\n(?:\s*\n)+(?=## |\Z)", r"\1\n\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"


def verify_markdown(md: str, task_type: str = "", contribution: str = "") -> list[str]:
    blockers = []
    for pat in BANNED_IN_STUDENT:
        if re.search(pat, md):
            blockers.append(pat)
    if re.search(r";\s*\.|\.\s*;", md):
        blockers.append("timeline_semicolon_artifact")
    if re.search(r"(?m)^#{1,3}\s+Debug\b", md):
        blockers.append("debug_heading")
    if re.search(r"(?m)^(#{1,3}\s+[^\n]+)\n(?:\s*\n)+(?=#{1,3}\s|\Z)", md):
        # empty section between headings
        for m in re.finditer(r"(?m)^(#{1,3}\s+[^\n]+)\n(?:\s*\n)+(?=#{1,3}\s|\Z)", md):
            body = re.sub(r"^#{1,3}\s+[^\n]+\n?", "", m.group(0)).strip()
            if not body:
                blockers.append(f"empty_heading:{m.group(1).strip()}")
    if re.search(r"(?i)## Q\d+\.\s*Why\b", md):
        for m in re.finditer(r"(?im)^## Q\d+\.\s*(Why[^\n]*)\n([^\n]*)", md):
            if not REASON_CUE_RE.search(m.group(1) + " " + m.group(2)):
                blockers.append(f"why_without_cue:{m.group(1)[:60]}")
    return blockers
