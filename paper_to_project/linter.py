"""Lint final Markdown for placeholder / cross-task leak strings."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .verify import INVENTED_PROBLEM_WORDS, REASON_CUE_RE, _problem_words_ok


@dataclass
class LintHit:
    pattern: str
    snippet: str
    severity: str = "error"


BANNED = [
    (r"(?i)20-second definition", "placeholder instruction in viva"),
    (r"(?i)status implement/approximate/cite", "placeholder instruction in viva"),
    (r"(?i)explain the evidence", "meta instruction leaked into answer"),
    (r"(?i)explain it in one sentence", "unresolved developer instruction"),
    (r"(?i)check the component table", "unresolved developer instruction"),
    (r"(?i)if the paper describes them", "unresolved conditional instruction"),
    (r"(?i)lexicon hit", "developer wording"),
    (r"(?i)\(abstract only\?\)", "developer wording"),
    (r"(?i)draft a thin", "developer wording"),
    (r"(?i)do not invent", "developer wording"),
    (r"(?i)when present", "unresolved hedge"),
    (r"(?i)when reported", "unresolved hedge"),
    (r"(?i)do not assume", "unresolved hedge"),
    (r"(?i)\be\.g\.", "hedge list"),
    (r"(?i)no proving span", "developer wording"),
    (r"(?i)inferred encoder load", "developer wording"),
    (r"(?i)^#\s*Debug\b", "debug heading in student output"),
    (r"(?i)\(verified\)|not verified in a proving|no proving span", "verified/developer label"),
    (r"(?i)\bverified\b", "verified label in student output"),
    (r"(?i)none listed by name", "hedge"),
    (r"(?i)named in the paper \(none", "hedge"),
    (r"(?i);\s*\.|\.\s*;", "timeline semicolon artifact"),
    (r"(?i):\s*A Practical Python Prototype\b", "title should be Study when required"),
    (r"(?im)^-\s*Implement contextual embeddings\.?\s*$", "vague technique step"),
]


TASK_BANNED: dict[str, list[tuple[str, str]]] = {
    # When task is NOT classification, ban ResNet-18 plan leaks in viva answers
    "nlp_text": [
        (r"(?i)ResNet-18", "ResNet-18 leak into non-vision paper"),
        (r"(?i)YOLO", "YOLO leak into NLP paper"),
        (r"(?i)fine-tuned a small BERT", "wrong for metric papers that freeze BERT"),
    ],
    "classic_tabular_ml": [
        (r"(?i)ResNet-18", "ResNet leak into tabular paper"),
        (r"(?i)YOLO", "YOLO leak into tabular paper"),
    ],
    "image_classification": [
        (r"(?i)\bYOLO\b", "YOLO leak into classification paper"),
    ],
}

CONTRIB_BANNED: dict[str, list[tuple[str, str]]] = {
    "new_metric": [
        (r"(?i)fine-tuned a small BERT", "metric papers do not fine-tune classifiers"),
        (r"(?i)TF-IDF \+ linear", "classification plan leaked into metric paper"),
        (r"(?i)SST-2 / IMDB", "classification substitute leaked into metric paper"),
        (r"(?i)A Practical Python Prototype", "metric papers use Study title"),
        (r"(?i)ResNet-18", "ResNet plan leak"),
        (r"(?i)20-second definition", "placeholder"),
    ],
}


def _snippet(md: str, start: int, end: int) -> str:
    lo = max(0, start - 40)
    hi = min(len(md), end + 40)
    return md[lo:hi].replace("\n", " ")


def lint_problem_against_source(problem: str, source_text: str) -> list[LintHit]:
    """Words in Problem must appear in the source sentence; no invented hedges."""
    hits: list[LintHit] = []
    if not problem or problem.startswith("Problem not stated"):
        return hits
    if INVENTED_PROBLEM_WORDS.search(problem):
        hits.append(LintHit("invented problem wording", problem[:120], "error"))
    if not _problem_words_ok(problem, source_text):
        hits.append(LintHit("problem words not in source", problem[:120], "error"))
    return hits


def lint_why_questions(md: str, reason_blob: str = "") -> list[LintHit]:
    """Flag Why-Qs that have no reason cue in question context or answer."""
    hits: list[LintHit] = []
    for m in re.finditer(r"(?im)^## Q\d+\.\s*(Why[^\n]*)\n([^\n]*)", md):
        q, a = m.group(1), m.group(2)
        if not REASON_CUE_RE.search(f"{q} {a} {reason_blob}"):
            hits.append(LintHit("why question without reason cue", q[:100], "error"))
    return hits


def lint_empty_headings(md: str) -> list[LintHit]:
    hits: list[LintHit] = []
    for m in re.finditer(r"(?m)^(#{1,3}\s+[^\n]+)\n(?:\s*\n)+(?=#{1,3}\s|\Z)", md):
        title = m.group(1).strip()
        if re.search(r"(?i)debug", title):
            hits.append(LintHit("empty or debug heading", title, "error"))
        elif re.match(r"(?i)^#{1,3}\s+Debug\b", title) or not title:
            hits.append(LintHit("empty heading", title or "(blank)", "error"))
        # Heading followed only by blank lines before next heading
        block = m.group(0)
        body = re.sub(r"^#{1,3}\s+[^\n]+\n?", "", block).strip()
        if not body:
            hits.append(LintHit("empty heading", title, "error"))
    return hits


def lint_verified_without_span(notes) -> list[LintHit]:
    """from_paper / verified provenance requires a proving span."""
    hits: list[LintHit] = []
    for row in getattr(notes, "status_table", None) or []:
        prov = getattr(row, "provenance", "") or ""
        if prov == "from_paper":
            span = ""
            for e in getattr(notes, "evidence", None) or []:
                if e.field == "component" and e.value == row.name and e.source_sentence:
                    span = e.source_sentence
                    break
            if not span and hasattr(row, "evidence_span"):
                span = row.evidence_span or ""
            if not span:
                hits.append(
                    LintHit("verified/from_paper without span", row.name, "error")
                )
    return hits


def lint_timeline_artifacts(md: str) -> list[LintHit]:
    """Flag ';.' / '.;' and bare ';' dumps inside Timeline rows only."""
    hits: list[LintHit] = []
    section = re.search(
        r"(?is)##\s*11\.\s*Timeline\b(.*?)(?=##\s*\d|\Z)", md
    ) or re.search(r"(?is)##\s*Timeline\b(.*?)(?=##|\Z)", md)
    if not section:
        return hits
    body = section.group(1)
    if re.search(r";\s*\.|\.\s*;", body):
        hits.append(LintHit("timeline semicolon artifact", "Timeline ;.", "error"))
    for line in body.splitlines():
        if not line.strip().startswith("|"):
            continue
        if re.match(r"^\|\s*-+", line):
            continue
        # Cell with only ';' or ending with ';.'
        if re.search(r"\|\s*;\s*\|", line) or re.search(r";\s*\.|\.\s*;", line):
            hits.append(LintHit("timeline cell artifact", line[:120], "error"))
    return hits


def lint_markdown(
    md: str,
    task_type: str = "unknown",
    contribution: str = "new_model",
    *,
    require_study_title: bool = False,
    source_text: str = "",
    problem: str = "",
    reason_blob: str = "",
    notes=None,
) -> list[LintHit]:
    hits: list[LintHit] = []
    rules = list(BANNED)
    if require_study_title or contribution in {"new_metric", "survey"}:
        rules.append(
            (r"(?i)A Practical Python Prototype", "use 'A Practical Python Study' for metric/survey"),
        )
    for pat, why in rules:
        m = re.search(pat, md)
        if m:
            hits.append(LintHit(why, _snippet(md, m.start(), m.end()), "error"))

    for pat, why in TASK_BANNED.get(task_type, []):
        # Skip BERT fine-tune ban unless metric contribution
        if "fine-tuned a small BERT" in pat and contribution != "new_metric":
            continue
        m = re.search(pat, md)
        if m:
            hits.append(LintHit(why, _snippet(md, m.start(), m.end()), "error"))

    for pat, why in CONTRIB_BANNED.get(contribution, []):
        m = re.search(pat, md)
        if m:
            hits.append(LintHit(why, _snippet(md, m.start(), m.end()), "error"))

    # Structural locks requested by product owner
    if problem or source_text:
        hits.extend(lint_problem_against_source(problem or "", source_text or md))
    hits.extend(lint_why_questions(md, reason_blob=reason_blob))
    hits.extend(lint_empty_headings(md))
    hits.extend(lint_timeline_artifacts(md))
    if notes is not None:
        hits.extend(lint_verified_without_span(notes))

    return hits


def lint_report(hits: list[LintHit]) -> str:
    if not hits:
        return ""
    lines = ["### Markdown linter", ""]
    for h in hits:
        lines.append(f"- **{h.severity}:** …{h.snippet}…")
    return "\n".join(lines)
