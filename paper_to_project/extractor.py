"""Structure-aware extraction of paper fields + task/contribution-templated scope."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .cleaner import clean_paper_text, extract_abstract, extract_section, guess_title
from .components import ComponentItem, components_blurb, extract_component_items
from .contribution import detect_contribution, score_contributions
from .dataset_match import DatasetMatch
from .datasets import extract_datasets_structured, strip_metadata_lines
from .evidence import FieldEvidence, attach_evidence, confidence_from_score, is_body_span, find_proving_span
from .llm_checker import CheckerReport, apply_safe_fixes, heuristic_self_check, llm_check_notes
from .plan_overlay import resolve_template
from .robustness import (
    assess_input,
    title_only_method,
    title_only_problem,
)
from .roles import build_role_records, timeline_from_steps
from .sentences import (
    clean_prose_sentence,
    derive_problem_from_contrast,
    is_noisy_sentence,
    is_result_sentence,
    sentences_for_role,
    strip_title_abstract_leak,
)
from .status import ComponentStatus, build_status_table, status_viva_questions
from .tables import TableFinding, TableReport, analyze_tables
from .verify import verify_notes
from .templates import (
    TEMPLATES,
    TaskScore,
    TaskTemplate,
    detect_paper_type,
    detect_task_type,
    get_template_by_type,
    top_task_candidates,
)


@dataclass
class PaperNotes:
    title: str = ""
    problem: str = ""
    method: str = ""
    dataset: str = ""
    metrics: str = ""
    components: list[str] = field(default_factory=list)
    component_items: list[ComponentItem] = field(default_factory=list)
    status_table: list[ComponentStatus] = field(default_factory=list)
    table_findings: list[TableFinding] = field(default_factory=list)
    table_rows_seen: int = 0
    dataset_match: DatasetMatch | None = None
    python_scope: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    suggested_stack: list[str] = field(default_factory=list)
    task_type: str = "unknown"
    task_label: str = "Unknown (please confirm)"
    task_candidates: list[TaskScore] = field(default_factory=list)
    contribution: str = "new_model"
    contribution_scores: list[dict[str, Any]] = field(default_factory=list)
    paper_type: str = "method"  # survey | method | application
    source_mode: str = "heuristic"
    viva_extras: list[tuple[str, str]] = field(default_factory=list)
    evidence: list[FieldEvidence] = field(default_factory=list)
    checker: CheckerReport | None = None
    warnings: list[str] = field(default_factory=list)
    unverified_components: list[str] = field(default_factory=list)
    dataset_suggested: str = ""
    metrics_suggested: str = ""
    missing_items: list[str] = field(default_factory=list)
    sections_found: list[str] = field(default_factory=list)
    runner_up_note: str = ""
    input_tier: str = "full"
    paper_kind: str = "ml_method"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["task_candidates"] = [
            {"task_type": c.task_type, "score": c.score, "normalized": c.normalized, "matched": c.matched}
            if isinstance(c, TaskScore)
            else c
            for c in (self.task_candidates or [])
        ]
        return d


METRIC_WHITELIST = [
    "mAP50-95", "mAP50", "mAP75", "mAP", "AP50", "AP75", "AP", "IoU",
    "precision", "recall", "F1-score", "F1", "accuracy", "RMSE", "MAE", "MSE",
    "AUC", "ROC-AUC", "BLEU", "METEOR", "ROUGE-L", "ROUGE", "perplexity", "FPS",
    "Pearson", "Spearman", "Kendall", "Kendall tau", "Kendall τ", "Pearson r", "Spearman ρ",
    "Hits@1", "Hits@10", "Hits@K",
    "MOTA", "MOTP", "IDF1", "HOTA", "IDS", "NDCG", "MRR",
    "top-1", "top-5", "top1", "top5", "logloss",
    "R@10", "R@5", "P@10", "P@5", "BERTScore",
]

PROBLEM_CUES = re.compile(
    r"(?i)\b(however|nevertheless|unfortunately|challenge|limitation|struggle|"
    r"difficult|poor|lack|fail|issue|problem|to address|motivated by|"
    r"not (?:yet )?fully understood|degradation problem|open (?:problem|question))\b"
)

METHOD_CUES = re.compile(
    r"(?i)\b(we (propose|present|introduce|develop|design|compute|use)|proposed method|"
    r"our (method|approach|model|framework|metric)|"
    r"this paper (proposes|presents|introduces)|residual learning framework)\b"
)

INTRO_NOISE = re.compile(
    r"(?i)^(introduction\b|\d+\.?\s*introduction\b|in recent years\b|with the development\b)"
)

_CONNECTOR = re.compile(r"(?i)^(however|finally|instead|moreover|furthermore|also|thus|therefore)\b")


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 25]


def _clip(text: str, limit: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"




def extract_problem(text: str, abstract: str, title: str = "") -> tuple[str, set[str]]:
    """Problem-role only; reject result verbs / equation junk; prefer abstract prose."""
    # Prefer abstract; fall back to front matter only (not full PDF body)
    pool = strip_metadata_lines(abstract or text[:2500])
    role_sents = sentences_for_role(pool, "problem", limit=3, title=title)
    role_sents = [
        s for s in role_sents
        if not re.search(r"(?i)\bwe (?:propose|compute|use|introduce|present)\b", s)
        and not is_result_sentence(s)
        and not is_noisy_sentence(s)
    ]

    used: set[str] = set()
    if role_sents:
        # One clean problem sentence — joining body scraps causes equation bleed
        s = role_sents[0]
        used.add(re.sub(r"\s+", " ", s.lower()))
        return _clip(s), used

    derived = derive_problem_from_contrast(pool, title=title)
    if derived and not is_result_sentence(derived) and not is_noisy_sentence(derived):
        used.add(re.sub(r"\s+", " ", derived.lower()))
        return _clip(derived), used

    return "Problem not stated in the provided text.", used


def extract_method(
    text: str,
    abstract: str,
    paper_type: str,
    title: str = "",
    used: set[str] | None = None,
) -> tuple[str, set[str]]:
    abs_clean = strip_metadata_lines(abstract or "")
    used = set(used or ())
    if paper_type == "survey":
        topic = title or "this topic"
        topic = re.sub(r"(?i)\s*:\s*a survey.*$", "", topic).strip() or "this topic"
        if re.search(r"(?i)residual|resnet", topic + " " + abs_clean):
            return "Survey of residual networks: why they work, variants, and open questions.", used
        return f"Survey of {topic}: synthesizes prior work; no new model proposed.", used
    if paper_type == "theory":
        bits = sentences_for_role(abs_clean, "method", limit=2, title=title)
        bits = [s for s in bits if not is_noisy_sentence(s)]
        if bits:
            return _clip(" ".join(bits[:2])), used
        return (
            "Theoretical / analysis paper: no implementable model stated in the provided text.",
            used,
        )

    role_sents = sentences_for_role(abs_clean, "method", limit=5, title=title)
    bits: list[str] = []
    for s in role_sents:
        s = clean_prose_sentence(s, title)
        if INTRO_NOISE.match(s) or is_noisy_sentence(s):
            continue
        if re.search(
            r"(?i)\b(outperform|correlates better|achieves|more robust|to show that|"
            r"we evaluate|we compare)\b",
            s,
        ) and not re.search(r"(?i)\bwe (?:propose|compute|introduce|use)\b", s):
            continue
        if re.search(r"(?i)\b(correlates better|outperform|more robust)\b", s):
            continue
        key = re.sub(r"\s+", " ", s.lower())
        if key in used:
            continue
        bits.append(s)
        used.add(key)

    # Only mine method section if abstract gave nothing usable
    if len(bits) < 2:
        method_sec = extract_section(
            text,
            start_patterns=[
                r"(?m)^\s*\d+(\.\d+)*\.?\s+(Proposed(?:\s+Method)?|Methodology|Methods?|Approach|Model|Architecture)\b",
                r"(?m)^\s*(Proposed Method|Methodology|Methods|Our Approach|Deep Residual Learning)\b",
            ],
            end_patterns=[
                r"(?m)^\s*\d+(\.\d+)*\.?\s+(Experiment|Evaluation|Result|Dataset|Conclusion|Discussion)\b",
                r"(?m)^\s*(Experiments?|Results?|Conclusion)\b",
            ],
        )
        if method_sec:
            for s in _sentences(method_sec[:1800])[:8]:
                s = clean_prose_sentence(s, title)
                if INTRO_NOISE.match(s) or re.search(r"(?i)\bintroduction\b", s):
                    continue
                if is_noisy_sentence(s):
                    continue
                key = re.sub(r"\s+", " ", s.lower())
                if key in used:
                    continue
                if METHOD_CUES.search(s):
                    bits.append(s)
                    used.add(key)
                    if len(bits) >= 3:
                        break
    if not bits:
        return "Not stated in the provided text.", used
    return _clip(" ".join(bits[:3])), used


def _evaluation_section(text: str) -> str:
    return extract_section(
        text,
        start_patterns=[
            r"(?m)^\s*\d+\.\d+\.?\s+Evaluation(?:\s+Index|\s+Metrics?|\s+Measures?)?\b",
            r"(?m)^\s*\d+(\.\d+)*\.?\s+(Evaluation|Metrics?|Performance Metrics|Experiments)\b",
            r"(?m)^\s*(Evaluation Index|Evaluation Metrics|Performance Metrics)\b",
            r"(?m)^\s*\d+(\.\d+)*\.?\s+(Experimental Results|Results)\b",
        ],
        end_patterns=[
            r"(?m)^\s*\d+(\.\d+)*\.?\s+(Discussion|Conclusion|Ablation|Related Work|References)\b",
            r"(?m)^\s*(Conclusion|References|Discussion|Ablation Study)\b",
        ],
    )


def _metric_value_ok(label: str, raw: str, ctx: str = "") -> bool:
    """Reject equation scraps and IoU thresholds mistaken for results."""
    lab = label.lower().strip()
    # IoU=0.50 is usually the mAP50 threshold, not a reported score
    if lab == "iou":
        return False
    # IDS is a count; keep name-only unless clearly a count with context
    if lab == "ids" and "=" in (raw or ""):
        return False
    if re.search(r"(?i)equation|loss\s*=|lnwd|nwd\s*\(|µ[ab]|distance measure", ctx):
        return False
    m = re.search(r"(\d+(?:\.\d+)?)\s*(%?)", raw or "")
    if not m:
        return True
    val = float(m.group(1))
    pct = bool(m.group(2)) or "%" in (raw or "")
    # Tracking / detection rates are rarely under ~5% when written as percent
    if lab in {"mota", "motp", "idf1", "hota", "map50", "map", "precision", "recall", "accuracy"}:
        if pct and val < 5:
            return False
        if not pct and 0 < val < 0.05:
            return False
    return True


def _whitelist_hits(hay: str) -> list[str]:
    if not hay:
        return []
    names = sorted(METRIC_WHITELIST, key=len, reverse=True)
    found: list[str] = []
    seen: set[str] = set()
    value_pat = re.compile(
        # Allow "mAP) of 62.8%" after matching inside parentheses
        r"^\s*\)?\s*(?:=|:|of|reaches|achieves|is|error)?\s*(\d+(?:\.\d+)?%?)",
        re.IGNORECASE,
    )

    def mark_seen(key: str) -> None:
        seen.add(key)
        if key in {"top-1", "top1"}:
            seen.update({"top-1", "top1"})
        if key in {"top-5", "top5"}:
            seen.update({"top-5", "top5"})

    for name in names:
        key = name.lower()
        if key in seen:
            continue
        if key in {"top-1", "top-5", "top1", "top5"}:
            dig = key[-1]
            pat = re.compile(rf"(?i)\btop[- ]?{dig}\b(?:\s*error)?")
            label = f"top-{dig} error"
        else:
            pat = re.compile(rf"(?i)(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])")
            label = name

        best = None
        matched = False
        for m in pat.finditer(hay):
            matched = True
            ctx = hay[max(0, m.start() - 40) : m.end() + 48]
            window = hay[m.end() : m.end() + 48]
            num = value_pat.match(window)
            if num and not re.fullmatch(r"[1-9]", num.group(1)):
                cand = f"{label} = {num.group(1)}"
                if _metric_value_ok(label, num.group(1), ctx):
                    best = cand
                    break
                continue
            pre = hay[max(0, m.start() - 24) : m.start()]
            pre_n = re.search(r"(\d+(?:\.\d+)?%?)\s*$", pre)
            if pre_n and _metric_value_ok(label, pre_n.group(1), ctx):
                best = f"{label} = {pre_n.group(1)}"
                break
        if matched:
            mark_seen(key)
            if best:
                found.append(best)
            elif _metric_value_ok(label, "", ""):
                found.append(label)

    # Correlation phrasing without named coeff
    if re.search(r"(?i)correlat\w*\s+with\s+human", hay) and not any(
        "pearson" in x.lower() or "spearman" in x.lower() or "kendall" in x.lower() for x in found
    ):
        found.append("correlation with human judgments")

    iou50 = re.search(
        r"(?i)\bmAP\b(?:\s*(?:=|of|:)\s*|\s+)(\d+(?:\.\d+)?%?).{0,40}?IoU\s*=?\s*0?\.?50",
        hay,
    )
    if iou50 and _metric_value_ok("mAP50", iou50.group(1)):
        found = [f"mAP50 = {iou50.group(1)}"] + [x for x in found if not x.lower().startswith("map50")]

    valued = [x for x in found if "=" in x]
    named = [x for x in found if "=" not in x]
    return valued + named



def _prioritize_metrics(
    hits: list[str],
    task_type: str = "",
    contribution: str = "new_model",
) -> list[str]:
    """Prefer valued metrics; drop weak name-only spam for detection papers."""
    weak_name_only = {
        "accuracy", "precision", "recall", "ap", "ids", "f1", "f1-score",
    }
    valued = [h for h in hits if "=" in h]
    named = [h for h in hits if "=" not in h]
    # Drop AP if mAP/mAP50 already present; drop weak bare names on vision/MOT
    has_map = any(re.match(r"(?i)mAP", h.split("=")[0].strip()) for h in valued + named)
    cleaned_named = []
    for h in named:
        key = h.split("=")[0].strip().lower()
        if key in weak_name_only and (
            task_type == "object_detection_tracking" or contribution == "new_model"
        ):
            continue
        if key == "ap" and has_map:
            continue
        cleaned_named.append(h)
    # Keep strong tracking metric names even without values
    strong = {"mota", "motp", "idf1", "hota", "map50", "map", "fps", "bleu", "meteor", "pearson", "kendall"}
    cleaned_named = [
        h for h in cleaned_named
        if h.split("=")[0].strip().lower() in strong
        or contribution == "new_metric"
    ] if task_type == "object_detection_tracking" else cleaned_named
    # Dedup by metric stem
    out: list[str] = []
    seen: set[str] = set()
    for h in valued + cleaned_named:
        stem = h.split("=")[0].strip().lower()
        if stem in seen:
            continue
        seen.add(stem)
        out.append(h)
    return out


def extract_metrics_structured(
    text: str,
    abstract: str,
    template: TaskTemplate,
    contribution: str = "new_model",
    proposed_metric: str = "",
) -> tuple[str, str]:
    """Return (from_paper, suggested). Suggested holds expanded coefficient names not in text."""
    eval_sec = _evaluation_section(text)
    # Prefer abstract + evaluation prose; avoid equation-heavy body for values
    combined = f"{abstract}\n{eval_sec}"
    hits = _whitelist_hits(combined)
    if not hits:
        hits = _whitelist_hits(text[:12000])
    # Prefer valued hits from abstract when present (more reliable than PDF tables)
    abs_hits = _whitelist_hits(abstract or "")
    if abs_hits:
        abs_valued = [h for h in abs_hits if "=" in h]
        if abs_valued:
            rest = [h for h in hits if h.split("=")[0].strip().lower() not in {
                v.split("=")[0].strip().lower() for v in abs_valued
            }]
            hits = abs_valued + rest

    from_paper_hits: list[str] = []
    suggested: list[str] = []
    body = abstract + "\n" + text[:12000]

    for h in hits:
        # Only keep if the metric name literally appears in text
        name = h.split("=")[0].strip()
        if re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(name.split()[0])}(?![A-Za-z0-9])", body):
            from_paper_hits.append(h)
        else:
            suggested.append(h)

    # Correlation phrase without naming Pearson etc.
    if re.search(r"(?i)correlat\w*\s+with\s+human", body):
        if not any("correlation" in x.lower() for x in from_paper_hits):
            from_paper_hits.append("correlation with human judgments")
        for coeff in ("Pearson", "Spearman", "Kendall"):
            if not re.search(rf"(?i)\b{coeff}\b", body):
                if coeff not in suggested:
                    suggested.append(coeff)

    from_paper_hits = _prioritize_metrics(
        from_paper_hits, task_type=getattr(template, "task_type", ""), contribution=contribution
    )

    parts: list[str] = []
    if proposed_metric and contribution == "new_metric":
        if re.search(rf"(?i)\b{re.escape(proposed_metric)}\b", body):
            parts.append(f"Metric proposed: {proposed_metric}.")
        eval_hits = [h for h in from_paper_hits if proposed_metric.lower() not in h.lower()]
        if eval_hits:
            parts.append("Metrics used to evaluate it: " + ", ".join(eval_hits[:10]) + ".")
        elif re.search(r"(?i)correlat", abstract):
            parts.append("Metrics used to evaluate it: correlation with human judgments.")
    elif from_paper_hits:
        parts.append(", ".join(from_paper_hits[:10]))
    else:
        parts.append("Not found in the provided text.")

    sug = ", ".join(suggested[:8]) if suggested else ""
    return _clip(" ".join(parts), limit=1200), sug


def extract_metrics(
    text: str,
    abstract: str,
    template: TaskTemplate,
    contribution: str = "new_model",
    proposed_metric: str = "",
) -> str:
    fp, _ = extract_metrics_structured(text, abstract, template, contribution, proposed_metric)
    return fp


def _detect_sections(text: str) -> list[str]:
    found = []
    checks = [
        ("abstract", r"(?im)^\s*Abstract\b"),
        ("introduction", r"(?im)^\s*(?:\d+\.?\s*)?Introduction\b"),
        ("method", r"(?im)^\s*(?:\d+\.?\s*)?(?:Proposed Method|Methodology|Methods|Approach)\b"),
        ("experiments", r"(?im)^\s*(?:\d+\.?\s*)?(?:Experiments?|Experimental Setup|Evaluation)\b"),
        ("results", r"(?im)^\s*(?:\d+\.?\s*)?Results\b"),
        ("dataset", r"(?im)^\s*(?:\d+\.?\s*)?(?:Dataset|Data Set)\b"),
    ]
    for name, pat in checks:
        if re.search(pat, text):
            found.append(name)
    return found


def apply_template(
    notes: PaperNotes,
    template: TaskTemplate,
    components: list[str],
    table_report: TableReport | None = None,
    paper_text: str = "",
    component_items: list[ComponentItem] | None = None,
) -> PaperNotes:
    notes.task_type = template.task_type
    notes.task_label = template.label
    notes.suggested_stack = list(template.stack)
    notes.python_scope = list(template.python_scope)
    notes.out_of_scope = list(template.out_of_scope)
    notes.components = components
    if component_items is not None:
        notes.component_items = component_items
    notes.status_table = build_status_table(
        component_items or components,
        task_type=template.task_type,
        paper_text=paper_text,
        contribution=notes.contribution,
    )

    if table_report is not None:
        notes.table_findings = list(table_report.findings)
        notes.table_rows_seen = table_report.rows_seen

    # Plan steps only for verified components (unverified filtered later in verify_notes)
    viva: list[tuple[str, str]] = []
    viva.extend(status_viva_questions(notes.status_table))

    if notes.task_candidates and len(notes.task_candidates) > 1:
        c1, c2 = notes.task_candidates[0], notes.task_candidates[1]
        if c2.score > 0 and (c1.score - c2.score) <= 3:
            viva.append(
                (
                    f"Task scores are close ({c1.task_type}={c1.score}, {c2.task_type}={c2.score}). Which protocol fits the paper?",
                    f"Scores only: {c1.task_type}={c1.score}, {c2.task_type}={c2.score}.",
                )
            )

    # Ablation viva only for plausible module names (skip OCR junk like "Vehicle +8")
    for f in notes.table_findings:
        if f.kind != "ablation":
            continue
        if re.search(
            r"(?i)Ablation:\s*(C2f_|Detect_|MSN|NWD|MLCA|SEAM|YOLO|ByteTrack|Re-?ID)",
            f.summary or "",
        ):
            viva.append(("What ablation gain does the paper report?", f.summary))
            break

    for item in (component_items or [])[:3]:
        row = next((r for r in notes.status_table if r.name == item.name), None)
        if not row or row.provenance == "unverified":
            continue
        why = row.reason
        if item.role == "proposed":
            viva.append((f"What does the paper propose as {item.name}?", why))
        elif item.role == "backbone":
            viva.append((f"What does the paper say about how {item.name} is used?", why))
        elif re.search(r"(?i)greedy|matching|idf|rescal|similarity", item.name):
            viva.append((f"What operation is {item.name} in this method?", why))

    notes.viva_extras = viva
    return notes


def _build_evidence(notes: PaperNotes, text: str) -> list[FieldEvidence]:
    title = notes.title or ""
    ev = [
        attach_evidence("title", notes.title, text, title=title),
        attach_evidence("problem", notes.problem, text, title=title),
        attach_evidence("method", notes.method, text, title=title),
        attach_evidence("dataset", notes.dataset, text, title=title),
        attach_evidence("metrics", notes.metrics, text, title=title),
    ]
    if notes.dataset_suggested:
        ev.append(
            attach_evidence(
                "dataset_suggested",
                notes.dataset_suggested,
                text,
                title=title,
                suggested=True,
                note="Suggested, not stated",
            )
        )
    if notes.metrics_suggested:
        ev.append(
            attach_evidence(
                "metrics_suggested",
                notes.metrics_suggested,
                text,
                title=title,
                suggested=True,
                note="Suggested, not stated",
            )
        )
    cs = notes.contribution_scores or []
    top_c = cs[0]["score"] if cs else 0
    run_c = cs[1]["score"] if len(cs) > 1 else 0
    # Contribution label: from_paper only if cue phrases appear in text
    cue_span = ""
    for cue in ("evaluation metric", "human judgments", "BERTScore", "we propose", "this survey"):
        if cue.lower() in text.lower():
            cue_span = find_proving_span(text, [cue], title=title)
            if cue_span:
                break
    if cue_span:
        contrib_prov, contrib_conf = "from_paper", confidence_from_score(int(top_c), int(run_c))
    elif top_c >= 4:
        contrib_prov, contrib_conf = "suggested", confidence_from_score(int(top_c), int(run_c))
    else:
        contrib_prov, contrib_conf = "not_found", "low"
    ev.append(
        FieldEvidence(
            field="contribution",
            value=notes.contribution,
            source_sentence=cue_span,
            confidence=contrib_conf,
            note=f"score={top_c}; runner-up={cs[1]['contribution'] if len(cs)>1 else 'none'}={run_c}",
            provenance=contrib_prov,
        )
    )
    for c in notes.components[:8]:
        ev.append(attach_evidence("component", c, text, title=title, key_terms=[c]))
    if notes.task_candidates:
        top = notes.task_candidates[0]
        runner = notes.task_candidates[1] if len(notes.task_candidates) > 1 else None
        run_score = runner.score if runner else 0
        ev.append(
            FieldEvidence(
                field="task_type",
                value=notes.task_type,
                source_sentence=", ".join(top.matched[:4]) or "",
                confidence=confidence_from_score(top.score, run_score),
                note=f"score={top.score}; {notes.runner_up_note}",
                provenance="inferred",  # detector label; confidence from score margin
            )
        )
    return ev


def extract_heuristic(
    text: str,
    dataset_match: DatasetMatch | None = None,
    run_checker: bool = True,
) -> PaperNotes:
    text = clean_paper_text(text)
    text = strip_metadata_lines(text)
    abstract = extract_abstract(text)
    title = guess_title(text)
    profile = assess_input(text, title=title, abstract=abstract)

    paper_type = detect_paper_type(f"{title}\n{abstract}\n{text[:4000]}")
    if profile.paper_kind == "survey":
        paper_type = "survey"
    elif profile.paper_kind == "theory":
        paper_type = "theory"
    elif profile.paper_kind in {"non_ml", "hardware", "math"} and paper_type == "method":
        # Keep method unless survey/theory already won; paper_kind drives the plan
        pass

    blob = f"{title}\n{abstract}\n{text[:6000]}"
    candidates = top_task_candidates(blob, k=2)
    task_type = candidates[0].task_type if candidates else detect_task_type(blob)

    # Non-ML / theory / hardware: do not force a vision/NLP template
    if profile.paper_kind in {"theory", "hardware", "math", "non_ml"}:
        task_type = "unknown"
        candidates = [TaskScore("unknown", 0, [])]
    elif profile.paper_kind == "unknown" and (
        not candidates or candidates[0].score < 4
    ):
        task_type = "unknown"
        candidates = [TaskScore("unknown", 0, [])]

    runner_up_note = "no second candidate above threshold"
    if len(candidates) > 1 and candidates[1].score > 0:
        runner_up_note = f"runner-up {candidates[1].task_type}={candidates[1].score}"
    display_cands = [candidates[0]] if candidates else [TaskScore("unknown", 0, [])]
    if len(candidates) > 1 and candidates[1].score > 0:
        display_cands.append(candidates[1])

    contrib_scores = score_contributions(blob)
    contribution = detect_contribution(blob)
    if paper_type == "survey" or profile.paper_kind == "survey":
        contribution = "survey"
    elif profile.paper_kind in {"theory", "non_ml"} and contribution == "new_model":
        contribution = "survey"  # study-style deliverable

    items = extract_component_items(
        f"{abstract}\n{text[:12000]}", task_type=task_type, contribution=contribution
    )
    # Sparse / metadata: do not invent components from title tokens alone
    if profile.input_tier == "metadata_only":
        items = []
    role_records = build_role_records(
        items,
        f"{abstract}\n{text[:12000]}",
        contribution,
        title=title,
        task_type=task_type,
    )
    components = [r.name for r in role_records if r.provenance != "unverified"]
    unverified = [r.name for r in role_records if r.provenance == "unverified"]
    proposed = next((c.name for c in items if c.role == "proposed"), "")

    template = resolve_template(
        task_type,
        contribution,
        components,
        paper_text=text,
        role_records=role_records,
        paper_kind=profile.paper_kind,
        title=title,
    )

    if profile.input_tier == "metadata_only":
        problem, used = title_only_problem(title), set()
        method, used = title_only_method(title), used
    else:
        problem, used = extract_problem(text, abstract, title=title)
        method, used = extract_method(text, abstract, paper_type, title=title, used=used)
        if profile.paper_kind == "theory" and (
            not method or method.startswith("Not stated")
        ):
            method = (
                "Theoretical / analysis paper: no implementable model stated in the provided text. "
                "Restate claims from the abstract for the study report."
            )

    table_report = analyze_tables(text) if profile.input_tier not in {"metadata_only"} else None
    sections = _detect_sections(text)

    ds = extract_datasets_structured(text, contribution=contribution)
    metrics_fp, metrics_sug = extract_metrics_structured(
        text, abstract, template, contribution, proposed_metric=proposed
    )
    if profile.input_tier == "metadata_only":
        ds_from = "Not stated in the provided text."
        metrics_fp = "Not found in the provided text."
        metrics_sug = ""
    else:
        ds_from = ds.from_paper

    warnings: list[str] = list(profile.warnings)
    missing: list[str] = list(profile.missing_hints)
    if len(display_cands) >= 2 and (display_cands[0].score - display_cands[1].score) <= 2:
        warnings.append(
            f"Close task scores: {display_cands[0].task_type}={display_cands[0].score} vs "
            f"{display_cands[1].task_type}={display_cands[1].score}."
        )

    body_sections = [s for s in sections if s not in {"abstract"}]
    thin_by_sections = not body_sections or body_sections == ["introduction"]
    if "experiments" not in sections and "results" not in sections and "dataset" not in sections:
        if "experiments/results/dataset sections" not in missing:
            missing.append("experiments/results/dataset sections")
    if ds_from.startswith("Not stated") and "named datasets" not in missing:
        missing.append("named datasets")
    if metrics_fp.startswith("Not found") and "named evaluation metrics" not in missing:
        missing.append("named evaluation metrics")
    if problem.startswith("Problem not stated") and "problem statement" not in missing:
        missing.append("problem statement")

    # Always remind: do not claim paper numbers
    claim_note = (
        "Do not claim the paper’s reported numbers unless you match the same data and protocol."
    )
    if claim_note not in warnings:
        warnings.append(claim_note)
    # Deduplicate warnings / missing
    warnings = list(dict.fromkeys(warnings))
    missing = list(dict.fromkeys(missing))

    notes = PaperNotes(
        title=title,
        problem=problem,
        method=method,
        dataset=ds_from,
        dataset_suggested=ds.suggested if profile.input_tier != "metadata_only" else "",
        metrics=metrics_fp,
        metrics_suggested=metrics_sug,
        components=components,
        component_items=items,
        unverified_components=unverified,
        paper_type=paper_type,
        contribution=contribution,
        contribution_scores=[
            {"contribution": s.contribution, "score": s.score, "matched": s.matched[:4]}
            for s in contrib_scores[:3]
        ],
        task_candidates=display_cands,
        runner_up_note=runner_up_note,
        dataset_match=dataset_match,
        source_mode="heuristic",
        warnings=warnings,
        missing_items=missing,
        sections_found=sections,
        input_tier=profile.input_tier,
        paper_kind=profile.paper_kind,
    )
    notes = apply_template(
        notes, template, components, table_report, paper_text=text, component_items=items
    )
    notes.evidence = _build_evidence(notes, text)

    # Refine abstract-only warning using evidence body spans
    has_body_ev = any(
        e.source_sentence and is_body_span(e.source_sentence, abstract)
        for e in notes.evidence
        if e.provenance == "from_paper"
    )
    if thin_by_sections and not has_body_ev and "experiments" not in sections:
        notes.warnings.append(
            "Input appears limited to front matter; experiments/results sections were not found."
        )
    # Never say abstract-only if we have body evidence
    notes.warnings = [
        w for w in notes.warnings
        if not (has_body_ev and "abstract-only" in w.lower())
    ]

    # Final verification pass
    verify_notes(notes, text, title=title)

    # Rebuild plan from remaining verified role records (keep paper_kind fallbacks)
    verified_recs = [r for r in role_records if r.name in set(notes.components)]
    template2 = resolve_template(
        notes.task_type,
        notes.contribution,
        notes.components,
        paper_text=text,
        role_records=verified_recs or None,
        paper_kind=notes.paper_kind,
        title=title,
    )
    notes.python_scope = list(template2.python_scope)
    notes.suggested_stack = list(template2.stack)
    notes.out_of_scope = list(template2.out_of_scope)
    notes.task_label = template2.label
    from .status import build_status_table as _bst
    # Rebuild role records after verify so status stays task-aware
    if verified_recs:
        verified_recs = build_role_records(
            [ComponentItem(r.name, r.role) for r in verified_recs],
            f"{abstract}\n{text[:12000]}",
            contribution,
            title=title,
            task_type=task_type,
        )
        verified_recs = [r for r in verified_recs if r.name in set(notes.components)]
    notes.status_table = _bst(
        notes.components, task_type=task_type, paper_text=text,
        contribution=contribution, role_records=verified_recs if verified_recs else None,
        title=title,
    )

    if run_checker:
        report = heuristic_self_check(notes.to_dict(), text)
        notes.checker = report
        if report.adjusted:
            fixed = apply_safe_fixes(notes.to_dict(), report)
            if fixed.get("components") != notes.components:
                notes.components = list(fixed.get("components") or notes.components)
                notes.status_table = build_status_table(
                    notes.components,
                    task_type=notes.task_type,
                    paper_text=text,
                    contribution=notes.contribution,
                )
    return notes


def _llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def extract_with_llm(
    text: str,
    dataset_match: DatasetMatch | None = None,
) -> PaperNotes:
    """Heuristic first; LLM only verifies and applies safe fixes (never invents from scratch)."""
    notes = extract_heuristic(text, dataset_match=dataset_match, run_checker=True)
    report = llm_check_notes(notes.to_dict(), text)
    notes.checker = report
    fixed = apply_safe_fixes(notes.to_dict(), report)
    if fixed.get("task_type") and fixed["task_type"] != notes.task_type:
        notes.task_type = fixed["task_type"]
        template = resolve_template(notes.task_type, notes.contribution)
        notes = apply_template(
            notes,
            template,
            list(fixed.get("components") or notes.components),
            TableReport(notes.table_findings, notes.table_rows_seen),
            paper_text=text,
            component_items=notes.component_items,
        )
    elif fixed.get("components") is not None:
        notes.components = list(fixed["components"])
        notes.status_table = build_status_table(
            notes.components,
            task_type=notes.task_type,
            paper_text=text,
            contribution=notes.contribution,
        )
    notes.evidence = _build_evidence(notes, text)
    notes.source_mode = "heuristic+llm_checker"
    notes.task_candidates = notes.task_candidates or top_task_candidates(text, k=2)
    return notes


def extract_paper_notes(
    text: str,
    prefer_llm: bool = True,
    dataset_match: DatasetMatch | None = None,
) -> PaperNotes:
    text = (text or "").strip()
    if not text:
        raise ValueError("No text provided. Paste a paper, upload a PDF, or search by title.")
    if prefer_llm and _llm_available():
        try:
            return extract_with_llm(text, dataset_match=dataset_match)
        except Exception:
            notes = extract_heuristic(text, dataset_match=dataset_match)
            notes.source_mode = "heuristic (llm checker failed)"
            return notes
    return extract_heuristic(text, dataset_match=dataset_match)
