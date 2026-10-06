"""Student-facing proposal and viva — print each fact once; no developer wording."""

from __future__ import annotations

import re
from datetime import date

from .components import components_blurb
from .datasets import suggest_substitute
from .extractor import PaperNotes
from .roles import limits_from_plan, objectives_from_steps, timeline_from_steps
from .status import status_markdown
from .verify import lint_student_output


REASON_CUE = re.compile(r"(?i)\b(because|so that|in order to|since|as a result)\b")


def _use_study_title(notes: PaperNotes) -> bool:
    return (
        notes.paper_type in {"survey", "theory"}
        or notes.contribution in {"new_metric", "survey", "new_dataset_benchmark"}
        or getattr(notes, "paper_kind", "") in {"theory", "hardware", "math", "non_ml", "survey"}
    )


def _project_title_line(notes: PaperNotes) -> str:
    title = notes.title or "Mini Project from Research Paper"
    kind = "Study" if _use_study_title(notes) else "Prototype"
    return f"{title}: A Practical Python {kind}"


def _routing_once(notes: PaperNotes) -> str:
    if not notes.task_candidates:
        return f"{notes.task_label}; contribution `{notes.contribution}`"
    top = notes.task_candidates[0]
    line = f"{notes.task_label} (score {top.score}); contribution `{notes.contribution}`"
    if len(notes.task_candidates) > 1 and notes.task_candidates[1].score > 0:
        r = notes.task_candidates[1]
        line += f"; also considered {r.task_type}={r.score}"
    return line


def _component_blurb(name: str, notes: PaperNotes) -> str:
    """Short, component-specific answer — never dump the full method for a tracker Q."""
    row = next(
        (r for r in (notes.status_table or []) if r.name.lower() == name.lower()),
        None,
    )
    if re.search(r"(?i)^bytetrack$", name):
        return (
            "ByteTrack associates detection boxes across frames (including low-score boxes) "
            "so vehicle IDs stay continuous for multi-object tracking."
        )
    if re.search(r"(?i)^yolo", name):
        return (
            f"{name} is the detector backbone: it finds vehicles with bounding boxes; "
            "tracking is handled separately (e.g. ByteTrack)."
        )
    if re.search(r"(?i)^re-?id$", name):
        return (
            "Re-ID provides appearance features so the tracker can reconnect the same "
            "vehicle after occlusion or brief misses."
        )
    if re.search(r"(?i)msn-?yolo", name):
        return (
            "MSN-YOLO is the paper’s improved detector (YOLOv8n + modules such as "
            "C2f_MLCA, Detect_SEAM, and NWD loss)."
        )
    if row:
        return f"{name}: {row.reason} Student status: {row.status}."
    return f"{name} is named in the paper method."


def _primary_component(notes: PaperNotes) -> str:
    """Prefer tracker/backbone over first lexicon hit for the component viva Q."""
    names = list(notes.components or [])
    for pref in ("ByteTrack", "YOLOv8n", "YOLOv8", "MSN-YOLO"):
        for n in names:
            if n.lower() == pref.lower():
                return n
    return names[0] if names else ""


def _paper_qa_question(notes: PaperNotes) -> tuple[str, str]:
    """Ask 'why' only when a reason cue exists; otherwise ask 'what'."""
    blob = f"{notes.problem} {notes.method}"
    if notes.contribution == "new_metric" and REASON_CUE.search(blob):
        return (
            "Why does the paper prefer its approach over exact matches?",
            notes.method or notes.problem,
        )
    if re.search(r"(?i)contextual embeddings|exact matches", blob):
        return (
            "What does the paper use instead of exact matches?",
            notes.method or notes.problem,
        )
    comp = _primary_component(notes)
    if comp:
        return (f"What is {comp} in this paper?", _component_blurb(comp, notes))
    return ("What is the core idea of the method?", notes.method or "Not stated in the provided text.")


def build_proposal(notes: PaperNotes, student_name: str = "Student") -> str:
    title = notes.title or "Mini Project from Research Paper"
    stack = ", ".join(notes.suggested_stack) if notes.suggested_stack else "Python 3"
    scope = [re.sub(r"^\[Plan\]\s*", "", s).strip() for s in notes.python_scope]
    scope = [re.sub(r"\s*;\s*\.", ".", s) for s in scope]
    scope_lines = "\n".join(f"- {s}" for s in scope) or "- Thin demo from named pieces."
    out_lines = "\n".join(
        f"- {re.sub(r'^\[Plan\]\s*', '', s)}" for s in notes.out_of_scope
    ) or "- Full paper-scale replication."
    comps = components_blurb(notes.components) if notes.components else "—"

    findings_md = "None flagged."
    if notes.table_findings:
        findings_md = "\n".join(f"- **{f.kind}:** {f.summary}" for f in notes.table_findings)

    substitute = suggest_substitute(notes.dataset, notes.task_type, notes.contribution)
    timeline = timeline_from_steps(scope, task_type=notes.task_type)
    objectives = objectives_from_steps(scope, notes.task_type, notes.contribution)

    # Optional blocks — hide when empty (no empty headings)
    front_notes = []
    if notes.warnings:
        front_notes.extend(notes.warnings)
    if notes.missing_items:
        front_notes.append("Still missing from the text: " + "; ".join(notes.missing_items))
    front_block = ""
    if front_notes:
        front_block = "## Input notes\n" + "\n".join(f"- {w}" for w in front_notes) + "\n\n"

    sug_ds = f"\nRelated (cue-based): {notes.dataset_suggested}\n" if notes.dataset_suggested else ""
    sug_met = f"\nRelated (not in text): {notes.metrics_suggested}\n" if notes.metrics_suggested else ""

    # Inferred names (plain words) — only if non-empty
    inferred = [
        r.name for r in (notes.status_table or [])
        if getattr(r, "provenance", "") == "inferred"
    ]
    inferred_block = ""
    if inferred:
        label = (
            "pretrained detectors/backbones"
            if notes.task_type in {
                "object_detection_tracking",
                "image_classification",
                "semantic_segmentation",
            }
            else "pretrained encoders"
        )
        inferred_block = (
            f"\nAlso treating as {label} (not fully described in the text): "
            f"{', '.join(inferred)}\n"
        )
    if notes.unverified_components:
        inferred_block += f"\nNames without a clear sentence: {', '.join(notes.unverified_components)}\n"

    md = f"""# Mini Project Proposal

**Student:** {student_name}  
**Date:** {date.today().isoformat()}  
**Based on:** {title}  
**Paper type:** {notes.paper_type}  
**Focus:** {_routing_once(notes)}

{front_block}## 1. Project Title
{_project_title_line(notes)}

## 2. Problem Statement
{notes.problem}

## 3. Objectives
{objectives}

## 4. Methodology
### Paper method
{notes.method}

**Components:** {comps}
{inferred_block}
### Component status
{status_markdown(notes.status_table)}

### Implementation plan
{scope_lines}

## 5. Dataset
{notes.dataset}
{sug_ds}
**Mini-project data:** {substitute}

## 6. Evaluation Metrics
{notes.metrics}
{sug_met}
## 7. Table checks
{findings_md}

## 8. Technology Stack
{stack}

## 9. Out of Scope
{out_lines}

## 10. Expected Deliverables
- Working Python code and optional Streamlit demo
- README with setup steps
- Short report

## 11. Timeline
{timeline}
"""
    return lint_student_output(md, notes)


def build_viva_notes(notes: PaperNotes) -> str:
    """Viva references proposal sections — does not reprint routing/table/plan."""
    title = notes.title or "the paper"
    paper_q, paper_a = _paper_qa_question(notes)
    substitute = suggest_substitute(notes.dataset, notes.task_type, notes.contribution)
    limits = limits_from_plan(
        notes.python_scope, notes.missing_items, task_type=notes.task_type
    )

    qa: list[tuple[str, str]] = [
        (f"What problem does “{title}” tackle?", notes.problem),
        ("What method did the authors propose?", notes.method),
        (paper_q, paper_a),
        (
            "Which components will you implement, approximate, or cite?",
            "See Component status in the proposal.",
        ),
        ("What data did the paper name?", notes.dataset),
        ("What data will you use for the mini project?", substitute),
        ("Which evaluation metrics did the paper name?", notes.metrics),
        ("What limits your student build?", limits),
        (
            "What is your end-to-end pipeline?",
            "See Implementation plan in the proposal.",
        ),
    ]

    # Only paper-grounded extras with reason cues for "why", else skip status restates
    seen = {q for q, _ in qa}
    for q, a in notes.viva_extras:
        if not q or q in seen:
            continue
        if re.search(r"(?i)why ", q) and not REASON_CUE.search(a + " " + notes.method + " " + notes.problem):
            # Rewrite as what-question or skip
            continue
        if re.search(r"(?i)approximate|status table|routed|score=|component status", q + a):
            continue
        if re.search(
            r"(?i)table check\s*\(|inconsistency:|regression/degradation|Ablation:\s*Vehicle\b",
            q + a,
        ):
            continue
        if re.search(r"(?i)no proving|inferred encoder|debug|lexicon|do not invent", a):
            continue
        qa.append((q, a))
        seen.add(q)

    lines = [
        "# Viva Notes",
        "",
        f"**Paper:** {title}",
        "",
        "_Focus, component status, and plan are in the proposal — answers below point there when needed._",
        "",
    ]
    for i, (q, a) in enumerate(qa, start=1):
        lines.append(f"## Q{i}. {q}")
        lines.append(a)
        lines.append("")
    return lint_student_output("\n".join(lines).strip() + "\n", notes)


def build_markdown_bundle(notes: PaperNotes, student_name: str = "Student") -> str:
    """Student proposal + viva only. Evidence stays in the app Debug expander."""
    return (
        build_proposal(notes, student_name=student_name)
        + "\n\n---\n\n"
        + build_viva_notes(notes)
    )
