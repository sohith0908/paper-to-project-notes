"""One role record drives status, compute, and plan step together."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .components import ComponentItem


@dataclass
class RoleRecord:
    name: str
    role: str  # proposed | backbone | technique
    use_mode: str  # inference | training | scoring
    status: str  # implement | approximate | cite
    compute_note: str
    plan_step: str
    provenance: str  # from_paper | inferred | unverified
    reason: str
    evidence_span: str = ""


_PRETRAINED_NEAR = re.compile(
    r"(?i)pretrained\s+(?:\w+\s+){0,3}(BERT|RoBERTa|XLNet|encoder|embeddings?)"
)
_ROLE_CUE = re.compile(r"(?i)\b(pretrained|backbone|encoder|frozen|embeddings?)\b")
_VISION_TASKS = {
    "object_detection_tracking",
    "image_classification",
    "semantic_segmentation",
    "generative_models",
}
_DET_TRACK = {"object_detection_tracking"}

# Concrete plan actions for named techniques (metric papers)
TECHNIQUE_ACTIONS: dict[str, str] = {
    "contextual embeddings": "Extract token-level vectors from the encoder for each reference and candidate.",
    "compute similarity": "Score token pairs with cosine similarity.",
    "greedy matching": "Match each candidate token to its best reference token by cosine similarity.",
    "idf weighting": "Weight token matches by inverse document frequency.",
    "baseline rescaling": "Rescale metric scores with a baseline for readability.",
    "cosine similarity": "Compute cosine similarity between token embeddings.",
    "token-level matching": "Match tokens greedily between candidate and reference.",
}

# Detection/tracking modules that students cite rather than re-code
_CITE_TECHNIQUES = re.compile(
    r"(?i)^(C2f_|Detect_|MSN|NWD|MLCA|SEAM|attention|FPN|NMS)$|loss$"
)
_TRACKERS_IMPLEMENT = re.compile(r"(?i)^(ByteTrack|FairMOT|StrongSORT)$")
_TRACKERS_CITE = re.compile(r"(?i)^(DeepSORT|SORT)$")
_DETECTOR_BACKBONE = re.compile(r"(?i)^YOLOv?\d")


def _whole_word_span(text: str, name: str) -> str:
    if not name or not text:
        return ""
    pat = re.compile(rf"(?i)(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])")
    parts = re.split(r"(?<=[.!?])\s+", text)
    for s in parts:
        if pat.search(s):
            return s.strip()[:320]
    for line in text.splitlines()[:8]:
        if pat.search(line):
            return line.strip()[:320]
    return ""


def _backbone_span(text: str, name: str, title: str = "") -> str:
    """Whole-word name + role cue, or title that names the backbone."""
    pat = re.compile(rf"(?i)(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])")
    hay = (title + "\n" + text) if title else text
    parts = re.split(r"(?<=[.!?])\s+", hay)
    candidates = ([title.strip()] if title else []) + [s.strip() for s in parts]
    for s in candidates:
        if not s or not pat.search(s):
            continue
        if _ROLE_CUE.search(s) or re.search(rf"(?i)\bwith\s+{re.escape(name)}\b", s):
            return s[:320]
        # Vision titles: "Improved YOLOv8n and ByteTrack"
        if _DETECTOR_BACKBONE.search(name) and re.search(r"(?i)yolo|detect|track", s):
            return s[:320]
    return ""


def _is_metric_path(contribution: str, task_type: str) -> bool:
    return contribution == "new_metric" or (
        task_type == "nlp_text" and contribution == "new_metric"
    )


def build_role_records(
    items: list[ComponentItem],
    paper_text: str,
    contribution: str = "new_model",
    title: str = "",
    task_type: str = "unknown",
) -> list[RoleRecord]:
    records: list[RoleRecord] = []
    body = paper_text[:12000]
    metric_path = contribution == "new_metric"
    vision = task_type in _VISION_TASKS
    det = task_type in _DET_TRACK

    for it in items:
        name, role = it.name, it.role
        if role == "backbone":
            span = _backbone_span(body, name, title=title)
        else:
            span = _whole_word_span(body, name)

        if metric_path and role == "proposed":
            use_mode = "scoring"
            status = "implement"
            compute = "CPU for scoring on a small set of sentence pairs."
            plan = f"Implement {name} scoring on reference and candidate pairs."
            reason = "Named as the proposed contribution."
            prov = "from_paper" if span else "unverified"

        elif metric_path and role == "backbone":
            pretrained = bool(_PRETRAINED_NEAR.search(body) or (span and _ROLE_CUE.search(span)))
            title_names = bool(title and re.search(
                rf"(?i)(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", title
            ))
            if pretrained or title_names:
                use_mode = "inference"
                compute = "CPU for inference; load pretrained weights, no training."
                plan = f"Load a pretrained {name} encoder and extract token embeddings (no training)."
                if span and (pretrained or re.search(rf"(?i)\bwith\s+{re.escape(name)}\b", span)):
                    status = "implement"
                    reason = "Named with the encoder in the paper text or title."
                    prov = "from_paper"
                elif span or title_names:
                    status = "approximate"
                    reason = f"The title or text mentions {name}; treat as a pretrained encoder."
                    prov = "inferred"
                    if not span and title_names:
                        span = title.strip()[:320]
                else:
                    status = "approximate"
                    reason = f"Assumed pretrained {name} encoder for this metric study."
                    prov = "inferred"
            else:
                use_mode = "training"
                status = "approximate"
                compute = "Single GPU for fine-tuning on a small public set."
                plan = f"Fine-tune a small {name} checkpoint on a short public set."
                reason = f"{name} is named as a backbone."
                prov = "from_paper" if span else "unverified"

        elif metric_path:
            use_mode = "scoring"
            status = "implement"
            compute = "CPU; tiny set of sentence pairs."
            plan = TECHNIQUE_ACTIONS.get(
                name.lower(),
                f"Code the {name} step in the metric pipeline.",
            )
            reason = "Named technique in the method."
            prov = "from_paper" if span else "unverified"

        elif role == "proposed":
            use_mode = "inference" if det else "training"
            if det and re.search(r"(?i)msn|yolo", name):
                # Proposed detector family: cite custom stack; implement via public YOLO
                status = "cite"
                compute = "-"
                plan = ""
                reason = "Paper’s proposed detector; cite modules and implement via a public YOLO checkpoint."
            else:
                status = "approximate"
                compute = (
                    "GPU preferred for detector inference; CPU OK for a short clip."
                    if det
                    else "Single GPU for a thin student demo."
                )
                plan = (
                    f"Prototype the paper idea around {name} on a short public set."
                    if not det
                    else f"Use a public YOLO checkpoint as a stand-in for {name}; cite paper modules."
                )
                reason = "Named as the proposed contribution."
            prov = "from_paper" if span else "unverified"

        elif role == "backbone" and (vision or det or _DETECTOR_BACKBONE.search(name)):
            title_names = bool(title and re.search(
                rf"(?i)(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", title
            ))
            use_mode = "inference"
            status = "implement" if (span or title_names) else "approximate"
            compute = "GPU preferred; pretrained detector weights on a short video/image set."
            plan = f"Run a pretrained {name} detector on a short public video or image set."
            reason = (
                f"{name} is the detector backbone named in the title or method."
                if (span or title_names)
                else f"{name} is treated as the detector backbone for this task."
            )
            prov = "from_paper" if span else ("inferred" if title_names else "unverified")
            if not span and title_names:
                span = title.strip()[:320]
                prov = "from_paper"

        elif role == "backbone":
            use_mode = "training"
            status = "approximate"
            compute = "Single GPU for fine-tuning on a small public set."
            plan = f"Fine-tune a small {name} checkpoint on a short public set."
            reason = f"{name} is named as a backbone."
            prov = "from_paper" if span else "unverified"

        else:  # technique — task-aware
            if det and _TRACKERS_IMPLEMENT.search(name):
                use_mode = "inference"
                status = "implement"
                compute = "CPU/GPU for online tracking on a short video."
                plan = f"Attach {name} for multi-object ID continuity across frames."
                reason = "Named tracker in the paper method or title."
                prov = "from_paper" if span else "unverified"
            elif det and _TRACKERS_CITE.search(name):
                use_mode = "inference"
                status = "cite"
                compute = "-"
                plan = ""
                reason = "Compared or mentioned as prior work; not the chosen tracker."
                prov = "from_paper" if span else "unverified"
            elif det and (_CITE_TECHNIQUES.search(name) or re.search(r"(?i)MLCA|SEAM|NWD|MSN", name)):
                use_mode = "inference"
                status = "cite"
                compute = "-"
                plan = ""
                reason = "Paper-specific module; cite it rather than reimplementing."
                prov = "from_paper" if span else "unverified"
            elif det and re.search(r"(?i)^Re-?ID$", name):
                use_mode = "inference"
                status = "approximate"
                compute = "CPU; optional appearance features on a tiny clip."
                plan = "Optionally add a simple appearance / Re-ID cue for tougher IDs."
                reason = "Named for tracking association in the paper."
                prov = "from_paper" if span else "unverified"
            else:
                use_mode = "inference" if vision else "scoring"
                status = "cite" if vision else "implement"
                compute = "-" if vision else "CPU for a thin demo."
                plan = "" if vision else f"Code the {name} step in the student pipeline."
                reason = "Named technique in the method."
                prov = "from_paper" if span else "unverified"

        if prov == "unverified":
            status = "cite"
            plan = ""
            compute = "-"

        records.append(
            RoleRecord(
                name=name,
                role=role,
                use_mode=use_mode,
                status=status,
                compute_note=compute,
                plan_step=plan,
                provenance=prov,
                reason=reason,
                evidence_span=span,
            )
        )
    return records


def _step_key(s: str) -> str:
    t = s.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\b(the|a|an|and|for|from|each|with|by)\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _steps_overlap(a: str, b: str) -> bool:
    ka, kb = _step_key(a), _step_key(b)
    if not ka or not kb:
        return False
    if ka == kb or ka in kb or kb in ka:
        return True
    wa = {w for w in ka.split() if len(w) >= 4}
    wb = {w for w in kb.split() if len(w) >= 4}
    if not wa or not wb:
        return False
    inter = wa & wb
    return len(inter) >= 3 and len(inter) / min(len(wa), len(wb)) >= 0.6


def plan_steps_from_records(
    records: list[RoleRecord],
    paper_text: str = "",
    contribution: str = "new_model",
    task_type: str = "unknown",
) -> list[str]:
    """Metric-paper plan only. Detection uses detection_plan_from_records."""
    if contribution != "new_metric":
        if task_type in _DET_TRACK:
            return detection_plan_from_records(records, paper_text)
        return []

    steps: list[str] = []
    body = paper_text[:5000]

    def add(step: str) -> None:
        if not step or not step.strip():
            return
        if any(_steps_overlap(step, s) for s in steps):
            return
        steps.append(step.strip())

    for r in records:
        if r.role == "proposed" and r.plan_step:
            add(r.plan_step)
            break
    for r in records:
        if r.role == "backbone" and r.plan_step:
            add(r.plan_step)
            break

    for r in records:
        if r.role == "technique" and r.plan_step:
            if r.name.lower() == "contextual embeddings" and any(
                re.search(r"(?i)token embeddings|token-level", s) for s in steps
            ):
                continue
            add(r.plan_step)

    baselines = []
    for b in ("BLEU", "METEOR", "ROUGE"):
        if re.search(rf"(?i)\b{b}\b", body):
            baselines.append(b)
    if baselines:
        add("Compare against " + ", ".join(baselines) + ".")
    elif re.search(r"(?i)existing metrics?", body):
        add("Compare against BLEU as a baseline.")

    if re.search(r"(?i)pearson", body):
        add("Compute Pearson correlation with human judgments on a small rated set.")
    elif re.search(r"(?i)kendall", body):
        add("Compute Kendall correlation with human judgments on a small rated set.")
    elif re.search(r"(?i)spearman", body):
        add("Compute Spearman correlation with human judgments on a small rated set.")
    elif re.search(r"(?i)correlat", body):
        add("Compute correlation with human judgments on a small rated set.")

    add("Build a Streamlit demo for reference and candidate scoring.")
    return steps


def detection_plan_from_records(
    records: list[RoleRecord],
    paper_text: str = "",
) -> list[str]:
    """Student plan grounded in named detector/tracker — no NLP/metric wording."""
    steps: list[str] = []
    body = (paper_text or "")[:5000]

    def add(step: str) -> None:
        if step and not any(_steps_overlap(step, s) for s in steps):
            steps.append(step)

    backbone = next(
        (r for r in records if r.role == "backbone" and r.plan_step),
        None,
    )
    if backbone:
        add(backbone.plan_step)
    else:
        add("Run a pretrained small YOLO detector on a short public traffic video.")

    tracker = next(
        (r for r in records if r.status == "implement" and _TRACKERS_IMPLEMENT.search(r.name)),
        None,
    )
    if tracker and tracker.plan_step:
        add(tracker.plan_step)
    elif re.search(r"(?i)track|byte.?track|mot\b", body):
        add("Optionally add a lightweight tracker for ID continuity across frames.")

    if any(r.name.lower() in {"re-id", "reid"} and r.status == "approximate" for r in records):
        add("Optionally add a simple appearance / Re-ID cue for tougher IDs.")

    metrics = []
    for m in ("mAP50", "mAP", "MOTA", "IDF1", "FPS"):
        if re.search(rf"(?i)\b{re.escape(m)}\b", body):
            metrics.append(m)
    if metrics:
        add("Report " + "/".join(metrics[:4]) + " on a tiny labeled clip; build a Streamlit video demo.")
    else:
        add("Report detection/tracking metrics named in the paper on a tiny clip; build a Streamlit demo.")

    # Cite last so timeline weeks 1–3 stay implementation-focused
    cite_mods = [
        r.name
        for r in records
        if r.status == "cite" and re.search(r"(?i)MLCA|SEAM|NWD|MSN|C2f_|Detect_", r.name)
    ]
    if cite_mods:
        add("Cite paper-specific modules (" + ", ".join(cite_mods[:4]) + ") without reimplementing them.")
    return steps


def objectives_from_steps(steps: list[str], task_type: str, contribution: str) -> str:
    """Build objectives from concrete plan steps."""
    clean = [re.sub(r"^\[Plan\]\s*", "", s).rstrip(".") for s in steps if s.strip()]
    if not clean:
        return (
            "1. Restate the paper problem and method from the provided text.\n"
            "2. Implement a thin Python demo for the named contribution.\n"
            "3. Report metrics named in the paper."
        )
    objs = [
        "1. Restate the paper problem and method from the provided text.",
    ]
    core = clean[0]
    if len(clean) > 1:
        core = clean[0] + "; " + clean[1]
    objs.append(f"2. {core}.")
    eval_step = next(
        (s for s in clean if re.search(r"(?i)correlat|compare against|demo|streamlit|report ", s)),
        None,
    )
    if eval_step:
        objs.append(f"3. {eval_step}.")
    else:
        objs.append(f"3. Evaluate the demo for `{contribution}` and document results.")
    return "\n".join(objs)


def _default_deliverables(task_type: str) -> tuple[str, str, str]:
    if task_type in _DET_TRACK:
        return (
            "Detector script and sample video frames",
            "Tracked IDs on a short clip",
            "Metrics table and Streamlit video demo",
        )
    if task_type == "nlp_text":
        return (
            "Encoder load script and sample pairs",
            "Working metric scores on a tiny set",
            "Correlation numbers and Streamlit demo",
        )
    return (
        "Runnable project skeleton",
        "Working pipeline on a tiny set",
        "Metrics table and Streamlit demo",
    )


def timeline_from_steps(steps: list[str], task_type: str = "unknown") -> str:
    """Goal + deliverable per week; week 4 = report/viva."""
    clean = []
    for s in steps:
        s = re.sub(r"^\[Plan\]\s*", "", s).strip().rstrip(".")
        s = re.sub(r"\s*;\s*", "; ", s)
        s = re.sub(r";\s*\.", ".", s)
        s = re.sub(r"\.\s*;", ".", s)
        if s:
            clean.append(s)
    lines = [
        "| Week | Goal | Deliverable |",
        "|------|------|-------------|",
    ]
    d1, d2, d3 = _default_deliverables(task_type)
    if not clean:
        lines += [
            f"| 1 | Set up environment and data | {d1} |",
            f"| 2 | Core method | {d2} |",
            f"| 3 | Evaluation and demo | {d3} |",
            "| 4 | Report and viva practice | Short report and Q&A notes |",
        ]
        return "\n".join(lines)

    # Prefer implementation / eval steps for weeks 1–3 (skip optional/cite filler)
    core = [
        s for s in clean
        if not re.match(r"(?i)^(optionally|cite )\b", s)
    ] or clean
    n = len(core)
    w1 = core[0] if n else ""
    w2 = core[1] if n > 1 else (core[0] if n else "")
    # Prefer an eval/demo step for week 3 when present
    eval_steps = [s for s in core if re.search(r"(?i)report |demo|streamlit|evaluat|metric", s)]
    if eval_steps:
        w3 = eval_steps[-1]
    else:
        w3_parts = core[2: min(n, 5)] if n > 2 else ([core[-1]] if n > 1 else [])
        w3 = w3_parts[0] if w3_parts else "Run evaluation and polish the demo"

    def goal_deliverable(step: str, default_del: str) -> tuple[str, str]:
        goal = step
        if len(goal) > 90:
            goal = goal[:87].rsplit(" ", 1)[0] + "…"
        return goal, default_del

    g1, dd1 = goal_deliverable(w1, d1)
    g2, dd2 = goal_deliverable(w2, d2)
    g3, dd3 = goal_deliverable(w3, d3)

    lines.append(f"| 1 | {g1} | {dd1} |")
    lines.append(f"| 2 | {g2} | {dd2} |")
    lines.append(f"| 3 | {g3} | {dd3} |")
    lines.append("| 4 | Write up results and practise viva answers | Short report and viva notes |")
    return "\n".join(lines)


def limits_from_plan(
    steps: list[str],
    missing: list[str] | None = None,
    task_type: str = "unknown",
) -> str:
    """Concrete limits that follow from the plan and task."""
    blob = " ".join(steps).lower()
    bits = []
    if task_type in _DET_TRACK:
        bits.append("short public video only")
        bits.append("no full MOT/UA-DETRAC training run")
        if re.search(r"(?i)cite |without reimplement", blob):
            bits.append("paper-specific detector modules cited, not rebuilt")
        bits.append("do not claim paper mAP/MOTA without the same protocol")
    else:
        if re.search(r"(?i)small rated set|tiny set|sentence pairs", blob):
            bits.append("correlation only on a small human-rated set")
        if re.search(r"(?i)without training|no training|pretrained", blob) and "yolo" not in blob:
            bits.append("no encoder fine-tuning")
        if re.search(r"(?i)streamlit|demo", blob):
            bits.append("demo-scale data only")
        if not re.search(r"(?i)shared.?task|full wmt|leaderboard", blob):
            if task_type == "nlp_text" or "bleu" in blob or "bert" in blob:
                bits.append("no full shared-task leaderboard run")
    if missing and task_type not in _DET_TRACK:
        bits.append("input gaps: " + "; ".join(missing[:3]))
    if not bits:
        bits.append("student-scale data and compute only")
    return "Limits of this build: " + "; ".join(bits) + "."
