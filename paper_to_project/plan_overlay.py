"""Contribution-aware plan from RoleRecords; stack matches baseline steps."""

from __future__ import annotations

import re

from .robustness import fallback_out_of_scope, fallback_scope, fallback_stack
from .roles import RoleRecord, detection_plan_from_records, plan_steps_from_records
from .templates import TaskTemplate, get_template_by_type


STEP_DEPS: list[tuple[str, str]] = [
    (r"(?i)streamlit", "Streamlit"),
    (r"(?i)correlation|pearson|spearman|kendall", "SciPy"),
    (r"(?i)bleu|meteor|rouge", "sacrebleu"),
    # Do NOT match bare "pretrained" (YOLO plans say "pretrained detector")
    (r"(?i)\b(?:bert|roberta|xlnet|token-level|token embeddings?|transformers)\b", "transformers"),
    (r"(?i)\bencoder\b.*\bembedding|\bembedding\b.*\bencoder\b", "transformers"),
    (r"(?i)yolo|ultralytics|detector|bytetrack|tracking", "Ultralytics YOLO / PyTorch"),
    (r"(?i)opencv|video|bounding box", "OpenCV"),
    (r"(?i)pytorch|torch", "PyTorch"),
    (r"(?i)plot|matplot", "Matplotlib"),
    (r"(?i)numpy|cosine|idf|rescal|similarity|matching", "NumPy"),
]

VISION_TASKS = {
    "object_detection_tracking",
    "image_classification",
    "semantic_segmentation",
    "generative_models",
}

FALLBACK_KINDS = {"theory", "hardware", "math", "non_ml", "unknown", "survey"}


def stack_from_steps(steps: list[str], base_stack: list[str] | None = None) -> list[str]:
    stack = ["Python 3"]
    seen = {"python 3"}
    for step in steps:
        for pat, lib in STEP_DEPS:
            if re.search(pat, step) and lib.lower() not in seen:
                stack.append(lib)
                seen.add(lib.lower())
    if base_stack:
        for lib in base_stack:
            if lib.lower() not in seen and lib.lower() != "python 3":
                if re.search(r"(?i)yolo|opencv|ultralytics|pytorch|torch", lib):
                    stack.append(lib)
                    seen.add(lib.lower())
    if "Streamlit" not in seen and any(re.search(r"(?i)streamlit", s) for s in steps):
        stack.append("Streamlit")
    elif "Streamlit" not in seen and base_stack and any(
        "streamlit" in (x or "").lower() for x in base_stack
    ):
        stack.append("Streamlit")
    return stack


def survey_scope(title: str = "") -> list[str]:
    return fallback_scope("survey", contribution="survey", title=title)


def apply_overlay(
    base: TaskTemplate,
    contribution: str,
    components: list[str] | None = None,
    paper_text: str = "",
    role_records: list[RoleRecord] | None = None,
    paper_kind: str = "ml_method",
    title: str = "",
) -> TaskTemplate:
    comps = components or []
    kind = paper_kind or "ml_method"

    # Honest fallbacks for non-standard papers (theory / hardware / math / non-ML)
    if kind in {"theory", "hardware", "math", "non_ml"} or (
        kind == "unknown" and base.task_type in {"unknown", "general_ml"} and not comps
    ):
        use_kind = kind if kind in FALLBACK_KINDS else "unknown"
        scope = fallback_scope(use_kind, contribution=contribution, title=title)
        return TaskTemplate(
            task_type="unknown",
            label=base.label + f" ({use_kind} study)",
            stack=fallback_stack(use_kind),
            python_scope=scope,
            out_of_scope=fallback_out_of_scope(use_kind),
            typical_metrics=list(base.typical_metrics),
            viva_extras=[],
            component_lexicon=list(base.component_lexicon),
            compute_tier="low",
        )

    if contribution == "new_metric":
        if role_records:
            scope = plan_steps_from_records(
                role_records, paper_text, contribution, task_type=base.task_type
            )
        else:
            from .roles import build_role_records
            from .components import ComponentItem
            items = [ComponentItem(c, "technique") for c in comps]
            scope = plan_steps_from_records(
                build_role_records(
                    items, paper_text, contribution, task_type=base.task_type
                ),
                paper_text,
                contribution,
                task_type=base.task_type,
            )
        return TaskTemplate(
            task_type=base.task_type,
            label=base.label + " (evaluation metric)",
            stack=stack_from_steps(scope),
            python_scope=scope,
            out_of_scope=[
                "Classifier fine-tuning.",
                "Full shared-task leaderboard reproduction.",
                "Claiming paper correlation numbers without matching ratings.",
            ],
            typical_metrics=["Pearson", "Kendall", "BLEU", "METEOR"],
            viva_extras=[],
            component_lexicon=list(base.component_lexicon)
            + [
                "BERTScore", "cosine similarity", "greedy matching",
                "IDF weighting", "baseline rescaling", "contextual embeddings",
            ],
            compute_tier="low",
        )

    if contribution == "survey":
        scope = survey_scope(title=title)
        return TaskTemplate(
            task_type=base.task_type,
            label=base.label + " (survey study)",
            stack=fallback_stack("survey"),
            python_scope=scope,
            out_of_scope=["Implementing every surveyed method."],
            typical_metrics=["accuracy", "F1"],
            viva_extras=[],
            component_lexicon=list(base.component_lexicon),
            compute_tier="low",
        )

    if contribution == "new_dataset_benchmark":
        scope = fallback_scope("unknown", contribution="new_dataset_benchmark", title=title)
        return TaskTemplate(
            task_type=base.task_type if base.task_type != "unknown" else "general_ml",
            label=base.label + " (dataset/benchmark study)",
            stack=fallback_stack("ml_method"),
            python_scope=scope,
            out_of_scope=fallback_out_of_scope("unknown"),
            typical_metrics=list(base.typical_metrics) or ["accuracy", "F1"],
            viva_extras=[],
            component_lexicon=list(base.component_lexicon),
            compute_tier="low",
        )

    if base.task_type == "object_detection_tracking" and role_records:
        scope = detection_plan_from_records(role_records, paper_text)
        return TaskTemplate(
            task_type=base.task_type,
            label=base.label,
            stack=stack_from_steps(scope, list(base.stack)),
            python_scope=scope,
            out_of_scope=list(base.out_of_scope),
            typical_metrics=list(base.typical_metrics),
            viva_extras=[],
            component_lexicon=list(base.component_lexicon),
            compute_tier=base.compute_tier,
        )

    # Weak task score → general_ml / unknown use honest fallback if no components
    if base.task_type in {"unknown", "general_ml"} and not comps:
        scope = fallback_scope(kind if kind != "ml_method" else "unknown", contribution, title)
        return TaskTemplate(
            task_type=base.task_type,
            label=base.label,
            stack=fallback_stack(kind if kind != "ml_method" else "unknown"),
            python_scope=scope,
            out_of_scope=fallback_out_of_scope(kind if kind != "ml_method" else "unknown"),
            typical_metrics=list(base.typical_metrics),
            viva_extras=[],
            component_lexicon=list(base.component_lexicon),
            compute_tier="low",
        )

    scope = list(base.python_scope)
    if base.task_type in VISION_TASKS:
        stack = list(base.stack)
    else:
        derived = stack_from_steps(scope, list(base.stack))
        stack = derived if len(derived) > 2 else list(base.stack)
    return TaskTemplate(
        task_type=base.task_type,
        label=base.label,
        stack=stack,
        python_scope=scope,
        out_of_scope=list(base.out_of_scope),
        typical_metrics=list(base.typical_metrics),
        viva_extras=[],
        component_lexicon=list(base.component_lexicon),
        compute_tier=base.compute_tier,
    )


def resolve_template(
    task_type: str,
    contribution: str,
    components: list[str] | None = None,
    paper_text: str = "",
    role_records: list[RoleRecord] | None = None,
    paper_kind: str = "ml_method",
    title: str = "",
) -> TaskTemplate:
    # Route unknown ML-ish text to general_ml when components exist
    tt = task_type
    if tt == "unknown" and components and paper_kind == "ml_method":
        tt = "general_ml"
    return apply_overlay(
        get_template_by_type(tt),
        contribution,
        components,
        paper_text,
        role_records,
        paper_kind=paper_kind,
        title=title,
    )
