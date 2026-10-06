"""Input-quality + paper-kind detection and honest fallback plans.

Goal: every paper still yields usable student notes without inventing facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class InputProfile:
    input_tier: str  # full | abstract_only | metadata_only | sparse_ocr | equation_heavy
    paper_kind: str  # ml_method | survey | theory | hardware | math | non_ml | unknown
    warnings: list[str]
    missing_hints: list[str]


_META_PLACEHOLDER = re.compile(
    r"(?i)abstract not available from open metadata|upload the pdf or paste"
)
_NON_ML = re.compile(
    r"(?i)\b("
    r"no (?:machine learning|ml|deep learning)|"
    r"without (?:ml|neural|deep learning)|"
    r"calibration|instrumentation|optics|spectroscop|"
    r"hardware (?:design|implementation)|fpga|asic|"
    r"circuit design|antenna|mechanical engineering|"
    r"clinical trial(?!.*model)|purely theoretical"
    r")\b"
)
_THEORY = re.compile(
    r"(?i)\b("
    r"theorem|lemma|corollary|proof of|we prove|"
    r"convergence analysis|complexity analysis|"
    r"theoretical analysis|pac[- ]bayes|generalization bound|"
    r"information.?theoretic"
    r")\b"
)
_MATH_HEAVY = re.compile(
    r"(?i)(\\begin\{equation\}|\\\[|LNWD|∇|∑|∫|∥|argmax|argmin)"
)
_HARDWARE = re.compile(
    r"(?i)\b(fpga|asic|gpu kernel|cuda kernel|edge device|embedded system|"
    r"hardware accelerator|rtl|verilog|vhdl|mcu|raspberry pi)\b"
)
_SURVEY = re.compile(
    r"(?i)\b(this survey|literature review|we review|taxonomy|comprehensive overview|"
    r"state of the art review|systematic review)\b"
)
_ML_CUE = re.compile(
    r"(?i)\b("
    r"neural|deep learning|machine learning|transformer|cnn|rnn|gan|"
    r"yolo|bert|resnet|gradient boosting|reinforcement learning|"
    r"object detection|classification|segmentation|nlp|dataset|training"
    r")\b"
)


def assess_input(text: str, title: str = "", abstract: str = "") -> InputProfile:
    """Classify how much usable text we have and what kind of paper it looks like."""
    raw = text or ""
    abs_ = (abstract or "").strip()
    title = (title or "").strip()
    warnings: list[str] = []
    missing: list[str] = []

    letters = len(re.findall(r"[A-Za-z]", raw))
    digits = len(re.findall(r"\d", raw))
    eq_glyphs = len(re.findall(r"[∑∫∇√≈≤≥]|\\[a-zA-Z]+\{", raw))
    meta_only = bool(_META_PLACEHOLDER.search(abs_)) or (
        len(abs_) < 40 and letters < 400 and not re.search(r"(?i)\babstract\b", raw)
    )
    sparse = letters < 500 and len(raw) > 0
    eq_heavy = eq_glyphs >= 12 or (digits > letters * 0.55 and letters > 80)

    if meta_only:
        tier = "metadata_only"
        warnings.append(
            "Only title/metadata was available. Paste the abstract or upload the PDF for fuller notes."
        )
        missing.extend(["abstract", "method section", "experiments/results"])
    elif sparse:
        tier = "sparse_ocr"
        warnings.append(
            "Extracted text looks sparse (possible scanned PDF or broken OCR). "
            "Paste the abstract/method if notes look thin."
        )
        missing.append("readable body text")
    elif eq_heavy and letters < 2500:
        tier = "equation_heavy"
        warnings.append(
            "Text is equation-heavy; prose fields may be short. Method/problem use clean sentences only."
        )
    elif abs_ and letters < 1800 and not re.search(
        r"(?im)^\s*(?:\d+\.?\s*)?(?:method|experiments?|results?)\b", raw
    ):
        tier = "abstract_only"
        warnings.append(
            "Input looks abstract-front-matter heavy; experiments/results may be missing."
        )
        missing.append("experiments/results/dataset sections")
    else:
        tier = "full"

    blob = f"{title}\n{abs_}\n{raw[:8000]}"
    if _SURVEY.search(blob):
        kind = "survey"
    elif _NON_ML.search(blob) and not _ML_CUE.search(blob):
        kind = "non_ml"
    elif _HARDWARE.search(blob) and not re.search(
        r"(?i)\b(yolo|bert|resnet|detection|classification accuracy)\b", blob
    ):
        kind = "hardware"
    elif _THEORY.search(blob) and not re.search(
        r"(?i)\b(we (?:train|fine-tune|evaluate)|experimental results|dataset)\b", blob
    ):
        kind = "theory"
    elif eq_heavy and not _ML_CUE.search(abs_[:800] if abs_ else blob[:800]):
        kind = "math"
    elif _ML_CUE.search(blob):
        kind = "ml_method"
    else:
        kind = "unknown"

    if kind in {"theory", "math"}:
        missing.append("empirical evaluation (may be none in this paper)")
    if kind == "non_ml":
        warnings.append(
            "Paper does not look like a standard ML model paper; plan is a study/report, not a YOLO/NLP stack."
        )
    if kind == "hardware":
        warnings.append(
            "Hardware/systems cues detected; student plan focuses on measurement/demo, not claiming chip results."
        )

    return InputProfile(tier, kind, warnings, missing)


def fallback_scope(paper_kind: str, contribution: str = "", title: str = "") -> list[str]:
    """Honest student plan when task templates do not fit."""
    t = title or "the paper"
    if paper_kind == "survey" or contribution == "survey":
        return [
            f"Write a structured summary of themes covered in “{t}”.",
            "Build a comparison table of 3–5 methods named in the text (cite only).",
            "Optional: reproduce one simple baseline idea on a tiny public set if the survey names one.",
            "Prepare viva answers from the comparison table; do not invent results.",
        ]
    if paper_kind == "theory":
        return [
            "Restate the main claim/theorem in plain words from the abstract only.",
            "List assumptions and what is proven vs assumed (quote the paper).",
            "Optional: implement a tiny numerical check only if the paper gives an algorithm.",
            "Write a short report + viva notes; do not invent experimental numbers.",
        ]
    if paper_kind == "math":
        return [
            "Extract definitions and the main objective in prose (skip equation dumps).",
            "Identify any named algorithm or procedure you could implement thinly.",
            "If none is implementable, deliver a derivation walkthrough + worked example.",
            "Do not claim paper numerical results without the same protocol.",
        ]
    if paper_kind == "hardware":
        return [
            "Summarize the system/hardware contribution from the abstract.",
            "List measured quantities the paper reports (latency, power, throughput) if named.",
            "Student demo: a software simulation or measurement script on available hardware only.",
            "Do not claim silicon/FPGA paper numbers without the same board and protocol.",
        ]
    if paper_kind == "non_ml":
        return [
            "Restate the problem and method from the provided text only.",
            "Document datasets/instruments named in the paper (or mark as not stated).",
            "Build a thin reproducible script or lab notebook for the named procedure if any.",
            "Do not force an ML stack (no YOLO/BERT) unless the paper uses one.",
        ]
    if paper_kind == "unknown" or contribution == "new_dataset_benchmark":
        if contribution == "new_dataset_benchmark":
            return [
                "Describe the dataset/benchmark contribution from the text only.",
                "List fields, splits, and metrics named in the paper (mark gaps honestly).",
                "Student proxy: a tiny public subset or synthetic sample with the same schema if possible.",
                "Report baseline metrics on the proxy; do not claim paper leaderboard numbers.",
            ]
        return [
            "Confirm the paper’s task from the abstract (do not force a template).",
            "Restate problem and method in one paragraph each from the text.",
            "If a named algorithm appears, implement a thin public-data demo; otherwise write a study report.",
            "Document missing inputs; never invent datasets or metrics.",
        ]
    # ml_method default thin
    return [
        "Restate the paper problem and method from the provided text.",
        "Implement a thin Python demo using only components named in the text.",
        "Evaluate with metrics named in the paper on a student-scale public set.",
        "Do not claim paper-scale numbers without the same protocol.",
    ]


def fallback_stack(paper_kind: str) -> list[str]:
    if paper_kind in {"theory", "math", "survey", "non_ml"}:
        return ["Python 3", "Markdown/LaTeX for notes", "Matplotlib (optional)", "Streamlit (optional)"]
    if paper_kind == "hardware":
        return ["Python 3", "NumPy", "platform tools as available", "Streamlit (optional)"]
    return ["Python 3", "pandas", "scikit-learn or PyTorch", "Matplotlib", "Streamlit"]


def fallback_out_of_scope(paper_kind: str) -> list[str]:
    return [
        "Inventing datasets, metrics, or results not in the provided text.",
        "Claiming paper numbers without the same protocol.",
        "Forcing YOLO/BERT/detection stacks when the paper is unrelated.",
    ]


def title_only_problem(title: str) -> str:
    if not title:
        return "Problem not stated in the provided text."
    return (
        "Problem not stated in the provided text. "
        f"Title only available: “{title}”. Paste the abstract to extract the problem sentence."
    )


def title_only_method(title: str) -> str:
    if not title:
        return "Not stated in the provided text."
    return (
        "Not stated in the provided text. "
        f"From the title alone, the topic appears related to “{title}”, "
        "but no method sentence was available. Paste the abstract or method section."
    )
