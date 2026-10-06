"""Dataset lexicon — separate From paper names vs Suggested cues."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .cleaner import extract_section

DATASET_LEXICON: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pat, re.I))
    for name, pat in [
        ("CIFAR-100", r"\bcifar-?100\b"),
        ("CIFAR-10", r"\bcifar-?10\b"),
        ("ImageNet", r"\bimage\s*net\b|\bimagenet\b"),
        ("MNIST", r"\bmnist\b"),
        ("Fashion-MNIST", r"\bfashion-?mnist\b"),
        ("PlantVillage", r"\bplantvillage\b"),
        ("CelebA", r"\bceleba\b"),
        ("Cora", r"\bcora\b"),
        ("CiteSeer", r"\bciteseer\b"),
        ("Cityscapes", r"\bcityscapes\b"),
        ("CamVid", r"\bcamvid\b"),
        ("LibriSpeech", r"\blibrispeech\b"),
        ("COCO Captions", r"\bcoco\s+captions?\b"),
        ("COCO", r"\b(?:ms[- ])?coco\b"),
        ("PASCAL VOC", r"\bpascal\s*voc\b|\bvoc20\d{2}\b"),
        ("UA-DETRAC", r"\bua-?detrac\b"),
        ("KITTI", r"\bkitti\b"),
        ("BDD100K", r"\bbdd100k\b"),
        ("VisDrone", r"\bvisdrone\b"),
        # MOT07 is a common PDF/OCR misread of MOT17 — normalize to MOT17
        ("MOT17", r"\bmot17\b|\bmot07\b"),
        ("MOT16", r"\bmot16\b"),
        ("WMT18", r"\bwmt18\b"),
        ("WMT17", r"\bwmt17\b"),
        ("WMT16", r"\bwmt16\b"),
        ("WMT14", r"\bwmt14\b"),
        ("WMT", r"\bwmt\d{2}\b"),
        ("PAWS", r"\bpaws\b"),
        ("QQP", r"\bqqp\b|\bquora question pairs?\b"),
        ("STS-B", r"\bsts-?b\b"),
        ("CNN/DailyMail", r"\bcnn[\s/-]*daily\s*mail\b"),
        ("XSum", r"\bxsum\b"),
        ("GLUE", r"\bglue\b"),
        ("SQuAD", r"\bsquad\b"),
        ("IMDB", r"\bimdb\b"),
        ("SST-2", r"\bsst-?2\b"),
        ("AG News", r"\bag\s*news\b"),
        ("UCI", r"\buci\b"),
        ("Adult", r"\badult(?:\s+dataset)?\b"),
        ("Higgs", r"\bhiggs\b"),
        ("Covertype", r"\bcovertype\b"),
    ]
]

# Cues that suggest a dataset but do not name it in the text
DATASET_HINTS: list[tuple[str, str]] = [
    (r"(?i)\badversarial paraphrase detection\b", "PAWS — cue: “adversarial paraphrase detection”"),
    (r"(?i)\bshared task\b", "WMT metrics task — cue: “shared task”"),
    (r"(?i)\bsemantic textual similarity\b", "STS-B — cue: “semantic textual similarity”"),
]

META_LINE = re.compile(
    r"(?i)^(year|authors?|doi|keywords?|abstract|title|url|source|citation)\s*:"
)


@dataclass
class DatasetExtract:
    from_paper: str = "Not stated in the provided text."
    suggested: str = ""
    names: list[str] = field(default_factory=list)


def strip_metadata_lines(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if META_LINE.match(s):
            if re.match(r"(?i)^abstract\s*:", s):
                lines.append(re.sub(r"(?i)^abstract\s*:", "", s).strip())
            continue
        if re.match(r"(?i)^(year|doi|keywords?)\b", s) and len(s) < 80:
            continue
        lines.append(s)
    out: list[str] = []
    prev = ""
    for ln in lines:
        key = re.sub(r"\s+", " ", ln.lower())
        if key == prev:
            continue
        out.append(ln)
        prev = key
    return "\n".join(out)


def _local_window(text: str, start: int, name: str, span: int = 180) -> str:
    window = text[start : start + span]
    for other, opat in DATASET_LEXICON:
        if other.lower() == name.lower():
            continue
        om = opat.search(window[len(name) + 1 :])
        if om:
            window = window[: len(name) + 1 + om.start()]
            break
    return window


def _mention_detail(text: str, name: str) -> str:
    pat = next((p for n, p in DATASET_LEXICON if n == name), None)
    if not pat:
        return name
    best_bits = [name]
    for m in pat.finditer(text):
        window = _local_window(text, m.start(), name)
        bits = [name]
        train = re.search(
            r"(\d+(?:\.\d+)?\s*[kKmM]?|\d[\d,]*)\s*(?:training|train)\b",
            window,
            re.I,
        )
        val = re.search(
            r"(\d+(?:\.\d+)?\s*[kKmM]?|\d[\d,]*)\s*(?:val(?:idation)?|test)\b",
            window,
            re.I,
        )
        if name.upper().startswith("CIFAR"):
            split45 = re.search(
                r"(?i)(\d+)\s*[kK]?\s*(?:train(?:ing)?|/)\s*(?:and\s+a?\s*)?(\d+)\s*[kK]?\s*(?:val|validation)",
                window,
            )
            if split45:
                bits.append(f"{split45.group(1)} train / {split45.group(2)} val")
        if train and val:
            bits.append(f"{train.group(1)} train, {val.group(1)} val/test")
        elif train:
            bits.append(f"{train.group(1)} train")
        elif val:
            bits.append(f"{val.group(1)} val/test")
        if len(bits) > len(best_bits):
            best_bits = bits
    if len(best_bits) == 1:
        return f"{name} (mentioned; no split/size)"
    return "; ".join(best_bits)


def suggest_substitute(
    dataset_field: str,
    task_type: str,
    contribution: str = "new_model",
) -> str:
    """Plan-only substitute — never repeats paper 'not stated'."""
    if contribution == "new_metric":
        return "STS-B or a tiny WMT segment-level sample with human scores."
    dl = (dataset_field or "").lower()
    if "not stated" in dl:
        return "Confirm paper datasets from the full PDF, then pick a public student-scale proxy."
    if task_type == "image_classification":
        if "imagenet" in dl:
            return "CIFAR-10 or a tiny ImageNet subset."
        if "cifar" in dl:
            return "Official CIFAR-10/100 train/test split."
    if task_type == "classic_tabular_ml":
        return "Public UCI Adult tabular set."
    if task_type == "nlp_text":
        return "SST-2 subset."
    if task_type == "object_detection_tracking":
        return "Short public video with labels."
    return "Smaller public proxy of the same task type."


def extract_datasets_structured(text: str, contribution: str = "new_model") -> DatasetExtract:
    cleaned = strip_metadata_lines(text)
    section = extract_section(
        cleaned,
        start_patterns=[
            r"(?m)^\s*\d+(\.\d+)*\.?\s+(Dataset|Data Set|Experimental Setup|Materials|Data)\b",
            r"(?m)^\s*(Dataset|Data Set|Experimental Data|Materials and Methods)\b",
        ],
        end_patterns=[
            r"(?m)^\s*\d+(\.\d+)*\.?\s+(Method|Model|Proposed|Evaluation|Metric|Experiment|Result|Loss|Architecture)\b",
            r"(?m)^\s*(Evaluation|Metrics?|Results?|Proposed Method|Architecture)\b",
        ],
    )
    hay = section if section else cleaned
    found: list[str] = []
    names: list[str] = []
    seen = set()
    for name, pat in DATASET_LEXICON:
        if pat.search(hay) and name.lower() not in seen:
            if name == "COCO" and "coco captions" in seen:
                continue
            if name == "WMT" and any(n.startswith("WMT") and n != "WMT" for n in names):
                continue  # prefer WMT16 over bare WMT
            seen.add(name.lower())
            names.append(name)
            found.append(_mention_detail(hay, name))

    suggested: list[str] = []
    for pat, label in DATASET_HINTS:
        if re.search(pat, hay):
            # Only suggest if the named dataset is NOT already in from_paper
            cue_name = label.split()[0]
            if cue_name.lower() not in seen:
                suggested.append(label)

    if found:
        return DatasetExtract(
            from_paper="; ".join(found),
            suggested="; ".join(suggested) if suggested else "",
            names=names,
        )
    return DatasetExtract(
        from_paper="Not stated in the provided text.",
        suggested="; ".join(suggested) if suggested else "",
        names=[],
    )


def extract_datasets(text: str, contribution: str = "new_model") -> str:
    """Backward-compatible string — From paper only (suggested is separate on PaperNotes)."""
    return extract_datasets_structured(text, contribution).from_paper
