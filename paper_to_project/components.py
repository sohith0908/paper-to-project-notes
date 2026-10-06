"""Extract named method components with roles: proposed | backbone | technique."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .templates import get_template_by_type


@dataclass
class ComponentItem:
    name: str
    role: str  # proposed | backbone | technique


# Generic nouns — never treat as standalone components
GENERIC_NOUNS = {
    "embedding", "embeddings", "model", "network", "layer", "feature", "features",
    "score", "scores", "method", "approach", "framework", "algorithm", "module",
    "system", "architecture", "pipeline", "baseline", "baselines", "metric",
    "metrics", "loss", "losses", "representation", "representations",
}

STOP = {
    "the", "this", "that", "these", "those", "with", "from", "using", "based",
    "proposed", "introduction", "abstract", "conclusion", "results", "discussion",
    "figure", "table", "section", "innovative", "improved", "novel",
    "traditional", "existing", "survey", "review", "imagenet", "cifar",
    "mnist", "coco", "dataset", "and", "for", "our", "residual",
} | GENERIC_NOUNS

DATASET_NAMES = {
    "imagenet", "cifar", "cifar-10", "cifar-100", "mnist", "coco", "glue",
    "squad", "kitti", "mot17", "mot16", "ua-detrac", "bdd100k", "visdrone",
    "paws", "qqp", "sts-b", "wmt", "xsum",
}

RELATED_WORK_ONLY = {
    "googlenet", "alexnet", "vgg", "vgg16", "vgg19", "inception", "squeezenet",
    "deepsort", "sort", "fairmot", "yolov3", "yolov4", "yolov5", "yolov6", "yolov7",
}

BACKBONE_NAMES = {
    "bert", "roberta", "xlnet", "gpt", "gpt-2", "t5", "resnet", "resnet-18",
    "resnet-34", "resnet-50", "resnet-101", "resnet-152", "yolov8", "yolov5",
    "efficientnet", "vit", "transformer", "whisper",
}

METRIC_LEXICON: list[tuple[str, str]] = [
    (r"\bBERTScore\b", "BERTScore"),
    (r"\bcosine similarity\b", "cosine similarity"),
    (r"\bgreedy matching\b", "greedy matching"),
    (r"\bIDF (?:weight(?:ing|s)?|importance)\b", "IDF weighting"),
    (r"\b(?:baseline )?rescal(?:ing|e)\b", "baseline rescaling"),
    (r"\bcontextual embeddings?\b", "contextual embeddings"),
    (r"\btoken[- ]level (?:matching|similarity)\b", "token-level matching"),
    (r"\bprecision[,/ ]?\s*recall[,/ ]?\s*(?:and\s+)?F1\b", "precision/recall/F1"),
]

# Operations / functions (not just nouns)
OPERATION_LEXICON: list[tuple[str, str]] = [
    (r"\bcompute similarity\b", "compute similarity"),
    (r"\bgreedy matching\b", "greedy matching"),
    (r"\bcosine similarity\b", "cosine similarity"),
    (r"\bIDF weight(?:ing|s)?\b", "IDF weighting"),
    (r"\bbaseline rescal(?:ing|e)\b", "baseline rescaling"),
    (r"\bmatch(?:ing)? (?:token|word) embeddings?\b", "token embedding matching"),
    (r"\brescal(?:ing|e)\b", "baseline rescaling"),
]

TASK_TERM_SCAN: dict[str, list[tuple[str, str]]] = {
    "image_classification": [
        (r"\bresidual learning\b", "residual learning"),
        (r"\bresidual blocks?\b", "residual block"),
        (r"\bidentity (?:mappings?|connections?|shortcuts?)\b", "identity shortcut"),
        (r"\bskip connections?\b", "skip connection"),
        (r"\bbottlenecks?\b", "bottleneck"),
        (r"\bbatch norm(?:alisation|alization)?\b", "batch normalization"),
        (r"\bdegradation problem\b", "degradation problem"),
        (r"\bResNet-?(18|34|50|101|152)\b", "ResNet-{g1}"),
        (r"\bResNet\b", "ResNet"),
    ],
    "classic_tabular_ml": [
        (r"\bxgboost\b", "XGBoost"),
        (r"\bgradient boosting\b", "gradient boosting"),
        (r"\bdecision trees?\b", "decision tree"),
        (r"\bregularization\b", "regularization"),
        (r"\bfeature importance\b", "feature importance"),
    ],
    "nlp_text": [
        (r"\bBERTScore\b", "BERTScore"),
        (r"\bbert\b", "BERT"),
        (r"\btransformer\b", "transformer"),
        (r"\battention\b", "attention"),
        (r"\btokenizer\b", "tokenizer"),
        (r"\bmasked language\b", "masked language modeling"),
        (r"\bcosine similarity\b", "cosine similarity"),
        (r"\bgreedy matching\b", "greedy matching"),
        (r"\bIDF\b", "IDF weighting"),
    ],
    "object_detection_tracking": [
        (r"\bYOLOv?\d+\w*\b", None),
        (r"\bByteTrack\b", "ByteTrack"),
        (r"\bRe-?ID\b", "Re-ID"),
        (r"\bMSN-?YOLO\b", "MSN-YOLO"),
        (r"\bC2f_MLCA\b", "C2f_MLCA"),
        (r"\bDetect_SEAM\b", "Detect_SEAM"),
        (r"\bNWD\b", "NWD"),
    ],
}

GENERIC_PATTERNS = [
    re.compile(r"\b(ResNet-?\d*)\b", re.I),
    re.compile(r"\b(YOLOv?\d+\w*)\b", re.I),
    re.compile(r"\b(ByteTrack|DeepSORT|FairMOT|StrongSORT)\b", re.I),
    re.compile(r"\b(XGBoost|LightGBM|CatBoost)\b", re.I),
    re.compile(r"\b(BERTScore)\b"),
    re.compile(r"\b(BERT|RoBERTa|XLNet)\b"),
    re.compile(r"\b(C2f_[A-Za-z0-9]+|Detect_[A-Za-z0-9]+|MSN-?YOLO)\b", re.I),
    re.compile(r"\b(Re-?ID)\b", re.I),
    re.compile(r"\b([A-Z]{2,5})\s+loss\b"),
    re.compile(
        r"\b(residual learning|residual blocks?|skip connections?|"
        r"identity (?:mappings?|connections?|shortcuts?)|bottlenecks?|"
        r"batch norm(?:alization)?|dropout|degradation problem)\b",
        re.I,
    ),
    re.compile(r"\b(gradient boosting|decision trees?|feature importance|regularization)\b", re.I),
    # "attention" alone is too vague for detection PDFs — keep only with NLP lexicon
    re.compile(r"\b(TF-?IDF|tokenizer|cosine similarity|greedy matching)\b", re.I),
]

PROPOSE_PATTERNS = [
    re.compile(
        r"(?i)\bwe (?:propose|introduce|present)\s+(?:a\s+|an\s+|the\s+)?"
        r"(?:(?:innovative|improved|novel|proposed|new)\s+)?"
        r"([A-Za-z][A-Za-z0-9_-]*(?:Score|Net|YOLO|BERT|Metric)?)",
    ),
    re.compile(
        r"(?i)\b(?:called|named)\s+([A-Za-z][A-Za-z0-9_-]*(?:Score|Net|YOLO)?)\b",
    ),
]


def _norm(token: str) -> str:
    token = re.sub(r"\s+", " ", token).strip(" .,;:()[]\"'")
    low = token.lower()
    aliases = {
        "residual blocks": "residual block",
        "residual block": "residual block",
        "residual learning": "residual learning",
        "skip connections": "skip connection",
        "skip connection": "skip connection",
        "identity mappings": "identity shortcut",
        "identity connections": "identity shortcut",
        "identity connection": "identity shortcut",
        "identity shortcuts": "identity shortcut",
        "identity shortcut": "identity shortcut",
        "identity mapping": "identity shortcut",
        "bottlenecks": "bottleneck",
        "bottleneck": "bottleneck",
        "batch normalisation": "batch normalization",
        "batch normalization": "batch normalization",
        "batch norm": "batch normalization",
        "degradation problem": "degradation problem",
        "re-id": "Re-ID",
        "reid": "Re-ID",
        "decision trees": "decision tree",
        "idf": "IDF weighting",
        "idf weighting": "IDF weighting",
        "idf importance": "IDF weighting",
        "contextual embedding": "contextual embeddings",
        "contextual embeddings": "contextual embeddings",
        "masked language modeling": "masked language modeling",
        "bertscore": "BERTScore",
        "cosine similarity": "cosine similarity",
        "greedy matching": "greedy matching",
        "baseline rescaling": "baseline rescaling",
        "baseline rescale": "baseline rescaling",
    }
    if low in aliases:
        return aliases[low]
    if re.fullmatch(r"nwd(?:\s+loss)?", low):
        return "NWD"
    if low.startswith("resnet"):
        m = re.search(r"(\d+)", token)
        return f"ResNet-{m.group(1)}" if m else "ResNet"
    return token


def _exact_key(name: str) -> str:
    """Exact token match key — never substring."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _infer_role(name: str, proposed_keys: set[str]) -> str:
    key = _exact_key(name)
    if key in proposed_keys:
        return "proposed"
    if key in BACKBONE_NAMES or re.match(r"(?i)^(?:bert|resnet|yolo|whisper)", name):
        return "backbone"
    return "technique"


def extract_proposed_entities(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pat in PROPOSE_PATTERNS:
        for m in pat.finditer(text[:8000]):
            raw = _norm(m.group(1))
            key = _exact_key(raw)
            if not raw or key in GENERIC_NOUNS or key in STOP or len(raw) < 3:
                continue
            if key in seen:
                continue
            seen.add(key)
            found.append(raw)
    # Title-like *Score names
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9]*Score)\b", text[:4000]):
        raw = _norm(m.group(1))
        key = _exact_key(raw)
        if key not in seen and key not in GENERIC_NOUNS:
            seen.add(key)
            found.insert(0, raw)
    return found


def extract_component_items(
    text: str,
    task_type: str = "unknown",
    contribution: str = "new_model",
    limit: int = 14,
) -> list[ComponentItem]:
    if not text:
        return []
    template = get_template_by_type(task_type)
    items: list[ComponentItem] = []
    seen: set[str] = set()
    proposed = extract_proposed_entities(text)
    proposed_keys = {_exact_key(p) for p in proposed}

    def add(token: str, role: str | None = None) -> None:
        token = _norm(token)
        if not token or len(token) < 2 or len(token) > 48:
            return
        key = _exact_key(token)
        if key in STOP or key in DATASET_NAMES or key in GENERIC_NOUNS:
            return
        if any(key == d or key.startswith(d + "-") for d in DATASET_NAMES):
            return
        if key in RELATED_WORK_ONLY and not re.search(
            rf"(?i)we\s+\w+.{{0,20}}{re.escape(token)}", text[:8000]
        ):
            return
        if key in seen:
            return
        # Exact-token dedupe only (BERTScore must not drop because BERT exists)
        seen.add(key)
        r = role or _infer_role(token, proposed_keys)
        items.append(ComponentItem(token, r))

    # #1: always put "we propose X" first
    for p in proposed:
        add(p, "proposed")

    if contribution == "new_metric":
        for pat, canon in METRIC_LEXICON:
            if re.search(pat, text[:16000], flags=re.I):
                add(canon, "proposed" if "score" in canon.lower() or canon == "BERTScore" else "technique")
        for pat, canon in OPERATION_LEXICON:
            if re.search(pat, text[:16000], flags=re.I):
                add(canon, "technique")

    for pat in GENERIC_PATTERNS:
        for m in pat.finditer(text[:16000]):
            add(m.group(1))

    for pat, canon in TASK_TERM_SCAN.get(task_type, []):
        for m in re.finditer(pat, text[:16000], flags=re.I):
            if canon is None:
                add(m.group(0))
            elif "{g1}" in canon:
                add(canon.replace("{g1}", m.group(1)))
            else:
                add(canon)

    for term in template.component_lexicon:
        if re.search(rf"(?i)(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text):
            add(term)

    if task_type == "image_classification" and re.search(
        r"(?i)\bresidual (?:network|net|learning)s?\b", text
    ):
        add("ResNet", "backbone")
        add("residual learning", "technique")

    for m in re.finditer(r"\b(18|34|50|101|152)(?:-|\s)?layer\b", text[:12000], flags=re.I):
        add(f"ResNet-{m.group(1)}", "backbone")

    # Prefer proposed → technique → backbone ordering within limit
    order = {"proposed": 0, "technique": 1, "backbone": 2}
    items.sort(key=lambda x: (order.get(x.role, 9), 0 if x.name in proposed else 1))
    # Re-dedupe after sort keep first
    final: list[ComponentItem] = []
    seen2: set[str] = set()
    for it in items:
        k = _exact_key(it.name)
        if k in seen2:
            continue
        seen2.add(k)
        final.append(it)

    # Drop names that only appear inside a longer component (BERT inside BERTScore)
    names = [it.name for it in final]
    kept: list[ComponentItem] = []
    for it in final:
        # Skip generic residual nouns on detection papers
        if task_type == "object_detection_tracking" and it.name.lower() in {
            "bottleneck", "attention", "fpn", "nms",
        }:
            continue
        longer = [n for n in names if n.lower() != it.name.lower() and it.name.lower() in n.lower()]
        if longer:
            stripped = text[:16000]
            for L in longer:
                stripped = re.sub(rf"(?i){re.escape(L)}", " ", stripped)
            if not re.search(
                rf"(?i)(?<![A-Za-z0-9]){re.escape(it.name)}(?![A-Za-z0-9])", stripped
            ):
                continue
        # Versioned family: keep YOLOv8n over bare YOLOv8 / YOLOv8 over YOLOv5 parent
        if _is_shadowed_version(it.name, names):
            continue
        # Also require whole-word presence in original text
        if not re.search(
            rf"(?i)(?<![A-Za-z0-9]){re.escape(it.name)}(?![A-Za-z0-9])", text[:16000]
        ):
            continue
        kept.append(it)
        if len(kept) >= limit:
            break
    return kept


def _is_shadowed_version(name: str, names: list[str]) -> bool:
    """True if a more specific sibling exists (YOLOv8 when YOLOv8n is present)."""
    low = name.lower()
    m = re.match(r"(?i)^(yolo)v?(\d+)([a-z]*)$", name.replace("-", ""))
    if not m:
        m = re.match(r"(?i)^(resnet)-?(\d+)$", name)
        if not m:
            return False
        family, ver = m.group(1).lower(), m.group(2)
        # Prefer ResNet-18 over bare ResNet when numbered exists
        if family == "resnet" and not re.search(r"\d", name):
            return any(re.match(r"(?i)^resnet-?\d+", n) for n in names)
        return False
    family, ver, suffix = m.group(1).lower(), m.group(2), (m.group(3) or "").lower()
    for other in names:
        if other.lower() == low:
            continue
        om = re.match(r"(?i)^yolov?(\d+)([a-z]*)$", other.replace("-", ""))
        if not om:
            continue
        over, osuf = om.group(1), (om.group(2) or "").lower()
        if over != ver:
            continue
        # Drop parent when child with extra suffix exists (v8 vs v8n)
        if not suffix and osuf:
            return True
        # Drop shorter suffix when longer specific exists
        if suffix and osuf and osuf != suffix and osuf.startswith(suffix):
            return True
    return False


def extract_components(
    text: str,
    task_type: str = "unknown",
    contribution: str = "new_model",
    limit: int = 14,
) -> list[str]:
    return [c.name for c in extract_component_items(text, task_type, contribution, limit)]


def components_blurb(components: list[str] | list[ComponentItem]) -> str:
    if not components:
        return "No named modules clearly detected — describe the pipeline stages in your own words."
    names = [c.name if isinstance(c, ComponentItem) else c for c in components]
    return ", ".join(names)
