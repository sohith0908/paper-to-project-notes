"""Task-type scoring + per-task templates. No cross-task leakage.

Returns top-k candidates for multi-task papers (e.g. BERT, ResNet+COCO).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TaskScore:
    task_type: str
    score: int
    matched: list[str] = field(default_factory=list)

    @property
    def normalized(self) -> float:
        # Soft 0..1 for UI (not a calibrated probability)
        return round(min(1.0, self.score / 20.0), 3)


@dataclass
class TaskTemplate:
    task_type: str
    label: str
    stack: list[str]
    python_scope: list[str]
    out_of_scope: list[str]
    typical_metrics: list[str] = field(default_factory=list)
    viva_extras: list[tuple[str, str]] = field(default_factory=list)
    component_lexicon: list[str] = field(default_factory=list)
    # Feasibility hints for status engine
    compute_tier: str = "low"  # low | medium | high | extreme


UNKNOWN_THRESHOLD = 4

TASK_KEYWORDS: dict[str, list[tuple[str, int]]] = {
    "object_detection_tracking": [
        (r"\bobject detection\b", 5),
        (r"\bmultiple object tracking\b", 6),
        (r"\bbytetrack\b", 6),
        (r"\bdeepsort\b", 5),
        (r"\byolov?\d*\w*\b", 5),
        (r"\bbounding box(?:es)?\b", 4),
        (r"\bmota\b", 5),
        (r"\bmotp\b", 4),
        (r"\bidf1\b", 4),
        (r"\bvehicle (?:flow|detection|tracking)\b", 5),
        (r"\bre-?id\b", 3),
        (r"\btracker\b", 3),
        (r"\bcoco detection\b", 4),
    ],
    "image_classification": [
        (r"\bimage classification\b", 6),
        (r"\bresnet-?\d*\b", 6),
        (r"\bresidual (?:network|learning|block|connection)s?\b", 6),
        (r"\bskip connection\b", 4),
        (r"\bbottleneck\b", 3),
        (r"\bimagenet\b", 3),  # weaker alone (also used by detection)
        (r"\bcifar-?(?:10|100)?\b", 4),
        (r"\btop-?[15]\b", 5),
        (r"\bdegradation problem\b", 5),
        (r"\befficientnet\b", 4),
        (r"\bvision transformer\b", 3),
        (r"\bcnn\b", 2),
        (r"\bbatch norm(?:alization)?\b", 2),
    ],
    "semantic_segmentation": [
        (r"\bsemantic segmentation\b", 7),
        (r"\binstance segmentation\b", 7),
        (r"\bpixel[- ]wise\b", 4),
        (r"\bune[t-]?\b", 5),
        (r"\bmask r-?cnn\b", 6),
        (r"\bdeeplab\b", 5),
        (r"\bmIoU\b", 5),
        (r"\bcityscapes\b", 4),
        (r"\bpascale? voc.*segmen", 3),
    ],
    "generative_models": [
        (r"\bgenerative adversarial\b", 6),
        (r"\b\bGAN\b", 5),
        (r"\bdiffusion model\b", 7),
        (r"\bstable diffusion\b", 6),
        (r"\bdenoising diffusion\b", 6),
        (r"\bvae\b", 3),
        (r"\bgenerator and discriminator\b", 5),
        (r"\bfid\b", 4),
        (r"\binception score\b", 4),
        (r"\btext-to-image\b", 5),
    ],
    "reinforcement_learning": [
        (r"\breinforcement learning\b", 7),
        (r"\b\bRL\b", 3),
        (r"\bpolicy gradient\b", 5),
        (r"\bq-?learning\b", 5),
        (r"\bppo\b", 5),
        (r"\bdqn\b", 5),
        (r"\bactor-critic\b", 5),
        (r"\breward function\b", 4),
        (r"\bopenai gym\b", 4),
        (r"\bmarkov decision\b", 4),
    ],
    "graph_ml": [
        (r"\bgraph neural\b", 7),
        (r"\b\bGNN\b", 5),
        (r"\b\bGCN\b", 5),
        (r"\bgraph convolution\b", 5),
        (r"\bnode classification\b", 5),
        (r"\blink prediction\b", 5),
        (r"\bmessage passing\b", 4),
        (r"\bciteseer\b", 3),
        (r"\bpubmed\b", 2),
        (r"\bknowledge graph\b", 3),
    ],
    "speech": [
        (r"\bspeech recognition\b", 7),
        (r"\bautomatic speech\b", 6),
        (r"\basr\b", 4),
        (r"\bwhisper\b", 5),
        (r"\bmel[- ]spectrogram\b", 4),
        (r"\bword error rate\b", 6),
        (r"\b\bWER\b", 4),
        (r"\blibrispeech\b", 5),
        (r"\btext-to-speech\b", 5),
        (r"\btts\b", 3),
    ],
    "nlp_text": [
        (r"\bnlp\b", 3),
        (r"\bsentiment\b", 4),
        (r"\btext classification\b", 5),
        (r"\bnamed entity\b", 5),
        (r"\bquestion answering\b", 5),
        (r"\bbertscore\b", 7),
        (r"\bbert\b", 5),
        (r"\btokenizer\b", 3),
        (r"\btf-?idf\b", 4),
        (r"\bglue\b", 4),
        (r"\bsquad\b", 4),
        (r"\btransformer\b", 2),
        (r"\bmasked language\b", 5),
        (r"\bevaluation metric\b", 5),
        (r"\bmachine translation\b", 4),
        (r"\bhuman judgments?\b", 4),
        (r"\bparaphrase detection\b", 4),
        (r"\bBLEU\b", 3),
        (r"\bwmt\d{0,2}\b", 3),
    ],
    "time_series": [
        (r"\btime[- ]series\b", 6),
        (r"\bforecast(?:ing)?\b", 4),
        (r"\barima\b", 5),
        (r"\blstm\b", 2),
        (r"\bstock prediction\b", 4),
    ],
    "recommender": [
        (r"\brecommender\b", 6),
        (r"\brecommendation system\b", 6),
        (r"\bcollaborative filtering\b", 5),
        (r"\bmatrix factorization\b", 4),
    ],
    "classic_tabular_ml": [
        (r"\bxgboost\b", 7),
        (r"\bgradient boosting\b", 5),
        (r"\brandom forest\b", 5),
        (r"\blogistic regression\b", 4),
        (r"\btabular\b", 4),
        (r"\bfeature importance\b", 3),
        (r"\bdecision tree\b", 3),
        (r"\blightgbm\b", 5),
        (r"\bcatboost\b", 5),
        (r"\bsvm\b", 2),
    ],
}


def _base(task: str, label: str, **kwargs) -> TaskTemplate:
    return TaskTemplate(task_type=task, label=label, **kwargs)


TEMPLATES: dict[str, TaskTemplate] = {
    "object_detection_tracking": _base(
        "object_detection_tracking",
        "Object detection / multi-object tracking",
        stack=["Python 3", "Ultralytics YOLO / PyTorch", "OpenCV", "NumPy", "Matplotlib", "Streamlit"],
        python_scope=[
            "Use a pretrained small detector on a short public video/image set.",
            "Optionally add a lightweight tracker for ID continuity.",
            "Report mAP/precision/recall when labels exist; demo detections.",
        ],
        out_of_scope=["Full custom detector training at paper scale.", "Claiming paper mAP/MOTA without same protocol."],
        typical_metrics=["mAP50", "mAP", "MOTA", "IDF1", "FPS"],
        viva_extras=[("Why detection and tracking are both needed?", "Detection finds objects; tracking links IDs across frames.")],
        component_lexicon=["YOLOv8n", "ByteTrack", "Re-ID", "MSN-YOLO", "NWD"],
        compute_tier="medium",
    ),
    "image_classification": _base(
        "image_classification",
        "Image classification",
        stack=["Python 3", "PyTorch", "torchvision", "Matplotlib", "Streamlit"],
        python_scope=[
            "Fine-tune pretrained ResNet-18 on CIFAR-10 or a tiny ImageNet-style subset.",
            "Compare to a shallow CNN; report top-1 accuracy + confusion matrix.",
            "Optional tiny residual block demo; Streamlit image upload.",
        ],
        out_of_scope=["Train ResNet-152 from scratch on full ImageNet.", "Claim paper top-1/top-5 numbers."],
        typical_metrics=["top-1 error", "top-5 error", "accuracy"],
        viva_extras=[("What is the degradation problem?", "Deeper plain nets can worsen training error; residuals ease optimization.")],
        component_lexicon=["ResNet", "residual learning", "residual block", "skip connection", "identity shortcut", "bottleneck", "batch normalization", "degradation problem"],
        compute_tier="medium",
    ),
    "semantic_segmentation": _base(
        "semantic_segmentation",
        "Semantic / instance segmentation",
        stack=["Python 3", "PyTorch", "segmentation-models / torchvision", "OpenCV", "Streamlit"],
        python_scope=[
            "Fine-tune a small U-Net / DeepLab head on a tiny Cityscapes/CamVid-style subset.",
            "Report mIoU; visualize color masks.",
            "Keep resolution and epochs small for a student GPU/CPU budget.",
        ],
        out_of_scope=["Full Cityscapes SOTA training.", "Mask R-CNN from scratch on COCO."],
        typical_metrics=["mIoU", "pixel accuracy", "Dice"],
        component_lexicon=["U-Net", "DeepLab", "Mask R-CNN", "encoder-decoder", "skip connection"],
        compute_tier="high",
    ),
    "generative_models": _base(
        "generative_models",
        "Generative models (GAN / diffusion)",
        stack=["Python 3", "PyTorch", "diffusers or torchvision", "Matplotlib", "Streamlit"],
        python_scope=[
            "Run a tiny pretrained diffusion/GAN demo (or train a toy DCGAN on a small image set).",
            "Report qualitative samples; optional FID on a tiny set if feasible.",
            "Document compute limits vs paper-scale training.",
        ],
        out_of_scope=["Train Stable Diffusion / StyleGAN from scratch.", "Multi-GPU weeks-long runs."],
        typical_metrics=["FID", "Inception Score", "CLIP score"],
        viva_extras=[("Why approximate instead of full training?", "Paper-scale generative training needs extreme compute.")],
        component_lexicon=["GAN", "generator", "discriminator", "diffusion", "U-Net denoiser", "VAE"],
        compute_tier="extreme",
    ),
    "reinforcement_learning": _base(
        "reinforcement_learning",
        "Reinforcement learning",
        stack=["Python 3", "PyTorch or Stable-Baselines3", "Gymnasium", "Matplotlib", "Streamlit"],
        python_scope=[
            "Train a small PPO/DQN agent on a classic Gymnasium env (CartPole/LunarLander).",
            "Plot reward curves; keep timesteps modest.",
            "Compare against a random baseline.",
        ],
        out_of_scope=["Large-scale robotics sim clusters.", "Claiming paper game-benchmark scores."],
        typical_metrics=["episodic reward", "success rate", "sample efficiency"],
        component_lexicon=["PPO", "DQN", "policy network", "replay buffer", "reward"],
        compute_tier="medium",
    ),
    "graph_ml": _base(
        "graph_ml",
        "Graph neural networks",
        stack=["Python 3", "PyTorch Geometric or DGL", "NetworkX", "Matplotlib", "Streamlit"],
        python_scope=[
            "Run a small GCN on Cora/CiteSeer-style node classification.",
            "Report accuracy; visualize embeddings or graph sample.",
        ],
        out_of_scope=["Billion-edge industrial graphs.", "Full paper multi-GPU GNN pretraining."],
        typical_metrics=["accuracy", "F1", "MRR", "Hits@K"],
        component_lexicon=["GCN", "GNN", "message passing", "graph convolution", "node embedding"],
        compute_tier="medium",
    ),
    "speech": _base(
        "speech",
        "Speech / ASR / TTS",
        stack=["Python 3", "torchaudio or Whisper", "librosa", "Matplotlib", "Streamlit"],
        python_scope=[
            "Use a pretrained ASR (e.g. tiny Whisper) on a few public clips.",
            "Report WER on a tiny labeled set if available; demo upload→transcript.",
        ],
        out_of_scope=["Train ASR from scratch on LibriSpeech full.", "Multi-week TTS voice cloning farms."],
        typical_metrics=["WER", "CER", "MOS"],
        component_lexicon=["ASR", "mel-spectrogram", "encoder-decoder", "CTC", "Whisper"],
        compute_tier="high",
    ),
    "nlp_text": _base(
        "nlp_text",
        "NLP / text understanding",
        stack=["Python 3", "pandas", "scikit-learn or transformers", "Streamlit"],
        python_scope=[
            "Baseline TF-IDF + linear model, optional small BERT fine-tune on a short dataset.",
            "Report accuracy/F1; demo paste-text prediction.",
        ],
        out_of_scope=["Full LLM/BERT pretraining from scratch.", "Entire GLUE mega-suite at paper scale."],
        typical_metrics=["accuracy", "precision", "recall", "F1"],
        component_lexicon=[
            "BERT", "BERTScore", "transformer", "attention", "tokenizer", "TF-IDF",
            "masked language modeling", "cosine similarity", "greedy matching", "IDF weighting",
        ],
        compute_tier="medium",
    ),
    "time_series": _base(
        "time_series",
        "Time-series forecasting",
        stack=["Python 3", "pandas", "scikit-learn / statsmodels", "Matplotlib", "Streamlit"],
        python_scope=["Lag features + ARIMA/ridge baseline; plot forecast vs actual; report MAE/RMSE."],
        out_of_scope=["Realtime production forecasting at company scale."],
        typical_metrics=["MAE", "RMSE", "MAPE"],
        component_lexicon=["ARIMA", "LSTM", "lag features"],
        compute_tier="low",
    ),
    "recommender": _base(
        "recommender",
        "Recommender system",
        stack=["Python 3", "pandas", "scikit-learn / Surprise", "Streamlit"],
        python_scope=["Popularity baseline + collaborative filtering on a public ratings set; Precision@K / RMSE."],
        out_of_scope=["Industrial ranking stacks / online A/B."],
        typical_metrics=["RMSE", "Precision@K", "NDCG"],
        component_lexicon=["collaborative filtering", "matrix factorization"],
        compute_tier="low",
    ),
    "classic_tabular_ml": _base(
        "classic_tabular_ml",
        "Classic tabular ML / gradient boosting",
        stack=["Python 3", "pandas", "scikit-learn", "XGBoost or LightGBM", "Matplotlib", "Streamlit"],
        python_scope=["EDA + logistic/RF baseline + XGBoost; metrics + feature importance demo."],
        out_of_scope=["Claiming paper SOTA without matching protocol."],
        typical_metrics=["accuracy", "AUC", "logloss", "RMSE", "MAE"],
        component_lexicon=["XGBoost", "gradient boosting", "decision tree", "regularization", "feature importance"],
        compute_tier="low",
    ),
    "unknown": _base(
        "unknown",
        "Unknown / study (confirm task)",
        stack=["Python 3", "Markdown notes", "Matplotlib (optional)", "Streamlit (optional)"],
        python_scope=[
            "Confirm the paper’s task from the abstract (do not force a template).",
            "Restate problem and method from the text only.",
            "If a named algorithm appears, thin public-data demo; otherwise a study report.",
            "Document missing inputs; never invent datasets or metrics.",
        ],
        out_of_scope=[
            "Guessing a detection/YOLO stack when unrelated.",
            "Claiming paper numbers without the same protocol.",
        ],
        typical_metrics=[],
        viva_extras=[("What task type is this paper?", "Cite abstract evidence; do not force a template.")],
        compute_tier="low",
    ),
    "general_ml": _base(
        "general_ml",
        "General applied ML prototype",
        stack=["Python 3", "pandas", "scikit-learn or PyTorch", "Matplotlib", "Streamlit"],
        python_scope=[
            "Restate the paper problem and method from the provided text.",
            "Smaller prototype with public data; baseline + one paper-inspired improvement.",
            "Report metrics named in the paper on a student-scale set.",
            "Do not claim paper-scale numbers without the same protocol.",
        ],
        out_of_scope=["Full paper-scale replication.", "Inventing metrics not in the text."],
        typical_metrics=["accuracy", "F1", "RMSE"],
        compute_tier="low",
    ),
}


def score_tasks(text: str) -> list[TaskScore]:
    t = text or ""
    scores: list[TaskScore] = []
    for task, rules in TASK_KEYWORDS.items():
        total = 0
        matched: list[str] = []
        for pat, w in rules:
            if re.search(pat, t, flags=re.IGNORECASE):
                total += w
                matched.append(pat)
        scores.append(TaskScore(task, total, matched))
    scores.sort(key=lambda s: (-s.score, s.task_type))
    return scores


def top_task_candidates(text: str, k: int = 2) -> list[TaskScore]:
    """Return strong candidates only; zeros are not padded as fake runner-ups."""
    scores = score_tasks(text)
    positive = [s for s in scores if s.score > 0]
    if not positive:
        return [TaskScore("unknown", 0, [])]
    top = positive[0].score
    strong = [s for s in positive if s.score >= max(UNKNOWN_THRESHOLD, top * 0.45)]
    out = list(strong or positive)[:k]
    if out[0].score < UNKNOWN_THRESHOLD:
        return [TaskScore("unknown", out[0].score, out[0].matched)]
    return out[:k]


def detect_task_type(text: str) -> str:
    cands = top_task_candidates(text, k=2)
    return cands[0].task_type if cands else "unknown"


def detect_paper_type(text: str) -> str:
    t = text.lower()
    survey_hits = len(
        re.findall(
            r"\b(survey|literature review|taxonomy|overview of|systematic review|"
            r"state of the art review|we review)\b",
            t,
        )
    )
    method_hits = len(
        re.findall(
            r"\b(we propose|we present a novel|this paper proposes|our method|"
            r"we introduce|we develop)\b",
            t,
        )
    )
    app_hits = len(re.findall(r"\b(case study|application to|deployed)\b", t))
    theory_hits = len(
        re.findall(r"\b(theorem|lemma|we prove|theoretical analysis|convergence proof)\b", t)
    )
    if survey_hits >= 1 and survey_hits >= method_hits:
        return "survey"
    if theory_hits >= 2 and method_hits == 0:
        return "theory"
    if method_hits >= 1:
        return "method"
    if app_hits >= 1:
        return "application"
    if theory_hits >= 1:
        return "theory"
    return "method"


def get_template(text: str) -> TaskTemplate:
    return TEMPLATES[detect_task_type(text)]


def get_template_by_type(task_type: str) -> TaskTemplate:
    return TEMPLATES.get(task_type) or TEMPLATES["unknown"]
