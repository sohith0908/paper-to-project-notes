"""Gold-set scoring: task accuracy, component P/R, metrics exact-match."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PaperScore:
    paper_id: str
    task_correct: bool
    task_in_top2: bool
    component_precision: float
    component_recall: float
    metrics_exact: float
    dataset_hit: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldReport:
    n: int
    task_accuracy: float
    task_top2_accuracy: float
    mean_component_precision: float
    mean_component_recall: float
    mean_metrics_exact: float
    mean_dataset_hit: float
    per_paper: list[PaperScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def _token_set(items: list[str]) -> set[str]:
    return {_norm(x) for x in items if _norm(x)}


def score_components(pred: list[str], gold: list[str]) -> tuple[float, float]:
    p, g = _token_set(pred), _token_set(gold)
    if not p and not g:
        return 1.0, 1.0
    if not p:
        return 0.0, 0.0 if g else 1.0
    if not g:
        return 0.0, 1.0
    inter = p & g
    precision = len(inter) / len(p)
    recall = len(inter) / len(g)
    return precision, recall


def score_metrics(pred: str, gold_metrics: list[str]) -> float:
    """Exact-ish match: fraction of gold metric strings found in pred text."""
    if not gold_metrics:
        # gold says none — credit if tool says not found / typical
        pl = (pred or "").lower()
        return 1.0 if ("not found" in pl or "typical for this task" in pl) else 0.0
    hits = 0
    pl = (pred or "").lower()
    for m in gold_metrics:
        if m.lower() in pl or _norm(m) in _norm(pl):
            hits += 1
    return hits / len(gold_metrics)


def score_dataset(pred: str, gold_datasets: list[str]) -> float:
    if not gold_datasets:
        return 1.0 if "no named dataset" in (pred or "").lower() else 0.5
    hits = sum(1 for d in gold_datasets if d.lower() in (pred or "").lower())
    return hits / len(gold_datasets)


def score_paper(gold: dict[str, Any], pred: dict[str, Any]) -> PaperScore:
    task_pred = pred.get("task_type")
    task_gold = gold.get("task_type")
    cands = pred.get("task_candidates") or []
    cand_types = []
    for c in cands:
        if isinstance(c, dict):
            cand_types.append(c.get("task_type"))
        else:
            cand_types.append(getattr(c, "task_type", None))
    prec, rec = score_components(pred.get("components") or [], gold.get("components") or [])
    return PaperScore(
        paper_id=str(gold.get("id")),
        task_correct=task_pred == task_gold,
        task_in_top2=task_gold in cand_types[:2] or task_pred == task_gold,
        component_precision=prec,
        component_recall=rec,
        metrics_exact=score_metrics(pred.get("metrics") or "", gold.get("metrics") or []),
        dataset_hit=score_dataset(pred.get("dataset") or "", gold.get("datasets") or []),
        details={"pred_task": task_pred, "gold_task": task_gold, "cand": cand_types[:2]},
    )


def evaluate_gold(gold_dir: Path, predict_fn) -> GoldReport:
    papers = sorted(gold_dir.glob("*.json"))
    scores: list[PaperScore] = []
    for path in papers:
        gold = json.loads(path.read_text(encoding="utf-8"))
        notes = predict_fn(gold["text"])
        if hasattr(notes, "to_dict"):
            pred = notes.to_dict()
        else:
            pred = dict(notes)
        # normalize task_candidates
        cands = pred.get("task_candidates") or []
        norm_cands = []
        for c in cands:
            if isinstance(c, dict):
                norm_cands.append(c)
            else:
                norm_cands.append({"task_type": getattr(c, "task_type", ""), "score": getattr(c, "score", 0)})
        pred["task_candidates"] = norm_cands
        scores.append(score_paper(gold, pred))
    n = max(1, len(scores))
    return GoldReport(
        n=len(scores),
        task_accuracy=sum(s.task_correct for s in scores) / n,
        task_top2_accuracy=sum(s.task_in_top2 for s in scores) / n,
        mean_component_precision=sum(s.component_precision for s in scores) / n,
        mean_component_recall=sum(s.component_recall for s in scores) / n,
        mean_metrics_exact=sum(s.metrics_exact for s in scores) / n,
        mean_dataset_hit=sum(s.dataset_hit for s in scores) / n,
        per_paper=scores,
    )
