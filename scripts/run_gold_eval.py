"""Run gold-set evaluation and print a scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_to_project.eval_gold import evaluate_gold
from paper_to_project.extractor import extract_paper_notes
from paper_to_project.gold_data import write_gold_set


def main() -> None:
    gold = write_gold_set()
    report = evaluate_gold(gold, lambda t: extract_paper_notes(t, prefer_llm=False))
    out = ROOT / "gold" / "last_report.json"
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"n={report.n}")
    print(f"task_accuracy={report.task_accuracy:.3f}")
    print(f"task_top2_accuracy={report.task_top2_accuracy:.3f}")
    print(f"component_precision={report.mean_component_precision:.3f}")
    print(f"component_recall={report.mean_component_recall:.3f}")
    print(f"metrics_exact={report.mean_metrics_exact:.3f}")
    print(f"dataset_hit={report.mean_dataset_hit:.3f}")
    print(f"wrote {out}")
    fails = [s for s in report.per_paper if not s.task_correct]
    if fails:
        print("task misses:")
        for s in fails:
            print(f"  {s.paper_id}: pred={s.details.get('pred_task')} gold={s.details.get('gold_task')} cand={s.details.get('cand')}")


if __name__ == "__main__":
    main()
