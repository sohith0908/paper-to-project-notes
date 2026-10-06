"""Pytest — lock student-facing quality so regressions cannot return."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from paper_to_project.eval_gold import evaluate_gold
from paper_to_project.extractor import extract_paper_notes
from paper_to_project.gold_data import write_gold_set
from paper_to_project.proposal import build_proposal, build_viva_notes, build_markdown_bundle
from paper_to_project.datasets import suggest_substitute
from paper_to_project.templates import top_task_candidates
from paper_to_project.linter import lint_markdown
from paper_to_project.sentences import is_result_sentence
from paper_to_project.verify import BANNED_IN_STUDENT, _problem_words_ok


ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "tests" / "snapshots"

BERTSCORE = """BERTScore: Evaluating Text Generation with BERT
Abstract
We propose BERTScore, an automatic evaluation metric for text generation. Existing metrics rely on exact matches and fail to capture paraphrases. Instead of exact matches, we compute similarity using contextual embeddings from pretrained BERT with greedy matching and optional IDF weighting. BERTScore correlates better with human judgments than BLEU and METEOR on WMT16 metrics shared tasks, COCO Captions, and adversarial paraphrase detection (PAWS). Finally, we use baseline rescaling. Pearson and Kendall correlations are reported.
"""


@pytest.fixture(scope="session")
def gold_dir():
    return write_gold_set()


def test_gold_set_size(gold_dir):
    assert len(list(gold_dir.glob("*.json"))) >= 29


def test_gold_task_accuracy_floor(gold_dir):
    report = evaluate_gold(gold_dir, lambda t: extract_paper_notes(t, prefer_llm=False))
    assert report.task_accuracy >= 0.70, report.to_dict()
    assert report.task_top2_accuracy >= 0.78, report.to_dict()
    assert report.mean_component_recall >= 0.30, report.to_dict()


def test_no_yolo_leak_on_tabular():
    n = extract_paper_notes(
        "XGBoost for Fraud\nAbstract\nWe apply XGBoost gradient boosting on a tabular fraud dataset and report AUC.\n",
        prefer_llm=False,
    )
    blob = " ".join(n.suggested_stack + n.python_scope + n.out_of_scope).lower()
    assert "yolo" not in blob
    assert n.task_type == "classic_tabular_ml"


def test_survey_method_one_line():
    n = extract_paper_notes(
        "Deep Residual Networks: A Survey\nAbstract\nWhy residual networks work is not fully understood yet. This survey reviews ResNet variants.\nImageNet and CIFAR-10 are common. No new method is proposed.\n",
        prefer_llm=False,
    )
    assert n.paper_type == "survey"
    assert n.method.startswith("Survey of residual")


def test_multitask_top2_bert():
    cands = top_task_candidates(
        "We fine-tune BERT for text classification, named entity recognition, and question answering on GLUE and SQuAD.",
        k=2,
    )
    assert cands[0].task_type == "nlp_text"


def test_new_task_types_routing():
    cases = [
        ("U-Net semantic segmentation mIoU Cityscapes", "semantic_segmentation"),
        ("denoising diffusion model FID generative adversarial", "generative_models"),
        ("PPO reinforcement learning Gym episodic reward", "reinforcement_learning"),
        ("GCN graph neural network node classification Cora", "graph_ml"),
        ("ASR speech recognition LibriSpeech WER mel-spectrogram", "speech"),
    ]
    for text, expect in cases:
        n = extract_paper_notes(f"Title\nAbstract\n{text}\n", prefer_llm=False)
        assert n.task_type == expect, (text, n.task_type)


def test_evidence_present():
    n = extract_paper_notes(
        "Toy CNN\nAbstract\nWe propose a CNN for image classification on CIFAR-10 with accuracy 0.9.\n",
        prefer_llm=False,
    )
    assert {e.field for e in n.evidence} >= {"task_type", "problem"}


def test_status_keeps_paper_component_names():
    n = extract_paper_notes(
        "Residual survey\nAbstract\nThis survey covers ResNet residual blocks and skip connections on ImageNet.\n",
        prefer_llm=False,
    )
    for row in n.status_table:
        assert row.name in n.components or row.name in (n.unverified_components or [])


def test_regression_snapshot_no_instruction_metrics():
    n = extract_paper_notes(
        "Optics notes\nAbstract\nWe document calibration drift. No ML model.\n",
        prefer_llm=False,
    )
    assert "look for evaluation" not in n.metrics.lower()
    viva = build_viva_notes(n)
    SNAP.mkdir(parents=True, exist_ok=True)
    path = SNAP / "unknown_optics_viva.txt"
    if not path.exists():
        path.write_text(viva, encoding="utf-8")
    assert "yolo" not in viva.lower()


def test_bertscore_student_output_locked():
    n = extract_paper_notes(BERTSCORE, prefer_llm=False)
    proposal = build_proposal(n)
    viva = build_viva_notes(n)
    bundle = build_markdown_bundle(n)

    # Problem: source words only, no invented wording
    assert not is_result_sentence(n.problem)
    assert "argues" not in n.problem.lower()
    assert "insufficient" not in n.problem.lower()
    assert _problem_words_ok(n.problem, BERTSCORE)
    assert "exact match" in n.problem.lower() or n.problem.startswith("Problem not stated")

    # BERT status/plan/compute aligned; plain reasons
    bert = next(r for r in n.status_table if r.name == "BERT")
    assert "fine-tun" not in bert.compute_note.lower()
    assert "no proving" not in bert.reason.lower()
    assert "inferred encoder load" not in bert.reason.lower()
    plan = " ".join(n.python_scope).lower()
    assert "without training" in plan or "no training" in plan
    assert bert.provenance in {"from_paper", "inferred"}
    # from_paper requires a proving span in evidence
    if bert.provenance == "from_paper":
        ev = next((e for e in n.evidence if e.field == "component" and e.value == "BERT"), None)
        assert ev and ev.source_sentence

    # Concrete technique steps — no vague "Implement contextual embeddings"
    assert not any(s.strip().lower() == "implement contextual embeddings." for s in n.python_scope)
    assert any("token" in s.lower() for s in n.python_scope)
    # Backbone already extracts tokens → no duplicate "token-level vectors" step
    assert sum(1 for s in n.python_scope if re.search(r"(?i)token-level vectors", s)) <= 1
    assert sum(1 for s in n.python_scope if re.search(r"(?i)extract token", s)) <= 1

    # Baseline step matches stack (BLEU → sacrebleu)
    assert any("bleu" in s.lower() for s in n.python_scope)
    assert "sacrebleu" in n.suggested_stack
    assert "none listed" not in " ".join(n.python_scope).lower()

    # Student docs: no developer wording
    for doc in (proposal, viva, bundle):
        low = doc.lower()
        assert "no proving span" not in low
        assert "inferred encoder load" not in low
        assert "# debug" not in low
        assert "[plan]" not in low
        assert ".;" not in doc
        assert ";." not in doc
        for pat in BANNED_IN_STUDENT:
            if pat in (r"(?i)\bverified\b",):
                continue
            assert not re.search(pat, doc), pat

    # Structural linter locks
    from paper_to_project.linter import lint_markdown
    hits = lint_markdown(
        proposal + "\n" + viva,
        task_type=n.task_type,
        contribution=n.contribution,
        source_text=BERTSCORE,
        problem=n.problem,
        reason_blob=n.problem + " " + n.method,
        notes=n,
    )
    assert not hits, hits

    # Timeline: goal+deliverable, week 4 report, no plan dump
    assert "| Week | Goal | Deliverable |" in proposal
    assert "viva" in proposal.lower().split("| 4 |")[-1].lower() or "report" in proposal.lower()

    # Objectives from plan
    assert "Restate the paper problem" in proposal or "2." in proposal.split("## 3. Objectives")[1][:400]

    # Viva: references proposal; limits from plan; no why without reason unless what-question
    assert "see" in viva.lower() and "proposal" in viva.lower()
    assert "small" in viva.lower() or "no encoder fine-tuning" in viva.lower() or "shared-task" in viva.lower()
    assert viva.lower().count("| component |") == 0  # table not reprinted
    # No trailing status-restating extras flooding Q10+
    assert viva.count("## Q") <= 12

    # Routing / missing printed once in proposal, not re-dumped as duplicate sections in viva
    assert viva.lower().count("still missing from the text") == 0


def test_problem_not_filled_with_result():
    n = extract_paper_notes(
        "Some Metric Paper\nAbstract\nWe propose CoolScore. CoolScore correlates better with human judgments on WMT16.\n",
        prefer_llm=False,
    )
    assert "correlates better" not in n.problem.lower()
    assert "Problem not stated" in n.problem or not is_result_sentence(n.problem)


def test_bert_not_only_substring_of_bertscore():
    n = extract_paper_notes(
        "Only BERTScore\nAbstract\nWe propose BERTScore as a metric. No separate encoder name appears.\n",
        prefer_llm=False,
    )
    assert "BERT" not in n.components


def test_why_requires_reason_cue():
    n = extract_paper_notes(BERTSCORE, prefer_llm=False)
    viva = build_viva_notes(n)
    # If a Why Q appears, surrounding method/problem must have a reason cue — otherwise generator uses What
    import re
    for m in re.finditer(r"## Q\d+\. (Why .+)\n(.+)", viva):
        q, a = m.group(1), m.group(2)
        assert re.search(r"(?i)because|so that|in order to|since", n.problem + n.method + a), q


def test_viva_linter_flags_placeholders():
    hits = lint_markdown(
        "Q11. 20-second definition; no proving span; # Debug",
        task_type="nlp_text",
        contribution="new_metric",
    )
    assert hits


# Grounded on public abstract of the MDPI vehicle-flow paper (regression signal, not a special-case).
VEHICLE_FLOW = """Vehicle Flow Detection and Tracking Based on an Improved YOLOv8n and ByteTrack Framework
Abstract
Vehicle flow detection and tracking are crucial components of intelligent transportation systems. However, traditional methods often struggle with challenges such as the poor detection of small objects and low efficiency when processing large-scale data. To address these issues, this paper proposes a vehicle flow detection and tracking method that integrates an improved YOLOv8n model with the ByteTrack algorithm. In the detection module, we introduce the innovative MSN-YOLO model, which combines the C2f_MLCA module, the Detect_SEAM module, and the NWD loss function to enhance feature fusion and improve cross-scale information processing. In the tracking module, we incorporate the ByteTrack algorithm and train unique vehicle re-identification (Re-ID) features. The experimental results demonstrate that the proposed method achieves a mean Average Precision (mAP) of 62.8% at IoU = 0.50 and a Multiple Object Tracking Accuracy (MOTA) of 72.16% in real-time tracking. These improvements represent increases of 2.7% and 3.16%, respectively, compared to baseline algorithms.

2.2 Normalized Wasserstein Distance
2 (1) However, the above equation is a distance measure and cannot be used directly for the calculation of similarity.
We use the normalized exponent to obtain a new metric called the normalized Wasserstein distance: NWD (µa, µb) = exp(−sqrt(W22)/C) (2)
For the NWD-based loss, we use the following: LNWD = 1− NWD (µa, µb) (3) 2.3.
Datasets include UA-DETRAC and MOT07. DeepSORT and YOLOv5 are discussed as prior work.
YOLOv8 is mentioned in related work; the student backbone is YOLOv8n.
MOT01 1.78K frames MOT01 1.23K frames
After mAP moves 50 1 introduction mAP 50 1.1
Vehicle improves mAP by 8
C2f_MLCA improves mAP by 0.7
"""


def test_metadata_only_title_gives_honest_notes():
    n = extract_paper_notes(
        "Some Obscure Theory Paper Title About Widgets\n\n"
        "Abstract not available from open metadata. "
        "Upload the PDF or paste the abstract for better notes.\n",
        prefer_llm=False,
    )
    assert n.input_tier == "metadata_only"
    assert "paste the abstract" in n.problem.lower() or "not stated" in n.problem.lower()
    assert "yolo" not in " ".join(n.suggested_stack).lower()
    assert any("metadata" in w.lower() or "abstract" in w.lower() for w in n.warnings)


def test_theory_paper_study_plan():
    n = extract_paper_notes(
        "On the Convergence of Optimistic Gradient Methods\n"
        "Abstract\n"
        "We prove a new theorem on the convergence rate of optimistic gradient descent. "
        "Our lemma shows that under Lipschitz assumptions the corollary holds. "
        "No experiments are reported.\n",
        prefer_llm=False,
    )
    assert n.paper_kind in {"theory", "unknown", "survey"} or n.paper_type == "theory"
    blob = " ".join(n.python_scope).lower()
    assert "theorem" in blob or "claim" in blob or "study" in blob or "prove" in n.method.lower() or "theoretical" in n.method.lower()
    assert "yolo" not in blob
    assert "do not claim" in " ".join(n.warnings).lower() or "protocol" in " ".join(n.out_of_scope).lower()


def test_non_ml_optics_no_yolo_stack():
    n = extract_paper_notes(
        "Optics notes\nAbstract\nWe document calibration drift in a spectrometer. "
        "No machine learning model is proposed. Instrumentation details only.\n",
        prefer_llm=False,
    )
    stack = " ".join(n.suggested_stack + n.python_scope).lower()
    assert "yolo" not in stack
    assert n.task_type in {"unknown", "general_ml"} or n.paper_kind in {"non_ml", "unknown"}


def test_detection_paper_not_metric_template():
    n = extract_paper_notes(VEHICLE_FLOW, prefer_llm=False)
    proposal = build_proposal(n)
    viva = build_viva_notes(n)

    assert n.task_type == "object_detection_tracking"
    assert n.contribution != "training_technique"

    # Problem: abstract gap only — no equation bleed
    assert "poor detection of small objects" in n.problem.lower() or "struggle" in n.problem.lower()
    assert "above equation" not in n.problem.lower()
    assert "distance measure" not in n.problem.lower()
    assert "2 (1)" not in n.problem

    # Method: modules named, not formula dump
    assert "MSN-YOLO" in n.method or "ByteTrack" in n.method
    assert "LNWD" not in n.method
    assert "2.3" not in n.method

    # Metrics grounded in abstract values — not weak name spam
    assert "62.8" in n.metrics
    assert "72.16" in n.metrics
    assert "1.885" not in n.metrics
    assert "IoU = 0.50" not in n.metrics  # threshold, not a score
    # Bare accuracy/AP/IDS should not appear as name-only spam
    assert not re.search(r"(?i)\baccuracy\b", n.metrics)
    assert not re.search(r"(?i)(?<![A-Za-z])AP(?![A-Za-z0-9])", n.metrics)
    assert not re.search(r"(?i)\bIDS\b", n.metrics)

    # Dataset: MOT07 OCR → MOT17
    assert "MOT17" in n.dataset
    assert "MOT07" not in n.dataset

    # No duplicate bare YOLOv8 when YOLOv8n exists
    yolo_names = [c for c in n.components if re.search(r"(?i)^yolo", c)]
    assert "YOLOv8n" in yolo_names
    assert "YOLOv8" not in yolo_names
    assert "bottleneck" not in (n.unverified_components or [])

    # No NLP compute / stack leakage
    blob = " ".join(r.compute_note + " " + r.reason for r in n.status_table).lower()
    assert "sentence pairs" not in blob
    assert "token embedding" not in blob
    stack = " ".join(n.suggested_stack).lower()
    assert "transformers" not in stack
    assert "ultralytics" in stack or "yolo" in stack or "opencv" in stack

    yolo = next(r for r in n.status_table if re.search(r"(?i)yolov8", r.name))
    assert "encoder" not in yolo.reason.lower()
    assert "sentence" not in yolo.compute_note.lower()

    # Plan / timeline are detection-shaped; week 3 is eval/demo not optional Re-ID
    plan = " ".join(n.python_scope).lower()
    assert "detector" in plan or "yolo" in plan
    assert "correlation numbers" not in proposal.lower()
    assert "encoder load script" not in proposal.lower()
    week3 = proposal.lower().split("| 3 |")[-1].split("| 4 |")[0]
    assert "report" in week3 or "demo" in week3 or "streamlit" in week3
    assert "optionally" not in week3

    # Table OCR junk suppressed; real module ablation may remain
    findings = " ".join(f.summary for f in n.table_findings).lower()
    assert "mot01" not in findings
    assert "vehicle improves" not in findings
    assert "after map" not in findings
    assert "introduction map" not in findings
    assert "1.78k" not in findings  # MOT frame counts must not stick to YOLO

    # Viva: ByteTrack answer is about tracking, not MSN-YOLO dump
    assert "table check (inconsistency)" not in viva.lower()
    assert viva.count("## Q") <= 12
    assert re.search(r"(?i)What is ByteTrack", viva)
    byte_ans = re.search(r"(?i)## Q\d+\. What is ByteTrack[^\n]*\n(.+)", viva)
    assert byte_ans and "associat" in byte_ans.group(1).lower()
    assert "msn-yolo" not in byte_ans.group(1).lower()
