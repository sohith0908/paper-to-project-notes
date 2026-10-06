"""LLM as a checker: verify heuristic notes against paper text (does not invent from scratch)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    ok: bool
    field: str
    issue: str = ""
    suggestion: str = ""


@dataclass
class CheckerReport:
    used_llm: bool
    findings: list[CheckResult] = field(default_factory=list)
    adjusted: dict[str, Any] = field(default_factory=dict)

    @property
    def n_issues(self) -> int:
        return sum(1 for f in self.findings if not f.ok)


def heuristic_self_check(notes_dict: dict[str, Any], text: str) -> CheckerReport:
    """Rule-based checker (always available)."""
    findings: list[CheckResult] = []
    text_l = text.lower()
    comps = [c.lower() for c in notes_dict.get("components") or []]
    for bad in ("imagenet", "cifar-10", "mnist", "coco"):
        if bad in comps:
            findings.append(
                CheckResult(False, "components", f"Dataset '{bad}' listed as component", "Remove datasets from components")
            )
    task = notes_dict.get("task_type")
    if task == "object_detection_tracking" and not re_search_any(
        text_l, ["yolo", "detect", "mota", "bounding box", "track"]
    ):
        findings.append(
            CheckResult(False, "task_type", "Detection task without detection cues", "Re-score tasks / prefer unknown")
        )
    if "yolo" in " ".join(notes_dict.get("suggested_stack") or []).lower() and task not in {
        "object_detection_tracking",
    }:
        findings.append(
            CheckResult(False, "suggested_stack", "YOLO stack on non-detection task", "Use task template stack only")
        )
    metrics = (notes_dict.get("metrics") or "").lower()
    if "look for evaluation" in metrics or "developer" in metrics:
        findings.append(CheckResult(False, "metrics", "Instruction text leaked into metrics", "Use typical-metrics fallback"))
    ds = notes_dict.get("dataset") or ""
    if ds.lower().startswith("abstract") or "year:" in ds.lower():
        findings.append(CheckResult(False, "dataset", "Metadata/abstract leaked into dataset", "Use lexicon extractor"))
    return CheckerReport(used_llm=False, findings=findings)


def re_search_any(text: str, needles: list[str]) -> bool:
    return any(n in text for n in needles)


def llm_check_notes(notes_dict: dict[str, Any], paper_text: str) -> CheckerReport:
    """Ask LLM to verify fields; return issues + optional safe adjustments (no free generation)."""
    base = heuristic_self_check(notes_dict, paper_text)
    if not os.getenv("OPENAI_API_KEY", "").strip():
        return base
    try:
        from openai import OpenAI

        client = OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        excerpt = paper_text[:10000]
        payload = {
            "title": notes_dict.get("title"),
            "paper_type": notes_dict.get("paper_type"),
            "task_type": notes_dict.get("task_type"),
            "task_candidates": notes_dict.get("task_candidates"),
            "problem": notes_dict.get("problem"),
            "method": notes_dict.get("method"),
            "dataset": notes_dict.get("dataset"),
            "metrics": notes_dict.get("metrics"),
            "components": notes_dict.get("components"),
        }
        prompt = f"""
You are a verifier, NOT a generator. Given PAPER TEXT and EXTRACTED JSON, list mistakes only.
Return JSON: {{"issues":[{{"field":"...","issue":"...","suggestion":"..."}}],
"safe_fixes":{{"task_type":null,"components_remove":[],"metrics_note":null}}}}
Rules:
- Do not invent new metrics/datasets/components absent from the paper.
- Flag YOLO stacks on non-detection papers.
- Flag datasets listed as components.
- Flag survey methods that copy the problem paragraph.
- safe_fixes may only remove bad components or set task_type to a candidate already in task_candidates.

EXTRACTED:
{json.dumps(payload)[:4000]}

PAPER:
{excerpt}
"""
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Verify extractions; never invent facts."},
                {"role": "user", "content": prompt},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        for item in data.get("issues") or []:
            base.findings.append(
                CheckResult(
                    False,
                    str(item.get("field") or "unknown"),
                    str(item.get("issue") or ""),
                    str(item.get("suggestion") or ""),
                )
            )
        base.adjusted = data.get("safe_fixes") or {}
        base.used_llm = True
    except Exception as exc:  # noqa: BLE001
        base.findings.append(CheckResult(False, "llm_checker", f"LLM check failed: {exc}", "Use heuristic-only"))
    return base


def apply_safe_fixes(notes_dict: dict[str, Any], report: CheckerReport) -> dict[str, Any]:
    out = dict(notes_dict)
    fixes = report.adjusted or {}
    remove = {c.lower() for c in fixes.get("components_remove") or []}
    if remove:
        out["components"] = [c for c in (out.get("components") or []) if c.lower() not in remove]
    cand_types = {c.get("task_type") for c in (out.get("task_candidates") or []) if isinstance(c, dict)}
    new_task = fixes.get("task_type")
    if new_task and new_task in cand_types:
        out["task_type"] = new_task
    return out
