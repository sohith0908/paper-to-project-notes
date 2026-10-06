"""Status table from RoleRecords — plain-language reasons only."""

from __future__ import annotations

from dataclasses import dataclass

from .components import ComponentItem
from .roles import RoleRecord, build_role_records


@dataclass
class ComponentStatus:
    name: str
    status: str
    reason: str
    compute_note: str = ""
    role: str = "technique"
    provenance: str = "from_paper"
    evidence_span: str = ""


def build_status_table(
    components: list[str] | list[ComponentItem],
    task_type: str = "unknown",
    paper_text: str = "",
    contribution: str = "new_model",
    roles: dict[str, str] | None = None,
    role_records: list[RoleRecord] | None = None,
    title: str = "",
) -> list[ComponentStatus]:
    if role_records is not None:
        return [
            ComponentStatus(
                name=r.name,
                status=r.status,
                reason=r.reason,
                compute_note=r.compute_note,
                role=r.role,
                provenance=r.provenance,
                evidence_span=r.evidence_span,
            )
            for r in role_records
            if r.provenance != "unverified"
        ]

    items: list[ComponentItem] = []
    for c in components:
        if isinstance(c, ComponentItem):
            items.append(c)
        else:
            role = (roles or {}).get(c.lower(), "technique")
            items.append(ComponentItem(c, role))
    records = build_role_records(
        items, paper_text, contribution, title=title, task_type=task_type
    )
    return [
        ComponentStatus(
            name=r.name,
            status=r.status,
            reason=r.reason,
            compute_note=r.compute_note,
            role=r.role,
            provenance=r.provenance,
            evidence_span=r.evidence_span,
        )
        for r in records
        if r.provenance != "unverified"
    ]


def status_viva_questions(rows: list[ComponentStatus]) -> list[tuple[str, str]]:
    """No status-table restatements for viva — return empty; proposal owns the table."""
    return []


def status_markdown(rows: list[ComponentStatus]) -> str:
    if not rows:
        return "_No components with a proving sentence._"
    lines = [
        "| Component | Role | Status | Why | Compute |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        # Never print provenance tags like verified/inferred in the table to students
        lines.append(
            f"| {r.name} | {r.role} | **{r.status}** | {r.reason} | {r.compute_note} |"
        )
    return "\n".join(lines)
