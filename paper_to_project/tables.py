"""Parse paper tables / numeric rows; flag inconsistencies, regressions, ablations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TableFinding:
    kind: str  # inconsistency | regression | ablation | note
    summary: str
    evidence: str = ""


@dataclass
class TableReport:
    findings: list[TableFinding] = field(default_factory=list)
    rows_seen: int = 0

    @property
    def has_issues(self) -> bool:
        return any(f.kind in {"inconsistency", "regression"} for f in self.findings)


_NUM = r"(\d+(?:\.\d+)?)\s*(%|M|K|mAP|MOTA|IDF1)?"
_ENTITY = re.compile(
    r"\b("
    r"YOLOv?\d+\w*|ByteTrack|DeepSORT|FairMOT|StrongSORT|"
    r"[A-Z][A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)+|"
    r"MOT\d+|BDD\d+|COCO|KITTI|UA-DETRAC|VisDrone"
    r")\b"
)
_METRIC_NAME = re.compile(
    r"(?i)\b(mAP50(?:-95)?|mAP|MOTA|MOTP|IDF1|HOTA|FPS|precision|recall|F1|Params?|GFLOPs?)\b"
)

# OCR / prose entities that are not real model/module names
_JUNK_ENT = re.compile(
    r"(?i)^(and|the|after|before|introduction|foundation|vehicle|table|figure|"
    r"section|with|from|this|that|when|for|our|using|based|proposed|method|"
    r"model|result|results|experiment|dataset|accuracy|precision|recall)$"
)
_MOT_SEQ = re.compile(r"(?i)^MOT\d+$")


def _to_float(num: str, unit: str | None) -> float | None:
    try:
        v = float(num)
    except ValueError:
        return None
    u = (unit or "").lower()
    if u == "m":
        return v * 1_000_000
    if u == "k":
        return v * 1_000
    return v


def _plausible_module(name: str) -> bool:
    if not name or _JUNK_ENT.match(name) or _MOT_SEQ.match(name):
        return False
    if re.search(r"(?i)yolo|track|c2f|seam|mlca|nwd|resnet|bert|msn|reid|byte", name):
        return True
    if "_" in name or (name.count("-") == 1 and re.search(r"\d", name)):
        return True
    if re.fullmatch(r"[A-Z]{2,6}", name):
        return True
    return len(name) >= 5 and name[0].isupper() and not name.isupper()


def parse_loose_tables(text: str) -> list[list[str]]:
    """Extract simple whitespace / pipe tables from plain PDF text."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or len(s) < 8:
            continue
        if "|" in s and s.count("|") >= 2:
            cells = [c.strip() for c in s.strip("|").split("|") if c.strip()]
            if len(cells) >= 2:
                rows.append(cells)
                continue
        if len(re.findall(r"\d+(?:\.\d+)?", s)) >= 2 and _ENTITY.search(s):
            cells = re.split(r"\s{2,}|\t+", s)
            if len(cells) >= 2:
                rows.append([c.strip() for c in cells if c.strip()])
    return rows


def _collect_entity_values(text: str) -> dict[str, list[tuple[float, str]]]:
    """Map entity -> list of (value, evidence snippet). Prefer M/K param units."""
    out: dict[str, list[tuple[float, str]]] = {}
    for m in _ENTITY.finditer(text):
        ent = m.group(1)
        if _MOT_SEQ.match(ent) or _JUNK_ENT.match(ent):
            continue
        # Tight window; cut before the next named entity so MOT frame counts
        # do not attach to a nearby YOLO mention
        window = text[m.start() : m.start() + 90]
        rest = window[len(ent) :]
        cut = re.search(
            r"\b(?:MOT\d+|YOLOv?\d+\w*|ByteTrack|DeepSORT|UA-DETRAC|COCO|KITTI)\b",
            rest,
        )
        if cut:
            window = window[: len(ent) + cut.start()]
        # Prefer params/GFLOPs-style contexts for K/M
        for nm in re.finditer(_NUM, window):
            unit = nm.group(2)
            if not unit or unit.lower() not in {"m", "k", "%"}:
                continue
            ctx = window[max(0, nm.start() - 20) : nm.end() + 20]
            if unit.lower() in {"m", "k"} and not re.search(
                r"(?i)param|gflop|size|model|params|mb|million|thousand", ctx
            ):
                # Bare "1.78K frames" is not a model size conflict
                if re.search(r"(?i)frame|image|seq|video|sample", ctx):
                    continue
                if not re.search(r"(?i)param|gflop|size|mb\b", window):
                    continue
            val = _to_float(nm.group(1), unit)
            if val is None:
                continue
            if unit == "%" and val < 5:
                continue
            out.setdefault(ent, []).append((val, window[:90].replace("\n", " ")))
    return out


def find_inconsistencies(text: str, rel_tol: float = 0.05) -> list[TableFinding]:
    """Same entity reported with conflicting M/K/% magnitudes."""
    findings: list[TableFinding] = []
    bag = _collect_entity_values(text)
    for ent, vals in bag.items():
        if _MOT_SEQ.match(ent):
            continue
        # Param-like (K/M) only — not MOT sequence frame counts
        big = [(v, e) for v, e in vals if v >= 1000]
        uniq = []
        for v, e in big:
            if all(abs(v - u) / max(u, 1) > rel_tol for u, _ in uniq):
                uniq.append((v, e))
        if len(uniq) >= 2:
            a, ea = uniq[0]
            b, eb = uniq[1]
            findings.append(
                TableFinding(
                    kind="inconsistency",
                    summary=(
                        f"Inconsistency: {ent} appears with conflicting sizes/values "
                        f"({_fmt(a)} vs {_fmt(b)}). Check table vs text."
                    ),
                    evidence=f"{ea} || {eb}",
                )
            )
    return findings


def find_regressions(text: str) -> list[TableFinding]:
    """Flag large metric drops for plausible model/module names only."""
    findings: list[TableFinding] = []
    row_pat = re.compile(
        r"(?i)(?P<ent>MOT\d+|[A-Z][A-Za-z0-9_-]{2,}).{0,40}?"
        r"(?P<metric>MOTA|mAP50|mAP|IDF1|HOTA|FPS|error|top-?1|top-?5).{0,20}?"
        r"(?P<a>\d+(?:\.\d+)?)\D{1,40}(?P<b>\d+(?:\.\d+)?)"
    )
    seen = set()
    for m in row_pat.finditer(text):
        ent = m.group("ent")
        if not _plausible_module(ent) and not _MOT_SEQ.match(ent):
            continue
        # MOT## sequence rows in PDFs are almost always OCR noise for size conflicts
        if _MOT_SEQ.match(ent):
            continue
        a = float(m.group("a"))
        b = float(m.group("b"))
        if a <= 0 or b < 0:
            continue
        # Require a real drop (not IoU 0.50 vs 1, or mAP50 threshold junk)
        if a > 100 or b > 100:
            continue
        drop = a - b
        if drop < 5:
            continue
        # 50 → 1 style is usually IoU/threshold OCR, not a reported regression
        if a in {50, 50.0} and b <= 2:
            continue
        if a in {10, 10.0} and b <= 6:
            continue
        key = (ent.lower(), m.group("metric").lower(), a, b)
        if key in seen:
            continue
        seen.add(key)
        findings.append(
            TableFinding(
                kind="regression",
                summary=(
                    f"Regression/degradation check: {ent} {m.group('metric')} "
                    f"moves {a:g} -> {b:g}. Be ready to explain why (domain shift, "
                    f"different split, degradation problem, or weaker setting)."
                ),
                evidence=m.group(0)[:140].replace("\n", " "),
            )
        )

    # ResNet-style: plain-34 higher error than plain-18
    deg2 = re.findall(
        r"(?i)plain[- ]?(18|34|50|101|152).{0,40}?(\d+(?:\.\d+)?)\s*%?",
        text,
    )
    if len(deg2) >= 2:
        (d1, e1), (d2, e2) = deg2[0], deg2[1]
        try:
            v1, v2 = float(e1), float(e2)
            if int(d2) > int(d1) and v2 > v1:
                findings.append(
                    TableFinding(
                        kind="regression",
                        summary=(
                            f"Degradation result: plain-{d2} error ({v2:g}) is higher than plain-{d1} "
                            f"({v1:g}) — deeper plain net is worse. Viva: this motivates residual learning."
                        ),
                        evidence=f"plain-{d1} {v1} vs plain-{d2} {v2}",
                    )
                )
        except ValueError:
            pass

    uniq = []
    seen_s = set()
    for f in findings:
        if f.summary in seen_s:
            continue
        seen_s.add(f.summary)
        uniq.append(f)
    return uniq[:5]


def find_ablation_deltas(text: str) -> list[TableFinding]:
    """Catch real ablation deltas tied to plausible module names."""
    findings: list[TableFinding] = []
    pat = re.compile(
        r"(?i)(?P<mod>[A-Z][A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*)"
        r".{0,40}?(?P<delta>\+\s*\d+(?:\.\d+)?)\s*"
        r"(?P<metric>mAP50|mAP|MOTA|IDF1|F1)"
    )
    seen = set()
    for m in pat.finditer(text):
        mod = m.group("mod")
        if not _plausible_module(mod):
            continue
        key = (mod.lower(), m.group("delta"), m.group("metric").lower())
        if key in seen:
            continue
        seen.add(key)
        findings.append(
            TableFinding(
                kind="ablation",
                summary=(
                    f"Ablation: {mod} alone contributes "
                    f"{m.group('delta').replace(' ', '')} {m.group('metric')}."
                ),
                evidence=m.group(0)[:120].replace("\n", " "),
            )
        )
    # "C2f_MLCA ... improves ... by 0.7"
    pat2 = re.compile(
        r"(?i)(?P<mod>[A-Z][A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)*)"
        r".{0,50}?(?:improv\w*|gain\w*|increase\w*).{0,20}?"
        r"(?P<delta>\d+(?:\.\d+)?)\s*(?P<metric>mAP50|mAP|MOTA|IDF1|F1)"
    )
    for m in pat2.finditer(text):
        mod = m.group("mod")
        if not _plausible_module(mod):
            continue
        delta = m.group("delta")
        metric = m.group("metric")
        key = (mod.lower(), delta, metric.lower())
        if key in seen:
            continue
        if float(delta) > 15:  # unlikely single-module ablation
            continue
        seen.add(key)
        findings.append(
            TableFinding(
                kind="ablation",
                summary=f"Ablation: {mod} improves {metric} by +{delta}.",
                evidence=m.group(0)[:120].replace("\n", " "),
            )
        )
    return findings


def _fmt(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.2f}K"
    return f"{v:g}"


def analyze_tables(text: str) -> TableReport:
    rows = parse_loose_tables(text)
    findings: list[TableFinding] = []
    findings.extend(find_inconsistencies(text))
    findings.extend(find_regressions(text))
    findings.extend(find_ablation_deltas(text))
    uniq: list[TableFinding] = []
    seen = set()
    for f in findings:
        if f.summary in seen:
            continue
        seen.add(f.summary)
        uniq.append(f)
    return TableReport(findings=uniq[:8], rows_seen=len(rows))
