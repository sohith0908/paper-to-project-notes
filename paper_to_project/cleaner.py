"""Clean noisy PDF / publisher text before field extraction."""

from __future__ import annotations

import re
from collections import Counter


METADATA_LINE = re.compile(
    r"(?i)^\s*("
    r"academic editor|received|revised|accepted|published|citation|copyright|"
    r"licensee|this article|https?://|doi:|mdpi|creativecommons|"
    r"world electr\.?\s*veh\.?\s*j\.|check for updates|article|"
    r"volume|issue|page|pp\.|issn|eissn|authors?|affiliation|"
    r"correspondence|keywords?:|open access|editor|publisher"
    r").*"
)

HEADER_NOISE = re.compile(
    r"(?i)("
    r"world electr\.?\s*veh\.?\s*j\.?\s*\d{4}|"
    r"[A-Za-z]+\.? [A-Za-z]+\.? [A-Za-z]+\.? J\.?\s*\d{4}|"  # generic journal run-heads
    r"https?://doi\.org/\S+|"
    r"creativecommons\.org/\S+|"
    r"©\s*\d{4}\s*by the authors|"
    r"^\s*\d+\s*/\s*\d+\s*$"  # page x/y
    r")"
)

SKIP_TITLE = re.compile(
    r"(?i)^(academic editor|received|revised|accepted|published|citation|"
    r"copyright|licensee|doi|http|www\.|keywords?|abstract|article|"
    r"check for updates|open access|mdpi|volume|issue|"
    r"introduction|related work|references|conclusion|authors?)"
)


def dehyphenate(text: str) -> str:
    """Fix PDF line-break hyphenation: 'vehi-\\ncle' -> 'vehicle'."""
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\r\n(\w)", r"\1\2", text)
    return text


def merge_wrapped_lines(text: str) -> str:
    """Join lines that were hard-wrapped mid-sentence / mid-title."""
    lines = text.splitlines()
    if not lines:
        return text
    out: list[str] = [lines[0]]
    for ln in lines[1:]:
        prev = out[-1]
        cur = ln.strip()
        if not cur:
            out.append("")
            continue
        prev_s = prev.rstrip()
        # Merge title/sentence wraps: previous has no end punct, current continues
        if (
            prev_s
            and not re.search(r"[.!?:;)]$", prev_s)
            and not METADATA_LINE.match(prev_s)
            and not METADATA_LINE.match(cur)
            and not re.match(r"(?i)^\s*(\d+(\.\d+)*\.?\s+)?[A-Z][A-Za-z]+$", cur)
            and (
                cur[:1].islower()
                or (
                    len(prev_s) < 90
                    and not prev_s.endswith("-")
                    and cur[:1].isupper()
                    and not re.match(r"(?i)^(abstract|keywords?|introduction|figure|table)\b", cur)
                )
            )
        ):
            out[-1] = prev_s + " " + cur
        else:
            out.append(ln)
    return "\n".join(out)


def strip_headers_footers(text: str) -> str:
    """Remove repeated short lines that look like running headers/footers."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    counts = Counter(ln.strip() for ln in lines if 8 <= len(ln.strip()) <= 90)
    repeated = {
        ln
        for ln, c in counts.items()
        if c >= 3 and not ln.lower().startswith("abstract")
    }
    cleaned: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            cleaned.append("")
            continue
        if s in repeated:
            continue
        if HEADER_NOISE.search(s):
            continue
        if METADATA_LINE.match(s):
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_equation_blocks(text: str) -> str:
    """Drop obvious LaTeX/PDF equation blocks so OCR junk does not flood prose."""
    text = re.sub(r"(?is)\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " ", text)
    text = re.sub(r"(?is)\\begin\{align\*?\}.*?\\end\{align\*?\}", " ", text)
    text = re.sub(r"\$\$[^$]{2,800}\$\$", " ", text)
    # Long glyph-heavy lines (PDF math dumps)
    kept = []
    for ln in text.splitlines():
        letters = len(re.findall(r"[A-Za-z]", ln))
        symbols = len(re.findall(r"[∑∫∇√\\=]", ln))
        if letters < 12 and symbols >= 4:
            continue
        if letters < 8 and len(re.findall(r"\d", ln)) >= 8:
            continue
        kept.append(ln)
    return "\n".join(kept)


def clean_paper_text(text: str) -> str:
    text = dehyphenate(text)
    text = strip_headers_footers(text)
    text = strip_equation_blocks(text)
    text = merge_wrapped_lines(text)
    return normalize_whitespace(text)


def extract_section(text: str, start_patterns: list[str], end_patterns: list[str]) -> str:
    """Return text between a start heading and the next end heading."""
    start_idx = -1
    start_pat_used = None
    for pat in start_patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m and (start_idx < 0 or m.start() < start_idx):
            start_idx = m.start()
            start_pat_used = m

    if start_idx < 0 or start_pat_used is None:
        return ""

    after = text[start_pat_used.end() :]
    end_idx = len(after)
    for pat in end_patterns:
        m = re.search(pat, after, flags=re.IGNORECASE | re.MULTILINE)
        if m and m.start() < end_idx:
            end_idx = m.start()
    return after[:end_idx].strip()


def extract_abstract(text: str) -> str:
    block = extract_section(
        text,
        start_patterns=[r"(?m)^\s*Abstract\b[:\s]*", r"(?i)\bAbstract\b[:\s]*"],
        end_patterns=[
            r"(?m)^\s*Keywords?\b",
            r"(?m)^\s*1[\.\s]+Introduction\b",
            r"(?m)^\s*Introduction\b",
            r"(?m)^\s*1\s+Introduction\b",
            # PDF body often continues without a clean Introduction header
            r"(?m)^\s*\d+\.\d+\.?\s+[A-Z]",
            r"(?m)^\s*2[\.\s]+(?:Related|Method|Proposed|Normalized|Detection)\b",
        ],
    )
    if block:
        block = re.split(r"(?i)\bKeywords?\b", block, maxsplit=1)[0].strip()
        # Never keep the word Abstract itself
        block = re.sub(r"(?i)^\s*Abstract\s*[:\-]?\s*", "", block).strip()
        # If title leaked into abstract first line, drop it
        title = guess_title(text)
        if title and block.lower().startswith(title.lower()):
            block = block[len(title) :].lstrip(" :-–—\n")
            block = re.sub(r"(?i)^\s*Abstract\s*[:\-]?\s*", "", block).strip()
        return block
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 180]
    return paras[0] if paras else ""


def guess_title(text: str) -> str:
    """Merge front-matter title lines; skip publisher sidebar metadata."""
    front = text
    m_abs = re.search(r"(?im)^\s*Abstract\b", text)
    if m_abs:
        front = text[: m_abs.start()]

    # Collect consecutive candidate fragments, then merge
    fragments: list[str] = []
    for raw in front.splitlines()[:50]:
        line = raw.strip()
        if not line or SKIP_TITLE.match(line) or METADATA_LINE.match(line):
            if fragments:
                break
            continue
        if HEADER_NOISE.search(line):
            continue
        if re.search(r"\d{4};\d+", line) or "doi.org" in line.lower():
            continue
        if re.match(r"(?i)^(we |this paper|in this|existing |traditional )", line):
            if fragments:
                break
            continue
        if len(line) < 8 or len(line) > 220:
            continue
        fragments.append(line)
        # Stop after we have a solid multi-word title
        merged_len = len(" ".join(fragments))
        if merged_len >= 40 and (line.endswith(".") or merged_len >= 120):
            if line.endswith("."):
                fragments[-1] = fragments[-1].rstrip(".")
            break
        if len(fragments) >= 4:
            break

    if not fragments:
        for raw in text.splitlines()[:40]:
            line = raw.strip()
            if 20 <= len(line) <= 200 and not SKIP_TITLE.match(line) and not line.endswith("."):
                return line
        return "Untitled paper"

    title = re.sub(r"\s+", " ", " ".join(fragments)).strip(" .-")
    # Drop leading author initials / footnote markers: "H. Vehicle Flow..."
    title = re.sub(r"^(?:[A-Z]\.\s*)+", "", title).strip()
    title = re.sub(r"^\d+\s*", "", title).strip()
    # Drop trailing metadata / author lists glued into the title
    title = re.split(r"\s+(?:Year|DOI|Keywords?|Abstract)\s*:", title, maxsplit=1, flags=re.I)[0].strip()
    title = re.split(r"\s+[A-Z][a-z]+,\s+[A-Z][a-z]+(?:,\s+[A-Z][a-z]+)+\s*$", title)[0].strip()
    title = re.split(r"\s+[A-Z][a-z]+,\s+[A-Z]\.", title)[0].strip()
    return title or "Untitled paper"
