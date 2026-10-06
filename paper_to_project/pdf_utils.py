"""PDF text + table extraction helpers."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader

from .cleaner import clean_paper_text
from .tables import TableFinding, TableReport, analyze_tables


def extract_text_from_pdf(file_bytes: bytes, max_pages: int = 20) -> str:
    """Extract plain text from a PDF; same cleaner as paste path."""
    reader = PdfReader(BytesIO(file_bytes))
    pages = reader.pages[:max_pages]
    chunks: list[str] = []
    empty_pages = 0
    for page in pages:
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(text.strip())
        else:
            empty_pages += 1
    raw = "\n\n".join(chunks).strip()
    cleaned = clean_paper_text(raw)
    if not cleaned or (empty_pages >= max(1, len(pages) // 2) and len(cleaned) < 400):
        # Scanned / image PDF signal — keep whatever we have; extractor will warn
        note = (
            "\n\n[Note: PDF text extraction was sparse — file may be scanned. "
            "Paste the abstract for better notes.]\n"
        )
        return (cleaned + note).strip() if cleaned else note.strip()
    return cleaned


def extract_tables_from_pdf(file_bytes: bytes, max_pages: int = 8) -> tuple[str, TableReport]:
    """Try pdfplumber for real PDF tables; fall back to text-based analyzer."""
    table_text_parts: list[str] = []
    try:
        import pdfplumber

        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:max_pages]:
                for tbl in page.extract_tables() or []:
                    for row in tbl:
                        cells = [str(c).strip() if c is not None else "" for c in row]
                        if any(cells):
                            table_text_parts.append(" | ".join(cells))
                t = page.extract_text() or ""
                if t.strip():
                    table_text_parts.append(t)
    except Exception:
        table_text_parts.append(extract_text_from_pdf(file_bytes, max_pages=max_pages))

    blob = "\n".join(table_text_parts)
    report = analyze_tables(blob)
    if not report.findings and table_text_parts:
        report.findings.append(
            TableFinding(
                kind="note",
                summary=(
                    f"Parsed table-ish rows via pdfplumber/text ({len(table_text_parts)} blocks); "
                    "no inconsistency/regression flagged."
                ),
            )
        )
    return blob, report
