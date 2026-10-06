"""Fetch paper metadata/abstract by title, DOI, or arXiv id (no API key needed)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

USER_AGENT = "PaperToProjectNotes/0.2 (student mini-project helper)"


@dataclass
class FetchedPaper:
    title: str
    abstract: str = ""
    doi: str = ""
    url: str = ""
    source: str = ""
    authors: str = ""
    year: str = ""

    @property
    def text_blob(self) -> str:
        parts = [self.title]
        if self.authors:
            parts.append(f"Authors: {self.authors}")
        if self.year:
            parts.append(f"Year: {self.year}")
        if self.doi:
            parts.append(f"DOI: {self.doi}")
        if self.abstract:
            parts.append("Abstract\n" + self.abstract)
        return "\n\n".join(parts)


def _looks_like_doi(query: str) -> bool:
    return bool(re.search(r"10\.\d{4,9}/\S+", query))


def _looks_like_arxiv(query: str) -> bool:
    return bool(re.search(r"(?i)\barxiv[:\s]?\d{4}\.\d{4,5}", query)) or bool(
        re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", query.strip())
    )


def _extract_doi(query: str) -> str:
    m = re.search(r"(10\.\d{4,9}/\S+)", query)
    return m.group(1).rstrip(".,;)") if m else ""


def fetch_by_doi(doi: str) -> FetchedPaper | None:
    doi = doi.strip()
    url = f"https://api.crossref.org/works/{quote(doi)}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        item = r.json().get("message", {})
        return _from_crossref_item(item, source="crossref-doi")
    except Exception:
        return None


def fetch_crossref_by_title(title: str, rows: int = 5) -> list[FetchedPaper]:
    url = "https://api.crossref.org/works"
    try:
        r = requests.get(
            url,
            params={"query.title": title, "rows": rows},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        return [
            p
            for p in (_from_crossref_item(it, source="crossref-title") for it in items)
            if p is not None
        ]
    except Exception:
        return []


def fetch_openalex_by_title(title: str, rows: int = 5) -> list[FetchedPaper]:
    url = "https://api.openalex.org/works"
    try:
        r = requests.get(
            url,
            params={"search": title, "per_page": rows},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        out: list[FetchedPaper] = []
        for it in results:
            abstract = _openalex_abstract(it.get("abstract_inverted_index"))
            authorships = it.get("authorships") or []
            authors = ", ".join(
                (a.get("author") or {}).get("display_name", "")
                for a in authorships[:8]
                if (a.get("author") or {}).get("display_name")
            )
            year = str(it.get("publication_year") or "")
            doi = (it.get("doi") or "").replace("https://doi.org/", "")
            out.append(
                FetchedPaper(
                    title=it.get("display_name") or title,
                    abstract=abstract,
                    doi=doi,
                    url=it.get("id") or "",
                    source="openalex",
                    authors=authors,
                    year=year,
                )
            )
        return out
    except Exception:
        return []


def fetch_semanticscholar_by_title(title: str, rows: int = 5) -> list[FetchedPaper]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    try:
        r = requests.get(
            url,
            params={
                "query": title,
                "limit": rows,
                "fields": "title,abstract,externalIds,url,year,authors",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json().get("data") or []
        out: list[FetchedPaper] = []
        for it in data:
            authors = ", ".join(
                a.get("name", "") for a in (it.get("authors") or [])[:8] if a.get("name")
            )
            ext = it.get("externalIds") or {}
            out.append(
                FetchedPaper(
                    title=it.get("title") or title,
                    abstract=it.get("abstract") or "",
                    doi=ext.get("DOI") or "",
                    url=it.get("url") or "",
                    source="semanticscholar",
                    authors=authors,
                    year=str(it.get("year") or ""),
                )
            )
        return out
    except Exception:
        return []


def _from_crossref_item(item: dict[str, Any], source: str) -> FetchedPaper | None:
    titles = item.get("title") or []
    if not titles:
        return None
    abstract = item.get("abstract") or ""
    abstract = re.sub(r"<[^>]+>", " ", abstract)
    abstract = re.sub(r"\s+", " ", abstract).strip()
    authors_list = []
    for a in item.get("author") or []:
        name = " ".join(x for x in [a.get("given"), a.get("family")] if x)
        if name:
            authors_list.append(name)
    year = ""
    issued = (item.get("issued") or {}).get("date-parts") or []
    if issued and issued[0]:
        year = str(issued[0][0])
    return FetchedPaper(
        title=titles[0],
        abstract=abstract,
        doi=item.get("DOI") or "",
        url=item.get("URL") or "",
        source=source,
        authors=", ".join(authors_list[:8]),
        year=year,
    )


def _openalex_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    size = max((max(pos) for pos in inverted.values()), default=-1) + 1
    words = [""] * size
    for word, positions in inverted.items():
        for p in positions:
            if 0 <= p < size:
                words[p] = word
    return " ".join(w for w in words if w).strip()


def _title_similarity(a: str, b: str) -> float:
    ta = {w for w in re.findall(r"[a-z0-9]+", a.lower()) if len(w) > 2}
    tb = {w for w in re.findall(r"[a-z0-9]+", b.lower()) if len(w) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _arxiv_id(query: str) -> str:
    m = re.search(r"(?i)arxiv[:\s]?(\d{4}\.\d{4,5})(v\d+)?", query)
    if m:
        return m.group(1)
    m = re.fullmatch(r"(\d{4}\.\d{4,5})(v\d+)?", query.strip())
    return m.group(1) if m else ""


def fetch_arxiv_by_id(arxiv_id: str) -> FetchedPaper | None:
    """Fetch title/abstract from the public arXiv Atom API (no key)."""
    aid = arxiv_id.strip()
    if not aid:
        return None
    url = f"http://export.arxiv.org/api/query?id_list={quote(aid)}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        xml = r.text
        # First <title> is feed title; entry title is usually second
        titles = re.findall(r"<title>(.*?)</title>", xml, flags=re.S)
        title = (titles[1] if len(titles) > 1 else (titles[0] if titles else "")).strip()
        title = re.sub(r"\s+", " ", title)
        abs_m = re.search(r"<summary>(.*?)</summary>", xml, flags=re.S)
        abstract = re.sub(r"\s+", " ", (abs_m.group(1) if abs_m else "").strip())
        authors = ", ".join(
            re.findall(r"<name>(.*?)</name>", xml)[:8]
        )
        year_m = re.search(r"<published>(\d{4})", xml)
        if not title:
            return None
        return FetchedPaper(
            title=title,
            abstract=abstract,
            doi="",
            url=f"https://arxiv.org/abs/{aid}",
            source="arxiv",
            authors=authors,
            year=year_m.group(1) if year_m else "",
        )
    except Exception:
        return None


def resolve_paper_query(query: str) -> FetchedPaper:
    """Resolve a title / DOI / arXiv-ish query into the best metadata+abstract hit."""
    query = query.strip()
    if not query:
        raise ValueError("Empty title/query.")

    if _looks_like_doi(query):
        hit = fetch_by_doi(_extract_doi(query))
        if hit and (hit.abstract or hit.title):
            return hit

    if _looks_like_arxiv(query):
        hit = fetch_arxiv_by_id(_arxiv_id(query))
        if hit and (hit.abstract or hit.title):
            return hit

    candidates: list[FetchedPaper] = []
    for fetcher in (
        fetch_openalex_by_title,
        fetch_semanticscholar_by_title,
        fetch_crossref_by_title,
    ):
        candidates.extend(fetcher(query))

    if not candidates:
        # Last chance: treat bare arXiv id patterns inside longer queries
        aid = _arxiv_id(query)
        if aid:
            hit = fetch_arxiv_by_id(aid)
            if hit:
                return hit
        raise ValueError(
            "Could not find that paper online. Try a fuller title, DOI, arXiv id, "
            "or upload/paste the PDF text."
        )

    # Prefer candidates with abstracts and similar titles
    ranked = sorted(
        candidates,
        key=lambda p: (
            1 if p.abstract else 0,
            _title_similarity(query, p.title),
            len(p.abstract),
        ),
        reverse=True,
    )
    best = ranked[0]
    if not best.abstract and best.doi:
        by_doi = fetch_by_doi(best.doi)
        if by_doi and by_doi.abstract:
            best.abstract = by_doi.abstract
            best.source = f"{best.source}+crossref-doi"
    # Try arXiv id from URL/title if still no abstract
    if not best.abstract:
        aid = _arxiv_id(query) or _arxiv_id(best.url or "")
        if aid:
            ax = fetch_arxiv_by_id(aid)
            if ax and ax.abstract:
                best.abstract = ax.abstract
                best.source = f"{best.source}+arxiv"
    if not best.abstract:
        # Explicit empty abstract — extractor treats as metadata_only
        best.abstract = (
            "Abstract not available from open metadata. "
            "Upload the PDF or paste the abstract for better notes."
        )
    return best
