"""Streamlit UI for Paper-to-Project Notes."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from paper_to_project.dataset_match import match_dataset_description, match_tabular_dataset
from paper_to_project.extractor import extract_paper_notes
from paper_to_project.fetch import resolve_paper_query
from paper_to_project.pdf_utils import extract_text_from_pdf
from paper_to_project.proposal import build_markdown_bundle, build_proposal, build_viva_notes
from paper_to_project.status import status_markdown

load_dotenv()

# Streamlit Cloud secrets → env (optional LLM checker)
try:
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        os.environ.setdefault("OPENAI_API_KEY", str(st.secrets["OPENAI_API_KEY"]))
except Exception:
    pass

SAMPLE_PATH = Path(__file__).resolve().parent / "samples" / "sample_paper.txt"

st.set_page_config(
    page_title="Paper-to-Project Notes",
    page_icon="📄",
    layout="wide",
)

st.title("Paper-to-Project Notes")
st.caption(
    "Upload a PDF, paste text, or search by title/DOI → structured notes, "
    "table checks, component status, dataset match, proposal & viva Q&A."
)

with st.sidebar:
    st.header("Settings")
    student_name = st.text_input("Your name", value="Student")
    prefer_llm = st.toggle(
        "Use LLM if API key is set",
        value=True,
        help="Optional. Without a key, structure-aware heuristics still work.",
    )
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    st.write(f"API key detected: **{'yes' if has_key else 'no'}**")

tab_input, tab_notes, tab_proposal, tab_viva = st.tabs(
    ["1. Input", "2. Extracted notes", "3. Proposal", "4. Viva notes"]
)

with tab_input:
    mode = st.radio(
        "How do you want to provide the paper?",
        ["Upload / paste", "Search by title or DOI"],
        horizontal=True,
    )

    paper_text = st.session_state.get("paper_text", "")

    if mode == "Search by title or DOI":
        query = st.text_input(
            "Paper title or DOI",
            placeholder="e.g. full paper title, or 10.xxxx/xxxxx",
        )
        if st.button("Fetch paper metadata", type="primary", use_container_width=True):
            try:
                with st.spinner("Searching Crossref / OpenAlex / Semantic Scholar…"):
                    hit = resolve_paper_query(query)
                st.session_state["fetch_hit"] = hit
                st.session_state["paper_text"] = hit.text_blob
                paper_text = hit.text_blob
                st.success(
                    f"Matched: **{hit.title}** via `{hit.source}`"
                    + (f" · DOI `{hit.doi}`" if hit.doi else "")
                )
                if "Abstract not available" in (hit.abstract or ""):
                    st.warning(
                        "No abstract online. Paste more text or upload the PDF for better notes."
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        hit = st.session_state.get("fetch_hit")
        if hit:
            with st.expander("Fetched metadata", expanded=True):
                st.write(f"**Title:** {hit.title}")
                if hit.authors:
                    st.write(f"**Authors:** {hit.authors}")
                if hit.year:
                    st.write(f"**Year:** {hit.year}")
                st.write(hit.abstract[:1200] + ("…" if len(hit.abstract) > 1200 else ""))

        paper_text = st.text_area(
            "Editable text used for extraction (you can paste more sections here)",
            value=st.session_state.get("paper_text", paper_text),
            height=260,
        )
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        with col_b:
            if st.button("Load sample paper"):
                if SAMPLE_PATH.exists():
                    st.session_state["paper_text"] = SAMPLE_PATH.read_text(encoding="utf-8")
                else:
                    st.warning("Sample file missing.")

        default_text = st.session_state.get("paper_text", "")
        if uploaded is not None:
            try:
                default_text = extract_text_from_pdf(uploaded.getvalue())
                st.session_state["paper_text"] = default_text
                st.success(f"Extracted text from PDF ({len(default_text):,} chars).")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not read PDF: {exc}")

        paper_text = st.text_area(
            "Or paste abstract + method + experiments + tables",
            value=default_text,
            height=320,
            placeholder="Paste paper text here…",
        )

    st.markdown("#### Optional: your dataset (match vs paper)")
    ds_file = st.file_uploader("Upload CSV/TSV dataset sample", type=["csv", "tsv"])
    ds_note = st.text_input(
        "Or describe your dataset",
        placeholder="e.g. 500 traffic frames with car/bus labels (subset of UA-DETRAC)",
    )

    run = st.button("Generate project notes", type="primary", use_container_width=True)

    if run:
        try:
            with st.spinner("Extracting notes + table checks…"):
                # dataset match needs paper dataset text — run extract first without match,
                # then re-attach match. Simpler: extract, then compute match from notes.dataset
                notes = extract_paper_notes(paper_text, prefer_llm=prefer_llm)
                dm = None
                if ds_file is not None:
                    dm = match_tabular_dataset(
                        ds_file.name,
                        ds_file.getvalue(),
                        notes.dataset,
                        notes.metrics,
                    )
                elif ds_note.strip():
                    dm = match_dataset_description(ds_note.strip(), notes.dataset)
                notes.dataset_match = dm
            st.session_state["notes"] = notes
            st.session_state["paper_text"] = paper_text
            st.success(
                f"Done via **{notes.source_mode}** · task **{notes.task_label}**. "
                f"Table findings: {len(notes.table_findings)}."
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

notes = st.session_state.get("notes")

with tab_notes:
    if not notes:
        st.info("Generate notes from the Input tab.")
    else:
        st.subheader(notes.title or "Extracted notes")
        st.caption(
            f"Mode: {notes.source_mode} · Paper type: {notes.paper_type} · "
            f"Kind: `{getattr(notes, 'paper_kind', 'ml_method')}` · "
            f"Input: `{getattr(notes, 'input_tier', 'full')}` · "
            f"Contribution: `{getattr(notes, 'contribution', 'new_model')}` · "
            f"Task: {notes.task_label} (`{notes.task_type}`)"
        )
        if getattr(notes, "warnings", None):
            for w in notes.warnings:
                st.warning(w)
        if getattr(notes, "missing_items", None):
            st.info("Missing from input: " + "; ".join(notes.missing_items))
        if getattr(notes, "task_candidates", None):
            top = notes.task_candidates[0]
            line = f"**Task routing:** {top.task_type}={top.score}"
            if len(notes.task_candidates) > 1 and notes.task_candidates[1].score > 0:
                r = notes.task_candidates[1]
                line += f" · runner-up {r.task_type}={r.score}"
            else:
                line += " · " + (getattr(notes, "runner_up_note", None) or "no second candidate above threshold")
            st.markdown(line)
        if getattr(notes, "contribution_scores", None):
            cs = notes.contribution_scores[:2]
            st.caption(
                "Contribution scores: "
                + ", ".join(f"{x.get('contribution')}={x.get('score')}" for x in cs)
            )
        if getattr(notes, "checker", None) and notes.checker and notes.checker.n_issues:
            st.warning(
                "Checker issues: "
                + "; ".join(f"{f.field}: {f.issue}" for f in notes.checker.findings if not f.ok)[:500]
            )
        st.markdown("### Problem")
        st.write(notes.problem)
        st.markdown("### Method")
        st.write(notes.method)
        if notes.components:
            st.markdown("### Named components")
            st.write(", ".join(notes.components))
        if getattr(notes, "unverified_components", None):
            st.markdown("### Names needing a clearer sentence")
            st.write(", ".join(notes.unverified_components))
        if notes.status_table:
            st.markdown("### Component status")
            st.markdown(status_markdown(notes.status_table))
        st.markdown("### Dataset (from paper)")
        st.write(notes.dataset)
        if getattr(notes, "dataset_suggested", None):
            st.markdown("### Dataset (suggested, not stated)")
            st.write(notes.dataset_suggested)
        if notes.dataset_match:
            st.markdown("### Dataset vs paper")
            st.write(notes.dataset_match.summary)
        st.markdown("### Metrics (from paper)")
        st.write(notes.metrics)
        if getattr(notes, "metrics_suggested", None):
            st.markdown("### Metrics (suggested, not stated)")
            st.write(notes.metrics_suggested)
        with st.expander("Debug: evidence tags", expanded=False):
            if getattr(notes, "evidence", None):
                for e in notes.evidence[:16]:
                    src = e.source_sentence[:200] if e.source_sentence else "(no span)"
                    st.markdown(
                        f"- **{e.field}** `{e.confidence}` `{e.provenance}`: `{e.value[:140]}` — _{src}_"
                    )
            else:
                st.write("Empty")
            if getattr(notes, "runner_up_note", None):
                st.caption(notes.runner_up_note)
            if getattr(notes, "sections_found", None):
                st.caption("Sections: " + ", ".join(notes.sections_found))
        st.markdown("### Table checks")
        if notes.table_findings:
            for f in notes.table_findings:
                st.markdown(f"- **{f.kind}:** {f.summary}")
        else:
            st.write("None flagged.")
        st.markdown("### Plan")
        for item in notes.python_scope:
            st.markdown(f"- {item}")
        st.markdown("### Out of scope")
        for item in notes.out_of_scope:
            st.markdown(f"- {item}")
        st.markdown("### Stack (from plan steps)")
        st.write(", ".join(notes.suggested_stack))

with tab_proposal:
    if not notes:
        st.info("Generate notes from the Input tab.")
    else:
        proposal_md = build_proposal(notes, student_name=student_name)
        st.markdown(proposal_md)
        st.download_button(
            "Download proposal.md",
            data=proposal_md,
            file_name="project_proposal.md",
            mime="text/markdown",
        )

with tab_viva:
    if not notes:
        st.info("Generate notes from the Input tab.")
    else:
        viva_md = build_viva_notes(notes)
        st.markdown(viva_md)
        bundle = build_markdown_bundle(notes, student_name=student_name)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download viva_notes.md",
                data=viva_md,
                file_name="viva_notes.md",
                mime="text/markdown",
            )
        with c2:
            st.download_button(
                "Download full bundle.md",
                data=bundle,
                file_name="paper_to_project_notes.md",
                mime="text/markdown",
            )
