# Paper-to-Project Notes

Turn a research paper into mini-project notes + viva pack (problem, method, plan, proposal, viva Q&A).

Works from **PDF upload**, **pasted text**, or **title / DOI / arXiv id** search. Heuristics always run; an OpenAI key is optional (checker only).

---

## For other people (easiest)

### Option A — Double-click (Windows)

1. Install [Python 3.10+](https://www.python.org/downloads/) (tick **Add Python to PATH**).
2. Copy / clone this folder to their PC.
3. Double-click **`run.bat`**.
4. Browser opens at **http://localhost:8501**.

### Option B — Mac / Linux

```bash
cd path/to/P_T_PR
chmod +x run.sh
./run.sh
```

### Option C — Share the repo (GitHub)

```bash
git clone <your-repo-url>
cd P_T_PR
# Windows: double-click run.bat
# Mac/Linux: ./run.sh
```

Or manually:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Option D — Put it on the web (Streamlit Community Cloud)

So others use it in a browser with **no CMD**:

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo, main file: **`app.py`**.
4. Deploy → share the URL.

Optional: in Streamlit Cloud secrets, add `OPENAI_API_KEY` if you want the LLM checker.

### Option E — Same Wi‑Fi (you run it, others open the link)

On your machine:

```bat
run.bat
```

In another terminal (or edit the bat later):

```bat
.venv\Scripts\python.exe -m streamlit run app.py --server.address 0.0.0.0
```

Others on the same network open: `http://YOUR-PC-IP:8501`

---

## Features

- Problem / Method / Dataset / Metrics with evidence
- Task routing + contribution type (metric, model, survey, dataset, …)
- Fallback plans for theory / hardware / non-ML / metadata-only inputs
- Component status: implement / approximate / cite
- Table checks, proposal + viva downloads
- Optional LLM checker when `OPENAI_API_KEY` is set in a `.env` file

---

## Optional API key

Create a `.env` next to `app.py`:

```env
OPENAI_API_KEY=sk-...
```

Without a key the app still works (heuristics only).

---

## Tests / accuracy

```bash
pytest -q
python scripts/run_gold_eval.py
```

---

## Layout

```
app.py                 # Streamlit UI
run.bat / run.sh       # one-click start for others
requirements.txt
paper_to_project/      # extraction + proposal logic
tests/
gold/
```
