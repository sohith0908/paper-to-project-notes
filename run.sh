#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo ""
echo " Paper-to-Project Notes"
echo " ----------------------"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 not found. Install Python 3.10+ then retry."
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

echo "Installing / updating dependencies..."
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

echo ""
echo "Starting app... open http://localhost:8501"
echo "Press Ctrl+C to stop."
echo ""
.venv/bin/python -m streamlit run app.py
