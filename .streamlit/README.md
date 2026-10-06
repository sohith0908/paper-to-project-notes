# Streamlit Cloud

This app is ready for [Streamlit Community Cloud](https://share.streamlit.io).

1. Push the repo to **GitHub** (public).
2. Sign in at https://share.streamlit.io with GitHub.
3. **New app** → select this repo → Main file path: `app.py` → Deploy.
4. Share the public URL with anyone.

Optional secrets (Cloud → App settings → Secrets):

```toml
OPENAI_API_KEY = "sk-..."
```

Without a key, heuristics still work for everyone.
