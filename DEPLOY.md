# Put Paper-to-Project on the public web

Anyone can open a URL — no CMD for them.

## 1. Log in to GitHub CLI (one time on your PC)

Open **cmd** or PowerShell in this folder:

```bat
gh auth login
```

Choose: **GitHub.com** → **HTTPS** → **Login with a web browser** → finish in the browser.

## 2. Create a public repo and push

```bat
cd /d d:\GITHUB\P_T_PR
gh repo create paper-to-project-notes --public --source=. --remote=origin --push
```

If the name is taken, use another name, e.g. `paper-to-project-notes-rishu`.

Or double-click **`deploy_github.bat`** after `gh auth login`.

## 3. Deploy on Streamlit Cloud (free)

1. Open https://share.streamlit.io and sign in with **GitHub**.
2. Click **New app**.
3. Repository: your new repo  
   Branch: `main`  
   Main file path: **`app.py`**
4. Click **Deploy**.
5. Copy the public URL (like `https://xxxx.streamlit.app`) and share it.

Optional: App settings → **Secrets**:

```toml
OPENAI_API_KEY = "sk-..."
```

Without a key the app still works for everyone.

## Done

Anyone with the Streamlit URL can upload a PDF / paste text / search a title — no install.
