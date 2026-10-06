@echo off
setlocal
cd /d "%~dp0"

echo.
echo  Publish this project to a PUBLIC GitHub repo
echo  --------------------------------------------
echo.

where gh >nul 2>&1
if errorlevel 1 (
  echo GitHub CLI not found. Install from https://cli.github.com/ then retry.
  pause
  exit /b 1
)

gh auth status >nul 2>&1
if errorlevel 1 (
  echo You are not logged in. Starting browser login...
  gh auth login -p https -w
  if errorlevel 1 (
    echo Login failed.
    pause
    exit /b 1
  )
)

set REPO_NAME=paper-to-project-notes
echo Creating public repo: %REPO_NAME%
gh repo create %REPO_NAME% --public --source=. --remote=origin --push
if errorlevel 1 (
  echo.
  echo Repo create/push failed. Name may be taken. Trying alternate name...
  set REPO_NAME=paper-to-project-notes-app
  gh repo create %REPO_NAME% --public --source=. --remote=origin --push
)

if errorlevel 1 (
  echo.
  echo Still failed. If origin already exists, try:
  echo   git push -u origin main
  pause
  exit /b 1
)

echo.
echo SUCCESS. Next:
echo   1. Open https://share.streamlit.io
echo   2. New app -^> select repo "%REPO_NAME%" -^> main file: app.py
echo   3. Deploy and share the URL with everyone.
echo.
gh repo view --web
pause
