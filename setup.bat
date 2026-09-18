@echo off
REM One-command setup for Windows.
REM   setup.bat

echo ==^> Creating virtual environment in .venv
if exist .venv (
  echo     .venv already exists, reusing it
) else (
  python -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment. Install Python 3.10+ and retry.
    exit /b 1
  )
)

call .venv\Scripts\activate.bat

echo ==^> Installing dependencies
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

if not exist .env (
  copy .env.example .env >nul
  echo ==^> Created .env - open it and add your Groq key from https://console.groq.com
) else (
  echo ==^> .env already present, leaving it alone
)

echo ==^> Building the vector index
python -m knowledge.ingest

echo ==^> Running tests
python -m pytest -q

echo.
echo Setup complete.
echo   .venv\Scripts\activate
echo   uvicorn app.main:app --reload
echo Then open http://localhost:8000
