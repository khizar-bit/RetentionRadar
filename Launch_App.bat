@echo off
echo Starting Retention Radar...
cd /d "%~dp0"

REM Check for virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found in .venv!
    echo Please make sure you have set up the environment.
    pause
    exit /b
)

REM Run the app
echo Launching Streamlit...
echo The browser should open automatically.
python -m streamlit run app.py

pause
