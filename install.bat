@echo off
py -3.11 -m venv .venv
if errorlevel 1 (echo Python 3.14.6 is required & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pause
