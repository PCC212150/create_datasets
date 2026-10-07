@echo off
rem Same as run.bat but keeps the console: you see the model loading line, the
rem per-image predictions, the warnings, and any traceback if it crashes.
rem Use this one when something looks wrong.
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo runtime\ not found. Run setup_runtime.bat first.
  pause
  exit /b 1
)
"runtime\python.exe" "annotate_root.py" %*
if errorlevel 1 pause
