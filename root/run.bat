@echo off
rem Start the root annotation tool using the portable runtime (no console window).
rem Args are passed through, e.g.:  run.bat --image root_C019-1_20241229CK
rem                                run.bat --prefetch
rem If it exits with an error, re-run it with a console so you can read why.
cd /d "%~dp0"
if not exist "runtime\pythonw.exe" (
  echo runtime\ not found. Run setup_runtime.bat first.
  pause
  exit /b 1
)
"runtime\pythonw.exe" "annotate_root.py" %*
if errorlevel 1 (
  echo.
  echo The tool exited with an error. Re-running with a console so you can see it:
  echo.
  "runtime\python.exe" "annotate_root.py" %*
  pause
)
