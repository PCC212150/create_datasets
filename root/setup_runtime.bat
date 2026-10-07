@echo off
rem ============================================================================
rem  Build the portable runtime for the root annotation tool.
rem  Result: runtime\  =  Python 3.12 embeddable + deps + torch (CPU only)
rem  Needs internet ONCE (~400 MB). After that the folder runs fully offline,
rem  and the whole folder can be copied to another Windows x64 machine.
rem
rem  Everything here is ASCII on purpose: a .bat containing non-ASCII text gets
rem  mangled by cmd.exe's code page, and the failure mode is baffling.
rem
rem  Two deliberate choices:
rem   1. Not "python -m venv": a venv records the ABSOLUTE path of the Python
rem      that created it (here C:\Users\21215\.conda\envs\pcc) and needs that
rem      install to exist. Copied to another machine it simply won't start.
rem      python.org's "embeddable" package is self-contained and relocatable.
rem   2. Python itself comes from the npmmirror mirror first, python.org second:
rem      measured on this machine, python.org served the 11 MB zip at ~10 KB/s
rem      while npmmirror did 4.8 MB/s (the same file). Falls back automatically.
rem ============================================================================
setlocal
cd /d "%~dp0"

set PYVER=3.12.10
set ZIP=%TEMP%\python-embed-amd64.zip
set GETPIP=%TEMP%\get-pip.py
set PYURL_MIRROR=https://registry.npmmirror.com/-/binary/python/%PYVER%/python-%PYVER%-embed-amd64.zip
set PYURL_ORG=https://www.python.org/ftp/python/%PYVER%/python-%PYVER%-embed-amd64.zip
set MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
set TORCHIDX=https://download.pytorch.org/whl/cpu

if not exist "runtime\python.exe" goto :download
echo [1/5] runtime\python.exe already exists - skipping download and extraction
goto :packages

:download
echo [1/5] downloading Python %PYVER% embeddable (~11 MB)
if not exist "runtime" mkdir "runtime"
del /q "%ZIP%" 2>nul
curl.exe -L --fail -o "%ZIP%" "%PYURL_MIRROR%"
if not errorlevel 1 goto :extract
echo       mirror failed - falling back to python.org (may be slow)
curl.exe -L --fail -o "%ZIP%" "%PYURL_ORG%"
if errorlevel 1 goto :err

:extract
rem Use tar.exe (bsdtar, ships with Windows 10 1803+) instead of PowerShell's
rem Expand-Archive: no execution-policy involvement, and it also works on
rem machines where PowerShell is locked down.
rem The FULL path matters: if Git-for-Windows (or any MSYS/Cygwin toolchain) is
rem installed and its bin dir is on PATH, a bare "tar" resolves to GNU tar,
rem which reads a "C:\..." argument as remote-host:path and dies with
rem "tar: Cannot connect to C: resolve failed".
echo [2/5] extracting into runtime\
set TAREXE=%SystemRoot%\System32\tar.exe
if not exist "%TAREXE%" set TAREXE=tar.exe
"%TAREXE%" -xf "%ZIP%" -C "runtime"
if errorlevel 1 goto :err

rem The embeddable build ships with site-packages DISABLED (that is the whole
rem point of it). These four lines turn it on and point it at runtime\Lib\site-packages.
echo [3/5] enabling site-packages (python312._pth)
> "runtime\python312._pth" echo python312.zip
>> "runtime\python312._pth" echo .
>> "runtime\python312._pth" echo Lib\site-packages
>> "runtime\python312._pth" echo import site

echo [4/5] bootstrapping pip
curl.exe -L --fail -o "%GETPIP%" "https://bootstrap.pypa.io/get-pip.py"
if errorlevel 1 goto :err
"runtime\python.exe" "%GETPIP%" -i %MIRROR%
if errorlevel 1 (
  echo       mirror failed - retrying with the default index
  "runtime\python.exe" "%GETPIP%"
)
if errorlevel 1 goto :err

:packages
echo [5/5] installing packages (long step, ~400 MB to download)
"runtime\python.exe" -m pip install -r requirements.txt -i %MIRROR%
if errorlevel 1 (
  echo       mirror failed - retrying with the default index
  "runtime\python.exe" -m pip install -r requirements.txt
)
if errorlevel 1 goto :err

rem torch is NOT in requirements.txt on purpose: PyPI's Windows torch bundles CUDA
rem (2-3 GB installed) and this tool runs fine on CPU. The CPU build is ~350 MB.
rem No fallback to PyPI here - silently installing the CUDA build would defeat the
rem point of a folder you can hand to someone.
echo       installing torch (CPU build, ~350 MB)
"runtime\python.exe" -m pip install torch --index-url %TORCHIDX%
if errorlevel 1 goto :err

echo.
echo Done. runtime\ is ready.
echo   run.bat          start the tool
echo   run_debug.bat    start with a console (shows errors and progress)
echo   verify:          run_debug.bat --selftest
exit /b 0

:err
echo.
echo *** FAILED ***  read the message above (network? disk space?)
exit /b 1
