@echo off
REM ============================================================================
REM  FORGE — one-click evaluation reproduction for Windows.
REM
REM  Just double-click this file. It will:
REM    1. install 'uv' (the environment manager) if it is missing,
REM    2. set up Python + all dependencies (uses your NVIDIA GPU if present),
REM    3. download the model + data from the internet,
REM    4. run the evaluation and print the results.
REM
REM  Needs: an internet connection and ~20 GB of free disk space.
REM  The first run takes a while (large download). Leave the window open.
REM  Results are written into the 'logs' folder next to this file.
REM
REM  Advanced: pass flags through, e.g.  reproduce.bat --full
REM ============================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM --- 0. force UTF-8 -------------------------------------------------------
REM Legacy Windows console code pages (e.g. Hebrew cp1255) cannot encode the
REM arrows/symbols the eval scripts print, which would crash with
REM UnicodeEncodeError. Switch the console to UTF-8 and put Python in UTF-8
REM mode so this works on any machine regardless of regional settings.
chcp 65001 >nul 2>nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

REM --- 1. ensure uv is installed ---------------------------------------------
where uv >nul 2>nul
if errorlevel 1 (
    echo [setup] 'uv' was not found. Installing it now ^(one-time^)...
    powershell -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    REM uv installs to %USERPROFILE%\.local\bin — add it to PATH for this session
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    where uv >nul 2>nul
    if errorlevel 1 (
        echo.
        echo [setup] uv was installed but is not visible yet.
        echo         Please CLOSE this window and double-click reproduce.bat again.
        echo.
        pause
        exit /b 1
    )
)

REM --- 2. provision environment ----------------------------------------------
echo.
echo [1/2] Setting up the environment ^(this can take several minutes the first time^)...
uv sync
if errorlevel 1 (
    echo.
    echo [error] Environment setup failed. Check your internet connection and try again.
    pause
    exit /b 1
)

REM --- 3. run the reproduction ------------------------------------------------
echo.
echo [2/2] Running the evaluation ^(downloads model + data on first run^)...
uv run python reproduce.py %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo Done. The numbers above should match the paper; full results are in the 'logs' folder.
) else (
    echo The run did not finish cleanly ^(exit %RC%^). See the messages above.
)
echo.
pause
exit /b %RC%
