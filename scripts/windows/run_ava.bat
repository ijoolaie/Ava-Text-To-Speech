@echo off
REM AVA Native launcher (Windows) — starts the API + web UI with a model bundle
REM Usage: run_ava.bat [path\to\bundle]
setlocal
if "%~1"=="" (
  if "%AVA_NATIVE_BUNDLES%"=="" (
    echo Usage: run_ava.bat path\to\ava_n1_bundle
    exit /b 2
  )
) else (
  set "AVA_NATIVE_BUNDLES=n1=%~1"
)
where python >nul 2>nul || (echo Python not found in PATH & exit /b 2)
python -m uvicorn ava.api.server:app --host 127.0.0.1 --port 8300
endlocal
