@echo off
setlocal enabledelayedexpansion

REM Launch Madakappy using the Conda env on Windows.
REM Double-click this file to run the app.

set "ENV_NAME=Madakappy"
set "APP_ROOT=%~dp0.."
pushd "%APP_ROOT%"

REM Locate conda.bat
set "CONDA_BAT="
if exist "%UserProfile%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%UserProfile%\miniconda3\condabin\conda.bat"
if "%CONDA_BAT%"=="" if exist "%UserProfile%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%UserProfile%\anaconda3\condabin\conda.bat"
if "%CONDA_BAT%"=="" for /f "usebackq delims=" %%i in (`where conda.bat 2^>nul`) do set "CONDA_BAT=%%i"

if "%CONDA_BAT%"=="" (
  echo [ERROR] Could not find conda.bat on this system.
  echo Please install Miniconda/Anaconda or add conda to PATH.
  pause
  popd
  exit /b 1
)

call "%CONDA_BAT%" activate "%ENV_NAME%"
if errorlevel 1 (
  echo [ERROR] Failed to activate Conda environment "%ENV_NAME%".
  echo Ensure the environment exists: conda env list
  pause
  popd
  exit /b 1
)

set "MADAKAPPY_UI=flet"
python -m app.main
set EXITCODE=%ERRORLEVEL%

if NOT %EXITCODE%==0 (
  echo.
  echo The application exited with code %EXITCODE%.
  echo Check the console output for details.
  pause
)

popd
endlocal

