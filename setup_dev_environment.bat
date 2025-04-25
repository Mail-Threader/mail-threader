@echo off
:: Script to set up the development environment for the Enron Email Analysis Pipeline
:: This script automates the initial setup process for new team members on Windows

echo Setting up development environment for Enron Email Analysis Pipeline...

:: Check if Python 3.9+ is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

for /f "tokens=2" %%I in ('python --version 2^>^&1') do set python_version=%%I
for /f "tokens=1 delims=." %%I in ("%python_version%") do set python_major=%%I
for /f "tokens=2 delims=." %%I in ("%python_version%") do set python_minor=%%I

if %python_major% LSS 3 (
    echo Error: Python 3.9 or higher is required. Found Python %python_version%
    exit /b 1
)

if %python_major% EQU 3 (
    if %python_minor% LSS 9 (
        echo Error: Python 3.9 or higher is required. Found Python %python_version%
        exit /b 1
    )
)

echo ✓ Python %python_version% detected

:: Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate
echo ✓ Virtual environment activated

:: Install dependencies
echo Installing development dependencies...
pip install -e .[dev]
echo ✓ Dependencies installed

:: Install pre-commit hooks
echo Installing pre-commit hooks...
pre-commit install
echo ✓ Pre-commit hooks installed

:: Create data and output directories if they don't exist
if not exist data (
    echo Creating data directory...
    mkdir data
    echo ✓ Data directory created
) else (
    echo ✓ Data directory already exists
)

if not exist output (
    echo Creating output directory...
    mkdir output
    echo ✓ Output directory created
) else (
    echo ✓ Output directory already exists
)

echo.
echo 🎉 Development environment setup complete! 🎉
echo.
echo Next steps:
echo 1. Make sure to keep your virtual environment activated:
echo    .venv\Scripts\activate
echo.
echo 2. Read the GETTING_STARTED.md file for more information on how to use the project
echo.
echo 3. Run the tests to make sure everything is working:
echo    pytest
echo.
echo Happy coding! 🚀
