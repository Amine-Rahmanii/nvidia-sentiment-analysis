@echo off
echo Setting up NVIDIA Sentiment Analysis Project...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found. Checking version...
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing required packages...
pip install -r requirements.txt

REM Download NLTK data
echo.
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo.
echo Setup complete!
echo.
echo To run the application:
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Run Streamlit app: streamlit run app.py
echo.
pause
