#!/bin/bash

echo "Setting up NVIDIA Sentiment Analysis Project..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "Python found. Checking version..."
python3 --version

# Create virtual environment
echo
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo
echo "Installing required packages..."
pip install -r requirements.txt

# Download NLTK data
echo
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo
echo "Setup complete!"
echo
echo "To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run Streamlit app: streamlit run app.py"
echo
