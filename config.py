"""
Configuration file for NVIDIA sentiment analysis project
"""

import os
from datetime import datetime, timedelta

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Stock symbol
STOCK_SYMBOL = "NVDA"

# Date range for analysis (last 6 months by default)
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=180)

# FinBERT model configuration
FINBERT_MODEL = "ProsusAI/finbert"

# News sources and keywords
KEYWORDS = ["NVIDIA", "NVDA", "Jensen Huang", "AI chips", "GPU"]

# RSS feeds for financial news
RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://feeds.reuters.com/business",
    "https://www.sec.gov/rss/seccareers.xml"
]

# Sentiment labels
SENTIMENT_LABELS = {
    0: "negative",
    1: "neutral", 
    2: "positive"
}

# Visualization settings
CHART_COLORS = {
    "positive": "#2E8B57",
    "neutral": "#FFD700", 
    "negative": "#DC143C",
    "stock": "#1f77b4"
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "NVIDIA Sentiment Analysis",
    "page_icon": "📊",
    "layout": "wide"
}
