"""
Utility functions for the NVIDIA sentiment analysis project
"""

import re
import string
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Clean and preprocess text data
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text

def remove_stopwords(text, language='english'):
    """
    Remove stopwords from text
    
    Args:
        text (str): Text to process
        language (str): Language for stopwords
        
    Returns:
        str: Text without stopwords
    """
    try:
        stop_words = set(stopwords.words(language))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
        return ' '.join(filtered_text)
    except Exception as e:
        logger.warning(f"Error removing stopwords: {e}")
        return text

def preprocess_text(text):
    """
    Complete text preprocessing pipeline
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Preprocessed text
    """
    # Clean text
    text = clean_text(text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove stopwords
    text = remove_stopwords(text)
    
    return text

def calculate_correlation(sentiment_data, stock_data):
    """
    Calculate correlation between sentiment and stock price movements
    
    Args:
        sentiment_data (pd.DataFrame): Sentiment data with date index
        stock_data (pd.DataFrame): Stock data with date index
        
    Returns:
        dict: Correlation metrics
    """
    try:
        if sentiment_data.empty or stock_data.empty:
            return {}
        
        # Ensure both DataFrames have datetime index
        if hasattr(sentiment_data.index, 'tz') and sentiment_data.index.tz is not None:
            sentiment_data = sentiment_data.copy()
            sentiment_data.index = sentiment_data.index.tz_localize(None)
        if hasattr(stock_data.index, 'tz') and stock_data.index.tz is not None:
            stock_data = stock_data.copy()
            stock_data.index = stock_data.index.tz_localize(None)
        
        # Merge data on date
        merged_data = pd.merge(sentiment_data, stock_data, left_index=True, right_index=True, how='inner')
        
        if merged_data.empty:
            return {}
        
        # Calculate correlations
        correlations = {}
        
        # Check for sentiment score column (could be 'sentiment_score' or 'sentiment_score_mean')
        sentiment_col = None
        if 'sentiment_score_mean' in merged_data.columns:
            sentiment_col = 'sentiment_score_mean'
        elif 'sentiment_score' in merged_data.columns:
            sentiment_col = 'sentiment_score'
        
        if sentiment_col and 'Close' in merged_data.columns:
            correlations['sentiment_price'] = merged_data[sentiment_col].corr(merged_data['Close'])
        
        if sentiment_col and 'Volume' in merged_data.columns:
            correlations['sentiment_volume'] = merged_data[sentiment_col].corr(merged_data['Volume'])
            
        # Calculate price change correlation
        if sentiment_col and 'Close' in merged_data.columns:
            merged_data['price_change'] = merged_data['Close'].pct_change()
            correlations['sentiment_price_change'] = merged_data[sentiment_col].corr(merged_data['price_change'])
        
        return correlations
    
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return {}

def aggregate_daily_sentiment(df):
    """
    Aggregate sentiment data by day
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment data
        
    Returns:
        pd.DataFrame: Daily aggregated sentiment data
    """
    try:
        if df.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Convert date column to datetime if it's not already
        if 'date' in df_copy.columns:
            df_copy['date'] = pd.to_datetime(df_copy['date'])
            df_copy.set_index('date', inplace=True)
        elif df_copy.index.name != 'date':
            # If no date column but index might be dates
            try:
                df_copy.index = pd.to_datetime(df_copy.index)
            except:
                logger.warning("No valid date column found for aggregation")
                return pd.DataFrame()
        
        # Check if required columns exist
        required_cols = ['sentiment_score']
        available_cols = {col: col for col in required_cols if col in df_copy.columns}
        
        optional_cols = ['positive_prob', 'neutral_prob', 'negative_prob']
        for col in optional_cols:
            if col in df_copy.columns:
                available_cols[col] = col
        
        if not available_cols:
            logger.warning("No sentiment columns found for aggregation")
            return pd.DataFrame()
        
        # Define aggregation functions
        agg_dict = {}
        for col in available_cols:
            if col == 'sentiment_score':
                agg_dict[col] = ['mean', 'std', 'count']
            else:
                agg_dict[col] = 'mean'
        
        # Group by date and calculate daily metrics
        daily_sentiment = df_copy.groupby(df_copy.index.date).agg(agg_dict).round(4)
        
        # Flatten column names
        daily_sentiment.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in daily_sentiment.columns]
        
        # Reset index to make date a column again
        daily_sentiment.reset_index(inplace=True)
        daily_sentiment.rename(columns={'index': 'date'}, inplace=True)
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        daily_sentiment.set_index('date', inplace=True)
        
        return daily_sentiment
    
    except Exception as e:
        logger.error(f"Error aggregating daily sentiment: {e}")
        return pd.DataFrame()

def format_number(num):
    """
    Format large numbers for display
    
    Args:
        num (float): Number to format
        
    Returns:
        str: Formatted number string
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return f"{num:.2f}"

def validate_dataframe(df, required_columns):
    """
    Validate that a DataFrame has required columns
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (list): List of required column names
        
    Returns:
        bool: True if valid, False otherwise
    """
    if df is None or df.empty:
        logger.warning("DataFrame is None or empty")
        return False
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing required columns: {missing_columns}")
        return False
    
    return True
