"""
Sentiment analysis using FinBERT for financial text
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import logging
from datetime import datetime
import os
from config import FINBERT_MODEL, SENTIMENT_LABELS, PROCESSED_DATA_DIR
from utils import preprocess_text, clean_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinBERTAnalyzer:
    """Sentiment analysis using FinBERT model"""
    
    def __init__(self, model_name=FINBERT_MODEL):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the FinBERT model and tokenizer"""
        try:
            logger.info(f"Loading FinBERT model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            logger.info("Falling back to simple sentiment analyzer")
            self._create_fallback_analyzer()
    
    def _create_fallback_analyzer(self):
        """Create a simple fallback sentiment analyzer"""
        try:
            from textblob import TextBlob
            self.pipeline = "fallback"
            logger.info("Using TextBlob as fallback sentiment analyzer")
        except ImportError:
            logger.warning("TextBlob not available, using rule-based analyzer")
            self.pipeline = "rule_based"
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a single text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        if not text or not isinstance(text, str):
            return self._default_sentiment()
        
        # Clean and preprocess text
        cleaned_text = clean_text(text)
        
        if not cleaned_text.strip():
            return self._default_sentiment()
        
        try:
            if self.pipeline == "fallback":
                return self._textblob_sentiment(cleaned_text)
            elif self.pipeline == "rule_based":
                return self._rule_based_sentiment(cleaned_text)
            else:
                return self._finbert_sentiment(cleaned_text)
                
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {e}")
            return self._default_sentiment()
    
    def _finbert_sentiment(self, text):
        """
        Analyze sentiment using FinBERT
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment results
        """
        try:
            # Truncate text if too long (FinBERT max length is 512 tokens)
            if len(text) > 1000:  # Conservative limit
                text = text[:1000]
            
            # Get prediction
            result = self.pipeline(text)[0]
            
            # Map FinBERT labels to our format
            label_mapping = {
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral'
            }
            
            sentiment = label_mapping.get(result['label'].lower(), 'neutral')
            confidence = result['score']
            
            # Calculate sentiment score (-1 to 1)
            if sentiment == 'positive':
                sentiment_score = confidence
            elif sentiment == 'negative':
                sentiment_score = -confidence
            else:
                sentiment_score = 0.0
            
            # Get probabilities for all classes
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                      padding=True, max_length=512)
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
            
            # Map probabilities to our labels
            if len(probs) == 3:
                negative_prob, neutral_prob, positive_prob = probs
            else:
                # Fallback if unexpected number of classes
                negative_prob = probs[0] if sentiment == 'negative' else 0.2
                neutral_prob = probs[0] if sentiment == 'neutral' else 0.6
                positive_prob = probs[0] if sentiment == 'positive' else 0.2
            
            return {
                'sentiment': sentiment,
                'sentiment_score': round(sentiment_score, 4),
                'confidence': round(confidence, 4),
                'positive_prob': round(positive_prob, 4),
                'neutral_prob': round(neutral_prob, 4),
                'negative_prob': round(negative_prob, 4)
            }
            
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment analysis: {e}")
            return self._default_sentiment()
    
    def _textblob_sentiment(self, text):
        """
        Fallback sentiment analysis using TextBlob
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment results
        """
        try:
            from textblob import TextBlob
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            # Map polarity to sentiment labels
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Convert to probabilities
            positive_prob = max(0, polarity)
            negative_prob = max(0, -polarity)
            neutral_prob = 1 - abs(polarity)
            
            # Normalize probabilities
            total = positive_prob + negative_prob + neutral_prob
            positive_prob /= total
            negative_prob /= total
            neutral_prob /= total
            
            return {
                'sentiment': sentiment,
                'sentiment_score': round(polarity, 4),
                'confidence': round(abs(polarity), 4),
                'positive_prob': round(positive_prob, 4),
                'neutral_prob': round(neutral_prob, 4),
                'negative_prob': round(negative_prob, 4)
            }
            
        except Exception as e:
            logger.error(f"Error in TextBlob sentiment analysis: {e}")
            return self._default_sentiment()
    
    def _rule_based_sentiment(self, text):
        """
        Simple rule-based sentiment analysis
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment results
        """
        try:
            text_lower = text.lower()
            
            # Define positive and negative keywords
            positive_words = [
                'positive', 'good', 'great', 'excellent', 'strong', 'growth', 'profit',
                'gain', 'rise', 'increase', 'record', 'high', 'success', 'beat',
                'outperform', 'bullish', 'optimistic', 'breakthrough', 'innovative'
            ]
            
            negative_words = [
                'negative', 'bad', 'poor', 'weak', 'decline', 'loss', 'fall',
                'decrease', 'low', 'failure', 'miss', 'underperform', 'bearish',
                'pessimistic', 'concern', 'risk', 'challenge', 'problem'
            ]
            
            # Count positive and negative words
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            # Calculate sentiment score
            total_count = positive_count + negative_count
            
            if total_count == 0:
                sentiment = 'neutral'
                sentiment_score = 0.0
                confidence = 0.5
            else:
                sentiment_score = (positive_count - negative_count) / max(total_count, 1)
                
                if sentiment_score > 0.2:
                    sentiment = 'positive'
                elif sentiment_score < -0.2:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                
                confidence = abs(sentiment_score)
            
            # Calculate probabilities
            if sentiment == 'positive':
                positive_prob = 0.7
                negative_prob = 0.1
                neutral_prob = 0.2
            elif sentiment == 'negative':
                positive_prob = 0.1
                negative_prob = 0.7
                neutral_prob = 0.2
            else:
                positive_prob = 0.3
                negative_prob = 0.3
                neutral_prob = 0.4
            
            return {
                'sentiment': sentiment,
                'sentiment_score': round(sentiment_score, 4),
                'confidence': round(confidence, 4),
                'positive_prob': round(positive_prob, 4),
                'neutral_prob': round(neutral_prob, 4),
                'negative_prob': round(negative_prob, 4)
            }
            
        except Exception as e:
            logger.error(f"Error in rule-based sentiment analysis: {e}")
            return self._default_sentiment()
    
    def _default_sentiment(self):
        """Return default neutral sentiment"""
        return {
            'sentiment': 'neutral',
            'sentiment_score': 0.0,
            'confidence': 0.5,
            'positive_prob': 0.33,
            'neutral_prob': 0.34,
            'negative_prob': 0.33
        }
    
    def analyze_dataframe(self, df, text_column='content'):
        """
        Analyze sentiment for all texts in a DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame with text data
            text_column (str): Name of the column containing text
            
        Returns:
            pd.DataFrame: DataFrame with sentiment analysis results
        """
        if df.empty or text_column not in df.columns:
            logger.warning(f"DataFrame is empty or missing {text_column} column")
            return df
        
        logger.info(f"Analyzing sentiment for {len(df)} texts")
        
        # Initialize result columns
        sentiment_results = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            
            # Combine title and content if both available
            if 'title' in df.columns and pd.notna(row['title']):
                text = f"{row['title']} {text}"
            
            # Analyze sentiment
            result = self.analyze_text(text)
            sentiment_results.append(result)
            
            # Log progress
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} texts")
        
        # Add sentiment results to DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df], axis=1)
        
        logger.info("Sentiment analysis complete")
        return result_df
    
    def get_sentiment_summary(self, df):
        """
        Get summary statistics of sentiment analysis
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment results
            
        Returns:
            dict: Sentiment summary statistics
        """
        if df.empty or 'sentiment' not in df.columns:
            return {}
        
        try:
            summary = {
                'total_articles': len(df),
                'positive_count': len(df[df['sentiment'] == 'positive']),
                'neutral_count': len(df[df['sentiment'] == 'neutral']),
                'negative_count': len(df[df['sentiment'] == 'negative']),
                'avg_sentiment_score': df['sentiment_score'].mean(),
                'avg_confidence': df['confidence'].mean(),
                'sentiment_distribution': df['sentiment'].value_counts().to_dict()
            }
            
            # Calculate percentages
            total = summary['total_articles']
            if total > 0:
                summary['positive_pct'] = (summary['positive_count'] / total) * 100
                summary['neutral_pct'] = (summary['neutral_count'] / total) * 100
                summary['negative_pct'] = (summary['negative_count'] / total) * 100
            
            return summary
            
        except Exception as e:
            logger.error(f"Error calculating sentiment summary: {e}")
            return {}
    
    def save_results(self, df, filename=None):
        """
        Save sentiment analysis results
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment results
            filename (str): Optional filename
            
        Returns:
            str: Filepath where results were saved
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_results_{timestamp}.csv"
        
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Sentiment results saved to {filepath}")
        
        return filepath

if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = FinBERTAnalyzer()
    
    # Test texts
    test_texts = [
        "NVIDIA reports record quarterly revenue driven by strong AI demand",
        "Concerns grow over NVIDIA's high valuation amid market volatility",
        "NVIDIA announces new GPU architecture for data centers",
        "Competition intensifies in the AI chip market"
    ]
    
    print("Testing sentiment analysis:")
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"Text: {text[:50]}...")
        print(f"Sentiment: {result['sentiment']} (score: {result['sentiment_score']:.3f})")
        print("-" * 50)
