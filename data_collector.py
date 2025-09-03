"""
Data collector for NVIDIA-related news and social media content
"""

import requests
import pandas as pd
import feedparser
from bs4 import BeautifulSoup
import time
import logging
from datetime import datetime, timedelta
import os
import json
from config import KEYWORDS, RSS_FEEDS, RAW_DATA_DIR
from utils import clean_text

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsCollector:
    """Collect news articles about NVIDIA from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def collect_rss_news(self, max_articles=100):
        """
        Collect news from RSS feeds
        
        Args:
            max_articles (int): Maximum number of articles to collect
            
        Returns:
            list: List of news articles
        """
        articles = []
        
        for feed_url in RSS_FEEDS:
            try:
                logger.info(f"Collecting from RSS feed: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:max_articles//len(RSS_FEEDS)]:
                    # Check if article is related to NVIDIA
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    content = f"{title} {summary}"
                    
                    if any(keyword.lower() in content.lower() for keyword in KEYWORDS):
                        article = {
                            'title': title,
                            'content': summary,
                            'url': entry.get('link', ''),
                            'date': self._parse_date(entry.get('published', '')),
                            'source': feed_url,
                            'type': 'rss'
                        }
                        articles.append(article)
                
                # Be respectful with requests
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error collecting from RSS feed {feed_url}: {e}")
        
        return articles
    
    def collect_yahoo_finance_news(self, symbol="NVDA", max_articles=50):
        """
        Collect news from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol
            max_articles (int): Maximum number of articles
            
        Returns:
            list: List of news articles
        """
        articles = []
        
        try:
            # Yahoo Finance news URL
            url = f"https://query1.finance.yahoo.com/v1/finance/search"
            params = {
                'q': symbol,
                'quotes_count': 1,
                'news_count': max_articles
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            news_items = data.get('news', [])
            
            for item in news_items:
                article = {
                    'title': item.get('title', ''),
                    'content': item.get('summary', ''),
                    'url': item.get('link', ''),
                    'date': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'source': item.get('publisher', 'Yahoo Finance'),
                    'type': 'yahoo_finance'
                }
                articles.append(article)
            
            logger.info(f"Collected {len(articles)} articles from Yahoo Finance")
            
        except Exception as e:
            logger.error(f"Error collecting Yahoo Finance news: {e}")
        
        return articles
    
    def collect_google_news(self, query="NVIDIA", max_articles=30):
        """
        Collect news from Google News RSS
        
        Args:
            query (str): Search query
            max_articles (int): Maximum number of articles
            
        Returns:
            list: List of news articles
        """
        articles = []
        
        try:
            # Google News RSS URL
            url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:max_articles]:
                article = {
                    'title': entry.get('title', ''),
                    'content': entry.get('summary', ''),
                    'url': entry.get('link', ''),
                    'date': self._parse_date(entry.get('published', '')),
                    'source': entry.get('source', {}).get('href', 'Google News'),
                    'type': 'google_news'
                }
                articles.append(article)
            
            logger.info(f"Collected {len(articles)} articles from Google News")
            
        except Exception as e:
            logger.error(f"Error collecting Google News: {e}")
        
        return articles
    
    def scrape_article_content(self, url):
        """
        Scrape full article content from URL
        
        Args:
            url (str): Article URL
            
        Returns:
            str: Article content
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try different selectors for article content
            content_selectors = [
                'article',
                '.article-body',
                '.entry-content', 
                '.post-content',
                '.content',
                'main'
            ]
            
            content = ""
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text()
                    break
            
            if not content:
                # Fallback to body text
                content = soup.get_text()
            
            return clean_text(content)
            
        except Exception as e:
            logger.warning(f"Error scraping article content from {url}: {e}")
            return ""
    
    def _parse_date(self, date_str):
        """
        Parse date string to datetime object
        
        Args:
            date_str (str): Date string
            
        Returns:
            datetime: Parsed date
        """
        try:
            # Try different date formats
            formats = [
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # If no format works, return current time
            return datetime.now()
            
        except Exception:
            return datetime.now()
    
    def collect_all_news(self, max_total=200):
        """
        Collect news from all sources
        
        Args:
            max_total (int): Maximum total articles to collect
            
        Returns:
            pd.DataFrame: DataFrame with collected articles
        """
        all_articles = []
        
        # Collect from different sources
        logger.info("Starting news collection...")
        
        # Yahoo Finance
        yahoo_articles = self.collect_yahoo_finance_news(max_articles=max_total//3)
        all_articles.extend(yahoo_articles)
        
        # Google News
        google_articles = self.collect_google_news(max_articles=max_total//3)
        all_articles.extend(google_articles)
        
        # RSS feeds
        rss_articles = self.collect_rss_news(max_articles=max_total//3)
        all_articles.extend(rss_articles)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_articles)
        
        if not df.empty:
            # Remove duplicates based on title similarity
            df = df.drop_duplicates(subset=['title'], keep='first')
            
            # Sort by date
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date', ascending=False)
            
            # Filter recent articles (last 6 months)
            cutoff_date = datetime.now() - timedelta(days=180)
            df = df[df['date'] >= cutoff_date]
            
            logger.info(f"Collected {len(df)} unique articles")
        
        return df
    
    def save_data(self, df, filename=None):
        """
        Save collected data to file
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Optional filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nvidia_news_{timestamp}.csv"
        
        filepath = os.path.join(RAW_DATA_DIR, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Data saved to {filepath}")
        
        return filepath

# Sample function to generate mock data for testing
def generate_sample_data():
    """
    Generate sample news data for testing purposes
    
    Returns:
        pd.DataFrame: Sample news data
    """
    sample_articles = [
        {
            'title': 'NVIDIA Reports Record Q4 Revenue Driven by AI Demand',
            'content': 'NVIDIA Corporation today reported record revenue for the fourth quarter, driven by unprecedented demand for AI and data center solutions. The company saw significant growth in its data center business.',
            'url': 'https://example.com/nvidia-q4-2024',
            'date': datetime.now() - timedelta(days=5),
            'source': 'TechNews',
            'type': 'sample'
        },
        {
            'title': 'Jensen Huang Discusses Future of AI Computing at Conference',
            'content': 'NVIDIA CEO Jensen Huang presented the companys vision for the future of AI computing, highlighting new GPU architectures and partnerships with major cloud providers.',
            'url': 'https://example.com/jensen-ai-conference',
            'date': datetime.now() - timedelta(days=10),
            'source': 'AI Weekly',
            'type': 'sample'
        },
        {
            'title': 'NVIDIA Stock Reaches New All-Time High Amid AI Boom',
            'content': 'Shares of NVIDIA reached a new all-time high today as investors continue to bet on the companys dominant position in the AI chip market. Analysts remain bullish on the stock.',
            'url': 'https://example.com/nvidia-stock-high',
            'date': datetime.now() - timedelta(days=15),
            'source': 'MarketWatch',
            'type': 'sample'
        },
        {
            'title': 'Competition Heats Up in AI Chip Market',
            'content': 'While NVIDIA maintains its lead, competitors are launching new AI chips that could challenge the companys dominance. Industry experts discuss the evolving competitive landscape.',
            'url': 'https://example.com/ai-chip-competition',
            'date': datetime.now() - timedelta(days=20),
            'source': 'Tech Insider',
            'type': 'sample'
        },
        {
            'title': 'NVIDIA Announces New Partnership with Major Automaker',
            'content': 'NVIDIA has announced a strategic partnership to provide AI computing platforms for autonomous vehicle development, expanding its reach in the automotive sector.',
            'url': 'https://example.com/nvidia-auto-partnership',
            'date': datetime.now() - timedelta(days=25),
            'source': 'AutoTech',
            'type': 'sample'
        }
    ]
    
    return pd.DataFrame(sample_articles)

if __name__ == "__main__":
    collector = NewsCollector()
    
    # Try to collect real data, fall back to sample data if needed
    try:
        df = collector.collect_all_news(max_total=50)
        if df.empty:
            raise Exception("No articles collected")
    except Exception as e:
        logger.warning(f"Using sample data due to error: {e}")
        df = generate_sample_data()
    
    # Save the data
    filepath = collector.save_data(df)
    print(f"Data collection complete. Saved to: {filepath}")
