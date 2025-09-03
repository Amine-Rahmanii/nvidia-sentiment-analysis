"""
NVIDIA Sentiment Analysis Dashboard - Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from config import STREAMLIT_CONFIG, STOCK_SYMBOL
from data_collector import NewsCollector, generate_sample_data
from stock_data import StockDataCollector, generate_sample_stock_data
from sentiment_analyzer import FinBERTAnalyzer
from visualizer import SentimentVisualizer
from utils import aggregate_daily_sentiment, calculate_correlation, format_number

# Configure Streamlit page
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"]
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sentiment-positive {
        color: #2E8B57;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #DC143C;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #FFD700;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_news_data():
    """Load and cache news data"""
    try:
        collector = NewsCollector()
        df = collector.collect_all_news(max_total=100)
        if df.empty:
            st.warning("No news data collected. Using sample data.")
            df = generate_sample_data()
        return df
    except Exception as e:
        st.error(f"Error loading news data: {e}")
        return generate_sample_data()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_data():
    """Load and cache stock data"""
    try:
        collector = StockDataCollector()
        df = collector.get_historical_data(period="6mo")
        if df.empty:
            st.warning("No stock data retrieved. Using sample data.")
            df = generate_sample_stock_data()
        return df
    except Exception as e:
        st.error(f"Error loading stock data: {e}")
        return generate_sample_stock_data()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def analyze_sentiment(news_df):
    """Analyze sentiment and cache results"""
    try:
        analyzer = FinBERTAnalyzer()
        sentiment_df = analyzer.analyze_dataframe(news_df)
        return sentiment_df
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return news_df

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 NVIDIA Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # Data loading options
    st.sidebar.subheader("Data Options")
    refresh_data = st.sidebar.button("🔄 Refresh Data")
    
    if refresh_data:
        st.cache_data.clear()
        st.rerun()
    
    # Date range selector
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date, end_date),
        max_value=end_date
    )
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_correlations = st.sidebar.checkbox("Show Correlations", value=True)
    show_source_analysis = st.sidebar.checkbox("Show Source Analysis", value=True)
    
    # Load data
    with st.spinner("Loading data..."):
        news_df = load_news_data()
        stock_df = load_stock_data()
    
    # Analyze sentiment
    with st.spinner("Analyzing sentiment..."):
        sentiment_df = analyze_sentiment(news_df)
    
    # Filter data by date range
    if len(date_range) == 2:
        start_filter, end_filter = date_range
        sentiment_df = sentiment_df[
            (pd.to_datetime(sentiment_df['date']).dt.date >= start_filter) &
            (pd.to_datetime(sentiment_df['date']).dt.date <= end_filter)
        ]
        stock_df = stock_df[
            (pd.to_datetime(stock_df['Date']).dt.date >= start_filter) &
            (pd.to_datetime(stock_df['Date']).dt.date <= end_filter)
        ]
    
    # Create visualizer
    visualizer = SentimentVisualizer()
    
    # Summary metrics
    st.header("📈 Key Metrics")
    
    if not sentiment_df.empty and not stock_df.empty:
        metrics = visualizer.create_summary_metrics(sentiment_df, stock_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Articles Analyzed",
                value=metrics.get('total_articles', 0)
            )
        
        with col2:
            avg_sentiment = metrics.get('avg_sentiment', 0)
            sentiment_class = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"
            st.metric(
                "Average Sentiment",
                value=f"{avg_sentiment:.3f}",
                delta=sentiment_class
            )
        
        with col3:
            latest_price = metrics.get('latest_price', 0)
            price_change = metrics.get('price_change', 0)
            st.metric(
                "Latest Stock Price",
                value=f"${latest_price:.2f}",
                delta=f"${price_change:.2f}"
            )
        
        with col4:
            price_change_pct = metrics.get('price_change_pct', 0)
            st.metric(
                "Price Change %",
                value=f"{price_change_pct:.2f}%"
            )
    
    # Sentiment distribution
    st.header("🎯 Sentiment Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Timeline plot
        daily_sentiment = aggregate_daily_sentiment(sentiment_df)
        if not daily_sentiment.empty and not stock_df.empty:
            timeline_fig = visualizer.plot_sentiment_timeline(daily_sentiment, stock_df)
            st.plotly_chart(timeline_fig, use_container_width=True, key="timeline_chart")
        else:
            st.warning("Insufficient data for timeline visualization")
    
    with col2:
        # Sentiment distribution
        if not sentiment_df.empty:
            dist_fig = visualizer.plot_sentiment_distribution(sentiment_df)
            st.plotly_chart(dist_fig, use_container_width=True, key="distribution_chart")
        
        # Sentiment breakdown
        if 'sentiment' in sentiment_df.columns:
            st.subheader("Sentiment Breakdown")
            sentiment_counts = sentiment_df['sentiment'].value_counts()
            total = len(sentiment_df)
            
            for sentiment, count in sentiment_counts.items():
                percentage = (count / total) * 100
                color_class = f"sentiment-{sentiment}"
                st.markdown(
                    f'<div class="{color_class}">{sentiment.title()}: {count} ({percentage:.1f}%)</div>',
                    unsafe_allow_html=True
                )
    
    # Correlations
    if show_correlations and not sentiment_df.empty and not stock_df.empty:
        st.header("🔗 Correlation Analysis")
        
        try:
            daily_sentiment = aggregate_daily_sentiment(sentiment_df)
            correlation_fig = visualizer.plot_correlation_analysis(daily_sentiment, stock_df)
            st.plotly_chart(correlation_fig, use_container_width=True, key="correlation_chart")
            
            # Calculate correlation coefficient
            correlations = calculate_correlation(daily_sentiment, stock_df)
            if correlations:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Sentiment-Price Correlation",
                        value=f"{correlations.get('sentiment_price', 0):.3f}"
                    )
                
                with col2:
                    st.metric(
                        "Sentiment-Volume Correlation",
                        value=f"{correlations.get('sentiment_volume', 0):.3f}"
                    )
                
                with col3:
                    st.metric(
                        "Sentiment-Price Change Correlation",
                        value=f"{correlations.get('sentiment_price_change', 0):.3f}"
                    )
        
        except Exception as e:
            st.error(f"Error in correlation analysis: {e}")
    
    # Source analysis
    if show_source_analysis and not sentiment_df.empty and 'source' in sentiment_df.columns:
        st.header("📰 Source Analysis")
        
        try:
            source_fig = visualizer.plot_sentiment_by_source(sentiment_df)
            st.plotly_chart(source_fig, use_container_width=True, key="source_analysis_chart")
        except Exception as e:
            st.error(f"Error in source analysis: {e}")
    
    # Recent articles
    st.header("📋 Recent Articles")
    
    if not sentiment_df.empty:
        # Sort by date
        if 'date' in sentiment_df.columns:
            display_df = sentiment_df.sort_values('date', ascending=False).head(10)
        else:
            display_df = sentiment_df.head(10)
        
        for idx, row in display_df.iterrows():
            with st.expander(f"{row['title'][:100]}..." if len(row['title']) > 100 else row['title']):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Source:** {row.get('source', 'Unknown')}")
                    st.write(f"**Date:** {row.get('date', 'Unknown')}")
                    st.write(f"**Content:** {row['content'][:300]}...")
                    if 'url' in row and row['url']:
                        st.write(f"**URL:** [Read full article]({row['url']})")
                
                with col2:
                    sentiment = row.get('sentiment', 'neutral')
                    sentiment_score = row.get('sentiment_score', 0)
                    confidence = row.get('confidence', 0)
                    
                    color_class = f"sentiment-{sentiment}"
                    st.markdown(f'<div class="{color_class}">Sentiment: {sentiment.title()}</div>', unsafe_allow_html=True)
                    st.write(f"Score: {sentiment_score:.3f}")
                    st.write(f"Confidence: {confidence:.3f}")
                    st.write(f"Confidence: {confidence:.3f}")
    
    # Data tables (in sidebar)
    st.sidebar.header("📊 Data Tables")
    
    if st.sidebar.checkbox("Show Raw Data"):
        st.sidebar.subheader("News Data")
        st.sidebar.dataframe(sentiment_df.head())
        
        st.sidebar.subheader("Stock Data")
        st.sidebar.dataframe(stock_df.head())
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        "This dashboard analyzes sentiment in NVIDIA-related news and compares it with stock performance. "
        "It uses FinBERT, a specialized financial NLP model, for sentiment analysis."
    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built By Amine Rahmani using Streamlit, FinBERT, and yfinance. "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()
