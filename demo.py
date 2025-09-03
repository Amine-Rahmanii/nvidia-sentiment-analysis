"""
Quick demo of NVIDIA sentiment analysis project
Run this script to see the project in action with sample data
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collector import generate_sample_data
from stock_data import generate_sample_stock_data
from sentiment_analyzer import FinBERTAnalyzer
from visualizer import SentimentVisualizer
from utils import aggregate_daily_sentiment, calculate_correlation

def main():
    print("🚀 NVIDIA Sentiment Analysis - Quick Demo")
    print("=" * 50)
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    news_df = generate_sample_data()
    stock_df = generate_sample_stock_data()
    
    print(f"✅ Generated {len(news_df)} news articles")
    print(f"✅ Generated {len(stock_df)} stock data points")
    
    # Display sample news
    print("\n📰 Sample news articles:")
    for i, row in news_df.head(3).iterrows():
        print(f"\n{i+1}. {row['title']}")
        print(f"   Source: {row['source']}")
        print(f"   Date: {row['date'].strftime('%Y-%m-%d')}")
        print(f"   Content: {row['content'][:100]}...")
    
    # Analyze sentiment
    print("\n🧠 Analyzing sentiment with FinBERT...")
    analyzer = FinBERTAnalyzer()
    sentiment_df = analyzer.analyze_dataframe(news_df)
    
    print("✅ Sentiment analysis complete!")
    
    # Display sentiment results
    print("\n📈 Sentiment Summary:")
    summary = analyzer.get_sentiment_summary(sentiment_df)
    
    if summary:
        print(f"   Total articles: {summary.get('total_articles', 0)}")
        print(f"   Positive: {summary.get('positive_count', 0)} ({summary.get('positive_pct', 0):.1f}%)")
        print(f"   Neutral: {summary.get('neutral_count', 0)} ({summary.get('neutral_pct', 0):.1f}%)")
        print(f"   Negative: {summary.get('negative_count', 0)} ({summary.get('negative_pct', 0):.1f}%)")
        print(f"   Average sentiment score: {summary.get('avg_sentiment_score', 0):.3f}")
    
    # Show sample sentiment results
    print("\n🎯 Sample sentiment analysis results:")
    for i, row in sentiment_df.head(3).iterrows():
        sentiment = row.get('sentiment', 'unknown')
        score = row.get('sentiment_score', 0)
        confidence = row.get('confidence', 0)
        
        emoji = "😊" if sentiment == 'positive' else "😐" if sentiment == 'neutral' else "😔"
        print(f"\n{i+1}. {row['title'][:60]}...")
        print(f"   {emoji} Sentiment: {sentiment.title()} (Score: {score:.3f}, Confidence: {confidence:.3f})")
    
    # Stock data summary
    print("\n📊 Stock Data Summary:")
    latest_price = stock_df['Close'].iloc[-1]
    first_price = stock_df['Close'].iloc[0]
    price_change = latest_price - first_price
    price_change_pct = (price_change / first_price) * 100
    
    print(f"   Price range: ${stock_df['Close'].min():.2f} - ${stock_df['Close'].max():.2f}")
    print(f"   Latest price: ${latest_price:.2f}")
    print(f"   Period change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
    print(f"   Average volume: {stock_df['Volume'].mean():,.0f}")
    
    # Create visualizations
    print("\n📈 Creating visualizations...")
    visualizer = SentimentVisualizer()
    
    # Aggregate daily sentiment
    daily_sentiment = aggregate_daily_sentiment(sentiment_df)
    
    # Calculate correlations
    correlations = calculate_correlation(daily_sentiment, stock_df.set_index('Date'))
    
    print("✅ Visualizations created!")
    
    # Display correlation results
    if correlations:
        print("\n🔗 Correlation Analysis:")
        for metric, correlation in correlations.items():
            direction = "positive" if correlation > 0.1 else "negative" if correlation < -0.1 else "weak"
            print(f"   {metric.replace('_', ' ').title()}: {correlation:.3f} ({direction})")
    
    # Create and save a simple plot
    try:
        print("\n💾 Creating sample plot...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot sentiment over time
        daily_sentiment.reset_index(inplace=True)
        ax1.plot(daily_sentiment['date'], daily_sentiment['sentiment_score_mean'], 
                marker='o', color='green', linewidth=2)
        ax1.set_title('Daily Average Sentiment Score')
        ax1.set_ylabel('Sentiment Score')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Plot stock price
        ax2.plot(stock_df['Date'], stock_df['Close'], 
                color='blue', linewidth=2)
        ax2.set_title('NVIDIA Stock Price')
        ax2.set_ylabel('Price ($)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('nvidia_sentiment_demo.png', dpi=150, bbox_inches='tight')
        print("✅ Plot saved as 'nvidia_sentiment_demo.png'")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not create plot: {e}")
    
    print("\n🎉 Demo completed successfully!")
    print("\n📋 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run full dashboard: streamlit run app.py")
    print("3. Run tests: python test_project.py")
    print("\n💡 Tips:")
    print("- The dashboard includes interactive plots and real-time data collection")
    print("- You can customize the analysis period and data sources in config.py")
    print("- The project supports both real data collection and sample data for testing")

if __name__ == "__main__":
    main()
