"""
Visualization components for NVIDIA sentiment analysis dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
from config import CHART_COLORS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SentimentVisualizer:
    """Create visualizations for sentiment analysis and stock data"""
    
    def __init__(self):
        self.colors = CHART_COLORS
    
    def plot_sentiment_timeline(self, sentiment_df, stock_df=None):
        """
        Create interactive timeline plot of sentiment vs stock price
        
        Args:
            sentiment_df (pd.DataFrame): Sentiment data with date index
            stock_df (pd.DataFrame): Stock data with date index
            
        Returns:
            plotly.graph_objects.Figure: Interactive plot
        """
        try:
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Sentiment Score Timeline', 'NVIDIA Stock Price'),
                row_heights=[0.4, 0.6]
            )
            
            # Prepare sentiment data
            if 'date' in sentiment_df.columns:
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                sentiment_df = sentiment_df.set_index('date')
            
            # Plot sentiment score
            fig.add_trace(
                go.Scatter(
                    x=sentiment_df.index,
                    y=sentiment_df['sentiment_score_mean'] if 'sentiment_score_mean' in sentiment_df.columns else sentiment_df.get('sentiment_score', []),
                    mode='lines+markers',
                    name='Sentiment Score',
                    line=dict(color=self.colors['positive'], width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            # Add horizontal line at y=0 for sentiment
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
            
            # Plot stock price if available
            if stock_df is not None and not stock_df.empty:
                if 'Date' in stock_df.columns:
                    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
                    stock_df = stock_df.set_index('Date')
                
                fig.add_trace(
                    go.Scatter(
                        x=stock_df.index,
                        y=stock_df['Close'],
                        mode='lines',
                        name='Stock Price',
                        line=dict(color=self.colors['stock'], width=2)
                    ),
                    row=2, col=1
                )
                
                # Add volume as bar chart
                fig.add_trace(
                    go.Bar(
                        x=stock_df.index,
                        y=stock_df['Volume'],
                        name='Volume',
                        yaxis='y3',
                        opacity=0.3,
                        marker_color='lightblue'
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'NVIDIA Sentiment Analysis vs Stock Performance',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                height=700,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Update y-axes
            fig.update_yaxes(title_text="Sentiment Score", row=1, col=1)
            fig.update_yaxes(title_text="Stock Price ($)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sentiment timeline: {e}")
            return go.Figure()
    
    def plot_sentiment_distribution(self, df):
        """
        Create sentiment distribution plots
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment data
            
        Returns:
            plotly.graph_objects.Figure: Distribution plot
        """
        try:
            if 'sentiment' not in df.columns:
                return go.Figure()
            
            # Count sentiments
            sentiment_counts = df['sentiment'].value_counts()
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Sentiment Distribution', 'Sentiment Score Distribution'),
                specs=[[{"type": "pie"}, {"type": "histogram"}]]
            )
            
            # Pie chart
            fig.add_trace(
                go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    marker_colors=[self.colors[label] for label in sentiment_counts.index],
                    textinfo='label+percent',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Histogram of sentiment scores
            fig.add_trace(
                go.Histogram(
                    x=df['sentiment_score'] if 'sentiment_score' in df.columns else [],
                    nbinsx=30,
                    name='Sentiment Score',
                    marker_color=self.colors['neutral'],
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title={
                    'text': 'Sentiment Analysis Distribution',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                height=400,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sentiment distribution: {e}")
            return go.Figure()
    
    def create_word_cloud(self, df, text_column='content'):
        """
        Create word cloud from text data
        
        Args:
            df (pd.DataFrame): DataFrame with text data
            text_column (str): Column containing text
            
        Returns:
            matplotlib.figure.Figure: Word cloud figure
        """
        try:
            if df.empty or text_column not in df.columns:
                return plt.figure()
            
            # Combine all text
            all_text = ' '.join(df[text_column].fillna('').astype(str))
            
            if not all_text.strip():
                return plt.figure()
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100,
                relative_scaling=0.5,
                random_state=42
            ).generate(all_text)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Most Frequent Words in News Articles', fontsize=16, pad=20)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating word cloud: {e}")
            return plt.figure()
    
    def plot_correlation_analysis(self, sentiment_df, stock_df):
        """
        Create correlation analysis visualization
        
        Args:
            sentiment_df (pd.DataFrame): Sentiment data
            stock_df (pd.DataFrame): Stock data
            
        Returns:
            plotly.graph_objects.Figure: Correlation plot
        """
        try:
            # Merge data
            if 'date' in sentiment_df.columns:
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], utc=True)
                sentiment_df = sentiment_df.set_index('date')
            
            if 'Date' in stock_df.columns:
                stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)
                stock_df = stock_df.set_index('Date')
            
            # Remove timezone info for merging
            if hasattr(sentiment_df.index, 'tz') and sentiment_df.index.tz is not None:
                sentiment_df.index = sentiment_df.index.tz_localize(None)
            if hasattr(stock_df.index, 'tz') and stock_df.index.tz is not None:
                stock_df.index = stock_df.index.tz_localize(None)
            
            # Merge on date
            merged_df = pd.merge(sentiment_df, stock_df, left_index=True, right_index=True, how='inner')
            
            if merged_df.empty:
                return go.Figure()
            
            # Calculate correlation
            sentiment_col = 'sentiment_score_mean' if 'sentiment_score_mean' in merged_df.columns else 'sentiment_score'
            if sentiment_col not in merged_df.columns:
                return go.Figure()
            
            correlation = merged_df[sentiment_col].corr(merged_df['Close'])
            
            # Create scatter plot
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=merged_df[sentiment_col],
                    y=merged_df['Close'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=merged_df['Volume'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Volume")
                    ),
                    text=merged_df.index.strftime('%Y-%m-%d'),
                    hovertemplate='<b>Date:</b> %{text}<br><b>Sentiment:</b> %{x:.3f}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
                )
            )
            
            # Add trend line
            z = np.polyfit(merged_df[sentiment_col], merged_df['Close'], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=merged_df[sentiment_col],
                    y=p(merged_df[sentiment_col]),
                    mode='lines',
                    name=f'Trend (r={correlation:.3f})',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig.update_layout(
                title=f'Sentiment vs Stock Price Correlation (r={correlation:.3f})',
                xaxis_title='Average Daily Sentiment Score',
                yaxis_title='Stock Price ($)',
                height=500
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation analysis: {e}")
            return go.Figure()
    
    def plot_daily_sentiment_heatmap(self, df):
        """
        Create heatmap of daily sentiment patterns
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment data
            
        Returns:
            plotly.graph_objects.Figure: Heatmap
        """
        try:
            if df.empty or 'date' not in df.columns:
                return go.Figure()
            
            # Prepare data
            df['date'] = pd.to_datetime(df['date'])
            df['weekday'] = df['date'].dt.day_name()
            df['week'] = df['date'].dt.isocalendar().week
            
            # Group by week and weekday
            sentiment_col = 'sentiment_score_mean' if 'sentiment_score_mean' in df.columns else 'sentiment_score'
            if sentiment_col not in df.columns:
                return go.Figure()
            
            heatmap_data = df.groupby(['week', 'weekday'])[sentiment_col].mean().unstack()
            
            # Reorder weekdays
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(columns=weekday_order)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn',
                zmid=0,
                hovertemplate='<b>Week:</b> %{y}<br><b>Day:</b> %{x}<br><b>Avg Sentiment:</b> %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Weekly Sentiment Heatmap',
                xaxis_title='Day of Week',
                yaxis_title='Week Number',
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sentiment heatmap: {e}")
            return go.Figure()
    
    def plot_sentiment_by_source(self, df):
        """
        Create visualization of sentiment by news source
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment and source data
            
        Returns:
            plotly.graph_objects.Figure: Source sentiment plot
        """
        try:
            if df.empty or 'source' not in df.columns or 'sentiment' not in df.columns:
                return go.Figure()
            
            # Group by source and sentiment
            source_sentiment = df.groupby(['source', 'sentiment']).size().unstack(fill_value=0)
            
            # Calculate percentages
            source_sentiment_pct = source_sentiment.div(source_sentiment.sum(axis=1), axis=0) * 100
            
            fig = go.Figure()
            
            # Add bars for each sentiment
            for sentiment in ['positive', 'neutral', 'negative']:
                if sentiment in source_sentiment_pct.columns:
                    fig.add_trace(
                        go.Bar(
                            name=sentiment.title(),
                            x=source_sentiment_pct.index,
                            y=source_sentiment_pct[sentiment],
                            marker_color=self.colors[sentiment]
                        )
                    )
            
            fig.update_layout(
                title='Sentiment Distribution by News Source',
                xaxis_title='News Source',
                yaxis_title='Percentage (%)',
                barmode='stack',
                height=500,
                xaxis_tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating source sentiment plot: {e}")
            return go.Figure()
    
    def create_summary_metrics(self, sentiment_df, stock_df):
        """
        Create summary metrics for dashboard
        
        Args:
            sentiment_df (pd.DataFrame): Sentiment data
            stock_df (pd.DataFrame): Stock data
            
        Returns:
            dict: Summary metrics
        """
        try:
            metrics = {}
            
            # Sentiment metrics
            if not sentiment_df.empty:
                sentiment_col = 'sentiment_score_mean' if 'sentiment_score_mean' in sentiment_df.columns else 'sentiment_score'
                if sentiment_col in sentiment_df.columns:
                    metrics['avg_sentiment'] = sentiment_df[sentiment_col].mean()
                    metrics['sentiment_trend'] = 'positive' if metrics['avg_sentiment'] > 0 else 'negative' if metrics['avg_sentiment'] < 0 else 'neutral'
                
                if 'sentiment' in sentiment_df.columns:
                    sentiment_counts = sentiment_df['sentiment'].value_counts()
                    total_articles = len(sentiment_df)
                    metrics['positive_pct'] = (sentiment_counts.get('positive', 0) / total_articles) * 100
                    metrics['negative_pct'] = (sentiment_counts.get('negative', 0) / total_articles) * 100
                    metrics['neutral_pct'] = (sentiment_counts.get('neutral', 0) / total_articles) * 100
                
                metrics['total_articles'] = len(sentiment_df)
            
            # Stock metrics
            if not stock_df.empty and 'Close' in stock_df.columns:
                latest_price = stock_df['Close'].iloc[-1]
                price_change = stock_df['Close'].iloc[-1] - stock_df['Close'].iloc[0]
                price_change_pct = (price_change / stock_df['Close'].iloc[0]) * 100
                
                metrics['latest_price'] = latest_price
                metrics['price_change'] = price_change
                metrics['price_change_pct'] = price_change_pct
                metrics['price_trend'] = 'up' if price_change > 0 else 'down'
                
                if 'Volume' in stock_df.columns:
                    metrics['avg_volume'] = stock_df['Volume'].mean()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error creating summary metrics: {e}")
            return {}

if __name__ == "__main__":
    # Test visualizations with sample data
    visualizer = SentimentVisualizer()
    
    # Create sample data for testing
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    sample_sentiment = pd.DataFrame({
        'date': dates,
        'sentiment_score': np.random.normal(0.1, 0.3, len(dates)),
        'sentiment': np.random.choice(['positive', 'neutral', 'negative'], len(dates)),
        'source': np.random.choice(['Yahoo Finance', 'Reuters', 'Bloomberg'], len(dates))
    })
    
    sample_stock = pd.DataFrame({
        'Date': dates,
        'Close': 400 + np.random.normal(0, 20, len(dates)).cumsum(),
        'Volume': np.random.normal(50000000, 10000000, len(dates))
    })
    
    print("Testing visualizations...")
    
    # Test timeline plot
    fig1 = visualizer.plot_sentiment_timeline(sample_sentiment, sample_stock)
    print("Timeline plot created")
    
    # Test distribution plot
    fig2 = visualizer.plot_sentiment_distribution(sample_sentiment)
    print("Distribution plot created")
    
    # Test correlation plot
    fig3 = visualizer.plot_correlation_analysis(sample_sentiment, sample_stock)
    print("Correlation plot created")
    
    print("All visualizations tested successfully!")
