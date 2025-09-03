"""
Stock data retrieval and processing for NVIDIA
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from config import STOCK_SYMBOL, START_DATE, END_DATE, PROCESSED_DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataCollector:
    """Collect and process NVIDIA stock data"""
    
    def __init__(self, symbol=STOCK_SYMBOL):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
    
    def get_historical_data(self, start_date=None, end_date=None, period="6mo"):
        """
        Get historical stock data
        
        Args:
            start_date (datetime): Start date for data collection
            end_date (datetime): End date for data collection
            period (str): Period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            pd.DataFrame: Historical stock data
        """
        try:
            if start_date and end_date:
                data = self.ticker.history(start=start_date, end=end_date)
            else:
                data = self.ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data retrieved for {self.symbol}")
                return pd.DataFrame()
            
            # Reset index to make Date a column
            data.reset_index(inplace=True)
            
            # Ensure Date column is datetime
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Calculate additional metrics
            data = self._calculate_technical_indicators(data)
            
            logger.info(f"Retrieved {len(data)} days of stock data for {self.symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving stock data: {e}")
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, df):
        """
        Calculate technical indicators
        
        Args:
            df (pd.DataFrame): Stock data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        try:
            # Price change and percentage change
            df['Price_Change'] = df['Close'].diff()
            df['Price_Change_Pct'] = df['Close'].pct_change() * 100
            
            # Moving averages
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_10'] = df['Close'].rolling(window=10).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            
            # Volatility (rolling standard deviation)
            df['Volatility_5'] = df['Close'].rolling(window=5).std()
            df['Volatility_20'] = df['Close'].rolling(window=20).std()
            
            # Volume moving average
            df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
            
            # High-Low spread
            df['HL_Spread'] = df['High'] - df['Low']
            df['HL_Spread_Pct'] = (df['HL_Spread'] / df['Close']) * 100
            
            # Trading range (True Range)
            df['True_Range'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                )
            )
            
            # Average True Range (ATR)
            df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
            
            # Relative Strength Index (RSI) approximation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['MA_20']
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def get_company_info(self):
        """
        Get company information
        
        Returns:
            dict: Company information
        """
        try:
            info = self.ticker.info
            
            # Extract relevant information
            company_info = {
                'name': info.get('longName', 'NVIDIA Corporation'),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', ''),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'NASDAQ')
            }
            
            return company_info
            
        except Exception as e:
            logger.error(f"Error retrieving company info: {e}")
            return {}
    
    def get_financial_metrics(self):
        """
        Get key financial metrics
        
        Returns:
            dict: Financial metrics
        """
        try:
            info = self.ticker.info
            
            metrics = {
                'pe_ratio': info.get('forwardPE', 0),
                'price_to_book': info.get('priceToBook', 0),
                'price_to_sales': info.get('priceToSalesTrailing12Months', 0),
                'profit_margin': info.get('profitMargins', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'return_on_equity': info.get('returnOnEquity', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'beta': info.get('beta', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving financial metrics: {e}")
            return {}
    
    def save_data(self, df, filename=None):
        """
        Save stock data to file
        
        Args:
            df (pd.DataFrame): DataFrame to save
            filename (str): Optional filename
            
        Returns:
            str: Filepath where data was saved
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.symbol}_stock_data_{timestamp}.csv"
        
        filepath = os.path.join(PROCESSED_DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Stock data saved to {filepath}")
        
        return filepath
    
    def get_recent_performance(self, days=30):
        """
        Get recent performance summary
        
        Args:
            days (int): Number of recent days to analyze
            
        Returns:
            dict: Performance metrics
        """
        try:
            # Get recent data
            recent_data = self.get_historical_data(period=f"{days}d")
            
            if recent_data.empty:
                return {}
            
            # Calculate performance metrics
            latest_price = recent_data['Close'].iloc[-1]
            start_price = recent_data['Close'].iloc[0]
            price_change = latest_price - start_price
            price_change_pct = (price_change / start_price) * 100
            
            # Volume analysis
            avg_volume = recent_data['Volume'].mean()
            latest_volume = recent_data['Volume'].iloc[-1]
            volume_ratio = latest_volume / avg_volume
            
            # Volatility
            volatility = recent_data['Close'].std()
            
            performance = {
                'period_days': days,
                'start_price': round(start_price, 2),
                'latest_price': round(latest_price, 2),
                'price_change': round(price_change, 2),
                'price_change_pct': round(price_change_pct, 2),
                'avg_volume': int(avg_volume),
                'latest_volume': int(latest_volume),
                'volume_ratio': round(volume_ratio, 2),
                'volatility': round(volatility, 2),
                'high': round(recent_data['High'].max(), 2),
                'low': round(recent_data['Low'].min(), 2)
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating recent performance: {e}")
            return {}

def generate_sample_stock_data():
    """
    Generate sample stock data for testing
    
    Returns:
        pd.DataFrame: Sample stock data
    """
    # Create date range
    dates = pd.date_range(end=datetime.now(), periods=180, freq='D')
    
    # Generate realistic stock data with trend and volatility
    np.random.seed(42)
    
    # Base price around NVIDIA's typical range
    base_price = 400
    trend = np.linspace(0, 200, len(dates))  # Upward trend
    noise = np.random.normal(0, 20, len(dates))  # Daily volatility
    
    # Generate price series
    close_prices = base_price + trend + noise.cumsum() * 0.1
    close_prices = np.maximum(close_prices, 50)  # Ensure positive prices
    
    # Generate other OHLC data
    data = []
    for i, date in enumerate(dates):
        close = close_prices[i]
        open_price = close + np.random.normal(0, 5)
        high = max(open_price, close) + abs(np.random.normal(0, 3))
        low = min(open_price, close) - abs(np.random.normal(0, 3))
        volume = int(np.random.normal(50000000, 20000000))  # Typical NVIDIA volume
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': max(volume, 1000000)  # Ensure positive volume
        })
    
    df = pd.DataFrame(data)
    
    # Calculate technical indicators
    collector = StockDataCollector()
    df = collector._calculate_technical_indicators(df)
    
    return df

if __name__ == "__main__":
    collector = StockDataCollector()
    
    # Try to collect real data, fall back to sample data if needed
    try:
        df = collector.get_historical_data(period="6mo")
        if df.empty:
            raise Exception("No stock data retrieved")
            
        # Get additional info
        company_info = collector.get_company_info()
        financial_metrics = collector.get_financial_metrics()
        performance = collector.get_recent_performance()
        
        print(f"Company: {company_info.get('name', 'NVIDIA Corporation')}")
        print(f"Recent performance: {performance.get('price_change_pct', 0):.2f}%")
        
    except Exception as e:
        logger.warning(f"Using sample stock data due to error: {e}")
        df = generate_sample_stock_data()
    
    # Save the data
    filepath = collector.save_data(df)
    print(f"Stock data collection complete. Saved to: {filepath}")
