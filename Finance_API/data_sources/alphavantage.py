"""Alpha Vantage Data Source Module"""
import os
import time
import logging
import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger('stock_agent.data_source.alpha_vantage')

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def get_daily_data(ticker, start_date=None, end_date=None):
    """Get daily price data for a single ticker"""
    if not API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY not found in .env file")
    
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={API_KEY}"
    
    logger.info(f"Fetching data for {ticker}")
    response = requests.get(url)
    
    if response.status_code != 200:
        logger.error(f"Failed to get data for {ticker}: {response.status_code}")
        return None
    
    data = response.json()
    
    # Check for error messages
    if "Error Message" in data:
        logger.error(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
        return None
    
    if "Time Series (Daily)" not in data:
        logger.error(f"No time series data for {ticker}")
        return None
    
    # Convert to dataframe
    time_series = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(time_series, orient='index')
    
    # Rename columns
    df = df.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. adjusted close': 'Adjusted Close',
        '6. volume': 'Volume',
    })
    
    # Convert string values to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    # Set index as datetime
    df.index = pd.to_datetime(df.index)
    
    # Filter by date range if provided
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    # Sort by date (ascending)
    df = df.sort_index()
    
    return df

def download_batch(tickers, start_date=None, end_date=None, rate_limit=12):
    """
    Download data for multiple tickers with rate limiting
    Note: Alpha Vantage free tier requires downloading one ticker at a time
    """
    all_data = {}
    
    for i, ticker in enumerate(tickers):
        try:
            # Rate limiting - 12 requests per minute (5 sec interval to be safe)
            if i > 0 and i % rate_limit == 0:
                logger.info(f"Rate limit reached, sleeping for 65 seconds...")
                time.sleep(65)  # 65 seconds to be safe
            
            data = get_daily_data(ticker, start_date, end_date)
            if data is not None:
                all_data[ticker] = data
                logger.info(f"Downloaded {ticker}: {len(data)} days of data")
            else:
                logger.warning(f"No data retrieved for {ticker}")
                
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {str(e)}")
    
    return all_data 