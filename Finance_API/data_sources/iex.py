"""IEX Cloud Data Source Module"""
import os
import time
import logging
import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger('stock_agent.data_source.iex')

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("IEX_API_KEY")
BASE_URL = "https://cloud.iexapis.com/stable"

def get_historical_data(ticker, range_str="1y"):
    """Get historical price data for a single ticker"""
    if not API_KEY:
        raise ValueError("IEX_API_KEY not found in .env file")
    
    url = f"{BASE_URL}/stock/{ticker}/chart/{range_str}"
    params = {"token": API_KEY}
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        logger.error(f"Failed to get data for {ticker}: {response.status_code}")
        return None
    
    data = response.json()
    
    if not data:
        logger.warning(f"No data available for {ticker}")
        return None
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Set date as index
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    
    # Rename columns to match yfinance format
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    
    return df

def download_batch(tickers, start_date=None, end_date=None, rate_limit=50):
    """Download data for multiple tickers with rate limiting"""
    all_data = {}
    
    # Determine range based on start date
    range_str = "1y"  # Default to 1 year
    
    for i, ticker in enumerate(tickers):
        try:
            # Rate limiting - 50 requests per second for sandbox
            if i > 0 and i % rate_limit == 0:
                logger.info(f"Rate limit reached, sleeping for 2 seconds...")
                time.sleep(2)
            
            data = get_historical_data(ticker, range_str)
            if data is not None:
                # Filter by date if needed
                if start_date:
                    data = data[data.index >= pd.to_datetime(start_date)]
                if end_date:
                    data = data[data.index <= pd.to_datetime(end_date)]
                
                all_data[ticker] = data
                logger.info(f"Downloaded {ticker}: {len(data)} days of data")
            else:
                logger.warning(f"No data retrieved for {ticker}")
                
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {str(e)}")
    
    return all_data 