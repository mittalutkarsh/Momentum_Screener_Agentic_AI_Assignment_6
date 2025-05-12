"""Finnhub Data Source Module"""
import os
import time
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

logger = logging.getLogger('stock_agent.data_source.finnhub')

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")

def get_stock_candles(ticker, start_timestamp, end_timestamp, resolution="D"):
    """Get OHLCV data for a single ticker"""
    if not API_KEY:
        raise ValueError("FINNHUB_API_KEY not found in .env file")
    
    url = f"https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": ticker,
        "resolution": resolution,
        "from": start_timestamp,
        "to": end_timestamp,
        "token": API_KEY
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        logger.error(f"Failed to get data for {ticker}: {response.status_code}")
        return None
    
    data = response.json()
    
    # Check for error status
    if data.get("s") == "no_data":
        logger.warning(f"No data available for {ticker}")
        return None
    
    # Create dataframe
    df = pd.DataFrame({
        "Open": data.get("o", []),
        "High": data.get("h", []),
        "Low": data.get("l", []),
        "Close": data.get("c", []),
        "Volume": data.get("v", []),
    })
    
    # Create timestamp index
    df.index = pd.to_datetime(data.get("t", []), unit="s")
    
    return df

def download_batch(tickers, start_date=None, end_date=None, rate_limit=30):
    """Download data for multiple tickers with rate limiting"""
    all_data = {}
    
    # Convert dates to unix timestamps - handle string or datetime objects
    if isinstance(start_date, str):
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    elif isinstance(start_date, datetime):
        start_timestamp = int(start_date.timestamp())
    else:
        # Default to 1 year ago if no date provided
        start_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
    
    if isinstance(end_date, str):
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    elif isinstance(end_date, datetime):
        end_timestamp = int(end_date.timestamp())
    else:
        # Default to today if no date provided
        end_timestamp = int(datetime.now().timestamp())
    
    logger.info(f"Downloading data for {len(tickers)} tickers from Finnhub")
    logger.info(f"Date range: {datetime.fromtimestamp(start_timestamp)} to {datetime.fromtimestamp(end_timestamp)}")
    
    # Process tickers in small batches to avoid timeouts
    for i, ticker in enumerate(tickers):
        try:
            # Rate limiting - 30 requests per minute (2 sec interval)
            if i > 0 and i % rate_limit == 0:
                logger.info(f"Rate limit reached, sleeping for 62 seconds...")
                time.sleep(62)
            
            logger.info(f"Downloading {ticker} ({i+1}/{len(tickers)})")
            data = get_stock_candles(ticker, start_timestamp, end_timestamp)
            
            if data is not None and not data.empty:
                all_data[ticker] = data
                logger.info(f"✓ Downloaded {ticker}: {len(data)} days of data")
            else:
                logger.warning(f"✗ No data retrieved for {ticker}")
                
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {str(e)}")
    
    if not all_data:
        logger.error("No data retrieved from Finnhub for any ticker")
    else:
        logger.info(f"Successfully downloaded data for {len(all_data)} tickers")
    
    return all_data 