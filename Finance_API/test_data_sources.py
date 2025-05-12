#!/usr/bin/env python3
"""
test_data_sources.py - Test data sources for momentum screener
"""
import os
import logging
from datetime import datetime, timedelta
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_data_sources')

# Import data sources
from data_sources.alphavantage import download_batch as av_download_batch
from data_sources.finnhub import download_batch as finnhub_download_batch
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

def test_finnhub():
    """Test Finnhub data source"""
    logger.info("=== Testing Finnhub Data Source ===")
    
    # Sample tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # Date range (last 30 days)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    
    try:
        logger.info(f"Downloading data for {len(tickers)} tickers...")
        data = finnhub_download_batch(tickers, start_date, end_date)
        
        if data and len(data) > 0:
            logger.info(f"Successfully downloaded data for {len(data)} tickers")
            for ticker, df in data.items():
                logger.info(f"{ticker}: {len(df)} days of data")
                if not df.empty:
                    logger.info(f"Sample data for {ticker}:\n{df.head(3)}")
            return True
        else:
            logger.error("No data returned from Finnhub")
            return False
    except Exception as e:
        logger.error(f"Error testing Finnhub: {str(e)}")
        return False

def test_alpha_vantage():
    """Test Alpha Vantage data source"""
    logger.info("=== Testing Alpha Vantage Data Source ===")
    
    # Sample tickers
    tickers = ["AAPL", "MSFT"]  # Just two to avoid rate limits
    
    # Date range (last 30 days)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    try:
        logger.info(f"Downloading data for {len(tickers)} tickers...")
        data = av_download_batch(tickers, start_date, end_date)
        
        if data and len(data) > 0:
            logger.info(f"Successfully downloaded data for {len(data)} tickers")
            for ticker, df in data.items():
                logger.info(f"{ticker}: {len(df)} days of data")
                if not df.empty:
                    logger.info(f"Sample data for {ticker}:\n{df.head(3)}")
            return True
        else:
            logger.error("No data returned from Alpha Vantage")
            return False
    except Exception as e:
        logger.error(f"Error testing Alpha Vantage: {str(e)}")
        return False

def test_yfinance():
    """Test Yahoo Finance data source"""
    if not YFINANCE_AVAILABLE:
        logger.error("yfinance not installed")
        return False
    
    logger.info("=== Testing Yahoo Finance Data Source ===")
    
    # Sample tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    
    # Date range (last 30 days)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=30)
    
    try:
        logger.info(f"Downloading data for {len(tickers)} tickers...")
        data = yf.download(tickers, start=start_date, end=end_date)
        
        if not data.empty:
            logger.info(f"Successfully downloaded data with shape {data.shape}")
            logger.info(f"Sample data:\n{data.head(3)}")
            return True
        else:
            logger.error("No data returned from Yahoo Finance")
            return False
    except Exception as e:
        logger.error(f"Error testing Yahoo Finance: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Testing data sources...")
    
    # Test each data source
    finnhub_result = test_finnhub()
    av_result = test_alpha_vantage()
    yf_result = test_yfinance() if YFINANCE_AVAILABLE else False
    
    # Print summary
    logger.info("\n=== Test Results ===")
    logger.info(f"Finnhub:       {'✓ Success' if finnhub_result else '✗ Failed'}")
    logger.info(f"Alpha Vantage: {'✓ Success' if av_result else '✗ Failed'}")
    logger.info(f"Yahoo Finance: {'✓ Success' if yf_result else '✗ Failed'}") 