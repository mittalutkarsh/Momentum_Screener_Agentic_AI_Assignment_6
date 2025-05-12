#!/usr/bin/env python3
"""
momentum_screener_llm.py - Enhanced Momentum Screener with LLM Integration

Features:
  - Robust stock data downloading with batch processing and error handling
  - Multiple universe options (S&P 500, S&P 1500, Russell indexes, TSX, custom)
  - LLM-enhanced reasoning and reporting for momentum screening
  - Breakout detection based on proximity to highs and volume surge
  - Excel export with detailed analysis

Dependencies:
  pandas, yfinance, requests, bs4, numpy, google-generativeai, python-dotenv
Install:
  pip install pandas yfinance requests beautifulsoup4 numpy google-generativeai python-dotenv
"""
import os
import json
import asyncio
import time
import pickle
import random
import logging
import traceback
from datetime import datetime, timedelta
from functools import lru_cache
import re

import pandas as pd
import yfinance as yf
import requests
import bs4 as bs
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from data_sources.alphavantage import download_batch as av_download_batch
from data_sources.finnhub import download_batch as finnhub_download_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('momentum_screener')

# ---------- LLM CONFIGURATION ----------
def load_llm_config():
    """Load environment variables and configure Gemini API"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    logger.info(f"API Key loaded: {'Yes' if api_key else 'No'}")
    if api_key:
        logger.info(f"API Key starts with: {api_key[:5]}...")
    
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in .env file. Please add it.")
    
    # Configure the API
    genai.configure(api_key=api_key)
    
    # Create and return the model directly
    try:
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        logger.info("Gemini model configured successfully")
        return model
    except Exception as e:
        logger.error(f"Error configuring Gemini model: {str(e)}")
        raise

# System prompt for the LLM reasoning framework
SYSTEM_PROMPT = """You are a Momentum Screener agent with deep expertise in technical analysis and stock screening. Follow this framework:

[VALIDATE] Check parameters: universe_choice, soft_breakout_pct(0.001-0.02), proximity_threshold(0.01-0.1), volume_threshold(1.0-3.0), lookback_days(90-365).
[HYPOTHESIS] Explain your reasoning about what kind of stocks this screening should identify.
[CALCULATION] Explain key calculations (rolling high, proximity, volume ratio) in plain language.
[ANALYZE] Interpret the results - what do the found breakouts indicate for trading?
[VERIFY] Sanity-check outputs and highlight any anomalies.
[SUGGEST] Recommend 1-3 additional screening parameters that might improve results.

IMPORTANT: Your response must be ONLY valid JSON with NO markdown formatting or code backticks.
DO NOT use ```json or ``` markers around your response. Just return the raw JSON object.

OUTPUT FORMAT: Return a JSON object with these keys:
  "reasoning": {
    "parameter_validation": "string",
    "hypothesis": "string",
    "calculation_explanation": "string"
  },
  "analysis": {
    "market_context": "string",
    "top_breakouts_analysis": "string",
    "data_quality_assessment": "string"
  },
  "suggestions": {
    "parameter_adjustments": "string",
    "additional_filters": "string"
  }
}
"""

# ---------- HELPER FUNCTIONS ----------
@lru_cache(maxsize=5)
def save_sp500_tickers():
    """Scrape S&P 500 tickers from Wikipedia and return the list"""
    try:
        logger.info("Scraping S&P 500 tickers from Wikipedia")
        resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        soup = bs.BeautifulSoup(resp.text, 'lxml')
        
        # Try to find the table - more verbose to help debug
        tables = soup.find_all('table', {'class': 'wikitable'})
        logger.info(f"Found {len(tables)} tables with class 'wikitable'")
        
        if not tables:
            raise ValueError("No tables with class 'wikitable' found")
        
        # Use the first table with class 'wikitable'
        table = tables[0]
        
        tickers = []
        # Find all rows except the header
        rows = table.find_all('tr')[1:]  # Skip header row
        logger.info(f"Found {len(rows)} rows in table")
        
        for row in rows:
            # Get all cells in the row
            cells = row.find_all('td')
            if cells:  # Make sure there are cells in the row
                # The ticker symbol should be in the first column
                ticker = cells[0].text.strip()
                if ticker:  # Only add non-empty tickers
                    tickers.append(ticker)
        
        if not tickers:
            raise ValueError("No tickers found in the table")
        
        logger.info(f"Successfully scraped {len(tickers)} tickers")
        
        # Save to pickle for future use
        with open("sp500tickers.pickle", "wb") as f:
            pickle.dump(tickers, f)
        
        return tickers
    except Exception as e:
        logger.error(f"Error scraping S&P 500 tickers: {e}")
        
        # Try alternative method: direct list of most S&P 500 stocks
        try:
            # This is a more comprehensive fallback list
            major_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "PG", 
                            "UNH", "XOM", "JNJ", "WMT", "MA", "LLY", "CVX", "HD", "AVGO", "MRK", 
                            "PEP", "COST", "ABBV", "KO", "BAC", "PFE", "TMO", "CSCO", "MCD", "CRM", 
                            "ABT", "DHR", "ACN", "ADBE", "WFC", "DIS", "AMD", "CMCSA", "TXN", "NEE",
                            "VZ", "PM", "INTC", "NFLX", "RTX", "QCOM", "IBM", "ORCL", "HON", "BMY"]
            
            logger.warning(f"Using extended fallback list of {len(major_tickers)} major tickers")
            return major_tickers
        except Exception:
            # Last resort - use the original small set
            logger.warning("Using small fallback set of 10 major tickers")
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "PG"]

def clean_tickers(tickers):
    """Clean and normalize ticker symbols"""
    cleaned_tickers = []
    for ticker in tickers:
        # Remove newlines and whitespace
        ticker = ticker.strip()
        
        # Handle special cases
        if ticker == 'BF.B':
            ticker = 'BF-B'
        elif '.' in ticker:
            ticker = ticker.replace('.', '-')
            
        cleaned_tickers.append(ticker)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(cleaned_tickers))

def download_yf_batch(tickers, start_date, end_date, batch_size=20):
    """
    Download data in batches from Yahoo Finance using browser fingerprinting to avoid rate limits
    This combines batching efficiency with browser impersonation for better performance
    """
    try:
        # Try to import curl_cffi
        from curl_cffi import requests as curl_requests
        logger.info("Using curl_cffi with Chrome impersonation to avoid Yahoo Finance rate limits")
    except ImportError:
        logger.error("curl_cffi not installed, trying to install it now...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "curl-cffi"])
            from curl_cffi import requests as curl_requests
            logger.info("Successfully installed curl_cffi")
        except Exception as e:
            logger.error(f"Failed to install curl_cffi: {str(e)}")
            logger.warning("Falling back to standard requests, expect rate limits")
            import requests as curl_requests
            # Will likely fail but we'll try anyway
    
    all_data = {}
    num_batches = len(tickers) // batch_size + (1 if len(tickers) % batch_size != 0 else 0)
    
    logger.info(f"Downloading data for {len(tickers)} tickers in {num_batches} batches of {batch_size}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create a Chrome-like session to bypass fingerprinting
    try:
        session = curl_requests.Session(impersonate="chrome")
        logger.info("Created Chrome impersonation session")
    except Exception as se:
        logger.error(f"Failed to create impersonation session: {str(se)}")
        session = None
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(tickers))
        batch_tickers = tickers[batch_start:batch_end]
        
        logger.info(f"Processing batch {i+1}/{num_batches}: {len(batch_tickers)} tickers")
        
        # Add a delay between batches (longer as we progress)
        if i > 0:
            delay = min(30, 5 + i * 2)  # Increasing delay, max 30 seconds
            logger.info(f"Waiting {delay} seconds before next batch...")
            time.sleep(delay)
        
        try:
            if session:
                # Process each ticker in the batch with our custom session
                batch_data = {}
                for ticker in batch_tickers:
                    try:
                        logger.info(f"Downloading {ticker} with browser impersonation")
                        ticker_obj = yf.Ticker(ticker, session=session)
                        ticker_data = ticker_obj.history(
                            start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'),
                            auto_adjust=True
                        )
                        
                        if not ticker_data.empty:
                            batch_data[ticker] = ticker_data
                            logger.info(f"Successfully downloaded {ticker} data: {len(ticker_data)} days")
                        else:
                            logger.warning(f"Empty data for {ticker}")
                        
                        # Small delay between tickers within batch
                        time.sleep(1)
                    except Exception as e:
                        logger.error(f"Error downloading {ticker}: {str(e)}")
                
                # Add successful downloads to all_data
                all_data.update(batch_data)
                logger.info(f"Batch {i+1} complete. Downloaded {len(batch_data)}/{len(batch_tickers)} tickers successfully")
                
            else:
                # Fallback to standard yfinance batch download if no session
                logger.info("Using standard yfinance download (no browser impersonation)")
                ticker_str = " ".join(batch_tickers)
                batch_data = yf.download(
                    ticker_str,
                    start=start_date,
                    end=end_date,
                    group_by='ticker',
                    auto_adjust=True,
                    threads=False,
                    progress=False
                )
                
                # Process the data
                if len(batch_tickers) == 1:
                    ticker = batch_tickers[0]
                    all_data[ticker] = batch_data
                else:
                    # For multiple tickers, data has multi-level columns
                    for ticker in batch_tickers:
                        try:
                            if ticker in batch_data.columns.levels[0]:
                                ticker_data = batch_data[ticker].copy()
                                if not ticker_data.empty:
                                    all_data[ticker] = ticker_data
                                    logger.info(f"Successfully downloaded {ticker} data: {len(ticker_data)} days")
                                else:
                                    logger.warning(f"Empty data for {ticker}")
                        except Exception as e:
                            logger.error(f"Error processing {ticker} data: {str(e)}")
                
                logger.info(f"Batch {i+1} complete. Downloaded data for {len(all_data) - len(all_data.keys() - set(batch_tickers))} tickers in this batch")
                        
        except Exception as e:
            logger.error(f"Error processing batch {i+1}: {str(e)}")
    
    # Log results
    if not all_data:
        logger.error("No data retrieved from Yahoo Finance for any ticker")
    else:
        logger.info(f"Successfully downloaded data for {len(all_data)}/{len(tickers)} tickers")
    
    return all_data

def get_stock_universe(universe_choice):
    """Load stock universe based on user choice"""
    if universe_choice == 0:
        universe_name = "SPY"
        tickers = save_sp500_tickers()
        universe = clean_tickers(tickers)
    elif universe_choice == 1:
        universe_name = "S&P1500"
        try:
            # Try to use our own scraper first for the S&P 500
            sp500_tickers = save_sp500_tickers()
            
            # Use pd.read_html for S&P 400 and 600
            df2 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]
            df3 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]
            
            midcap_tickers = df2['Symbol'].tolist()
            smallcap_tickers = df3['Symbol'].tolist()
            
            universe = clean_tickers(sp500_tickers + midcap_tickers + smallcap_tickers)
        except Exception as e:
            logger.error(f"Error loading S&P1500: {e}")
            logger.warning("Falling back to S&P 500 only")
            universe = clean_tickers(save_sp500_tickers())
            universe_name = "SPY (fallback)"
    elif universe_choice == 2:
        universe_name = "Russell 1000"
        try:
            universe = clean_tickers(pd.read_csv("russell1000.csv")['Symbol'].tolist())
        except FileNotFoundError:
            logger.warning("russell1000.csv not found. Falling back to S&P 500")
            universe = clean_tickers(save_sp500_tickers())
            universe_name = "SPY (fallback)"
    elif universe_choice == 3:
        universe_name = "Russell 3000"
        try:
            universe = clean_tickers(pd.read_csv("russell3000.csv")['Symbol'].tolist())
        except FileNotFoundError:
            logger.warning("russell3000.csv not found. Falling back to S&P 500")
            universe = clean_tickers(save_sp500_tickers())
            universe_name = "SPY (fallback)"
    elif universe_choice == 4:
        universe_name = "TSX Composite"
        try:
            df = pd.read_html('https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index')[0]
            universe = clean_tickers(df['Symbol'].tolist())
        except Exception as e:
            logger.error(f"Error loading TSX Composite: {e}")
            logger.warning("Falling back to S&P 500")
            universe = clean_tickers(save_sp500_tickers())
            universe_name = "SPY (fallback)"
    elif universe_choice == 5:
        custom_file = input("Enter path to text file with one ticker per line: ")
        universe_name = f"Custom ({os.path.basename(custom_file)})"
        try:
            with open(custom_file, 'r') as f:
                universe = clean_tickers([line.strip() for line in f.readlines()])
        except FileNotFoundError:
            logger.warning(f"File {custom_file} not found. Falling back to S&P 500")
            universe = clean_tickers(save_sp500_tickers())
            universe_name = "SPY (fallback)"
    else:
        logger.warning("Invalid choice. Using S&P 500")
        universe = clean_tickers(save_sp500_tickers())
        universe_name = "SPY (fallback)"
    
    return universe, universe_name

# ---------- CORE FUNCTIONALITY ----------
def momentum_screener(
    universe_choice=0,
    soft_breakout_pct=0.005,
    proximity_threshold=0.05,
    volume_threshold=1.2,
    lookback_days=365,
    data_source="yfinance",
    use_sample_data_on_failure=True,
    use_offline_mode=False,
    debug_mode=False
):
    """Execute momentum screening strategy"""
    # Create necessary directories
    os.makedirs("data_cache", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Get stock universe
    universe, universe_name = get_stock_universe(universe_choice)
    universe_cleaned = clean_tickers(universe)
    
    # Limit to 100 tickers for testing/reliability
    if len(universe_cleaned) > 100:
        logger.info(f"Limiting to 100 tickers for testing (from {len(universe_cleaned)})")
        universe_cleaned = universe_cleaned[:100]
    
    logger.info(f"\nUniverse Loaded: {universe_name} ({len(universe_cleaned)} tickers)")
    
    # Set date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)
    
    # OFFLINE MODE: Try to load from cache first
    data_dict = None
    if use_offline_mode:
        logger.info("OFFLINE MODE: Attempting to load from cache...")
        data_dict = load_cached_data(universe_name, lookback_days)
    
    # If not in offline mode or cache miss, try to download
    if data_dict is None:
        if use_offline_mode:
            logger.info("Cache miss in offline mode, using sample data")
            data_dict = get_sample_data(lookback_days)
        else:
            logger.info(f"Downloading data from {data_source}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            try:
                # Use the appropriate data source
                if data_source == "alphavantage":
                    logger.info("Using Alpha Vantage as data source")
                    data_dict = av_download_batch(universe_cleaned, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                elif data_source == "finnhub":
                    logger.info("Using Finnhub as data source")
                    data_dict = finnhub_download_batch(universe_cleaned, start_date, end_date)
                else:  # Default to Yahoo Finance
                    logger.info("Using Yahoo Finance with direct API access")
                    data_dict = download_yahoo_finance_direct(universe_cleaned, start_date, end_date)
                
                # Cache the data
                save_to_cache(data_dict, universe_name, lookback_days)
                
            except Exception as primary_e:
                logger.error(f"Primary data source {data_source} failed: {str(primary_e)}")
                
                if use_sample_data_on_failure:
                    logger.warning("Using sample data as fallback")
                    data_dict = get_sample_data(lookback_days)
    
    # Debug output
    if debug_mode and data_dict:
        logger.info(f"Got data for {len(data_dict)} tickers")
        if len(data_dict) > 0:
            sample_ticker = next(iter(data_dict))
            logger.info(f"Sample data for {sample_ticker}:")
            logger.info(f"Columns: {data_dict[sample_ticker].columns.tolist()}")
            logger.info(f"First few rows:\n{data_dict[sample_ticker].head()}")
    
    # Check if we have valid data at this point
    if not data_dict or len(data_dict) == 0:
        logger.error("No data available to analyze")
        return {
            "error": {
                "type": "DataError",
                "description": "No data available to analyze",
                "suggestion": "Try a different universe or data source"
            }
        }
    
    # Create Excel writer for output
    today = datetime.today().strftime('%Y-%m-%d')
    excel_path = f"outputs/{universe_name}_Momentum_Screener_{today}.xlsx"
    
    try:
        # Process data into DataFrames for the analyzer (back to the original approach)
        close_prices, volumes = process_stock_data(data_dict, lookback_days)
        
        if debug_mode:
            logger.info(f"Processed data shapes: Close prices {close_prices.shape}, Volumes {volumes.shape}")
        
        # Validate the processed data
        validation_result = validate_data(close_prices, volumes)
        if validation_result:
            logger.error(f"Data validation failed: {validation_result}")
            return validation_result
        
        # Data quality verification
        empty_price_count = (close_prices.isna().sum() > close_prices.shape[0] * 0.9).sum()
        if empty_price_count > 0:
            logger.warning(f"\n⚠️ WARNING: {empty_price_count} tickers have more than 90% missing price data")
        
        # Recent data check - ensure we have recent data
        if not close_prices.empty:
            latest_date = close_prices.index[-1].strftime('%Y-%m-%d')
            today = datetime.today().strftime('%Y-%m-%d')
            days_diff = (datetime.today() - close_prices.index[-1]).days
        
            if days_diff > 5:
                logger.warning(f"\n⚠️ WARNING: Most recent data is from {latest_date}, which is {days_diff} days old")
        
        # Calculate rolling highs and volume averages
        rolling_high = close_prices.rolling(lookback_days, min_periods=1).max()
        avg_vol_50 = volumes.rolling(50, min_periods=5).mean()
        
        # Get today's values
        current_close = close_prices.iloc[-1]
        rolling_high_today = rolling_high.iloc[-1]
        latest_volume = volumes.iloc[-1]
        latest_avg_volume = avg_vol_50.iloc[-1]
        
        # Calculate volume ratio and proximity
        volume_ratio = latest_volume / latest_avg_volume
        proximity = (rolling_high_today - current_close) / rolling_high_today
        
        # Debug output for calculations
        if debug_mode:
            logger.info("\n=== CALCULATION DEBUG ===")
            logger.info(f"Current close shape: {current_close.shape}")
            logger.info(f"Rolling high today shape: {rolling_high_today.shape}")
            logger.info(f"Volume ratio shape: {volume_ratio.shape}")
            logger.info(f"Proximity shape: {proximity.shape}")
            
            # Sample calculations for a few tickers
            for ticker in current_close.index[:5]:
                if pd.notna(current_close[ticker]) and pd.notna(rolling_high_today[ticker]):
                    logger.info(f"\n{ticker}:")
                    logger.info(f"  Current close: ${current_close[ticker]:.2f}")
                    logger.info(f"  52-week high: ${rolling_high_today[ticker]:.2f}, distance: {proximity[ticker]*100:.2f}%")
                    if pd.notna(volume_ratio[ticker]):
                        logger.info(f"  Volume ratio: {volume_ratio[ticker]:.2f}x")
                    else:
                        logger.info("  Volume ratio: N/A")
        
        # Identify breakouts and near breakouts
        universe_cleaned = proximity.index.tolist()
        
        # Convert to Series with index for easier filtering
        high_breakers = (proximity <= soft_breakout_pct) & (volume_ratio > volume_threshold)
        high_breakers = pd.Series(high_breakers, index=universe_cleaned).fillna(False)
        
        near_highs = (proximity <= proximity_threshold) & (~high_breakers) & (rolling_high_today > 0)
        near_highs = pd.Series(near_highs, index=universe_cleaned).fillna(False)
        
        # Log breakout statistics
        logger.info("=== Breakout Condition Stats ===")
        close_to_high = sum((proximity <= proximity_threshold) & (rolling_high_today > 0))
        high_volume = sum(volume_ratio > volume_threshold)
        soft_breakouts = sum(proximity <= soft_breakout_pct)
        
        logger.info(f"Tickers close to high ({proximity_threshold*100}% threshold): {close_to_high}")
        logger.info(f"Tickers with high volume ({volume_threshold}x threshold): {high_volume}")
        logger.info(f"Tickers at breakout level ({soft_breakout_pct*100}% threshold): {soft_breakouts}")
        logger.info(f"Breakouts with volume confirmation: {high_breakers.sum()}")
        
        # Create results
        breakouts = []
        for ticker in universe_cleaned:
            if high_breakers[ticker]:
                breakouts.append({
                    "Symbol": ticker,
                    "Price": round(float(current_close[ticker]), 2),
                    "52_Week_High": round(float(rolling_high_today[ticker]), 2),
                    "Distance_to_High_pct": round(float(proximity[ticker] * 100), 2),
                    "Volume_Ratio": round(float(volume_ratio[ticker]), 2)
                })
        
        near_breakouts = []
        for ticker in universe_cleaned:
            if near_highs[ticker]:
                near_breakouts.append({
                    "Symbol": ticker,
                    "Price": round(float(current_close[ticker]), 2),
                    "52_Week_High": round(float(rolling_high_today[ticker]), 2),
                    "Distance_to_High_pct": round(float(proximity[ticker] * 100), 2),
                    "Volume_Ratio": round(float(volume_ratio[ticker]), 2)
                })
        
        # Sort results by proximity to high and then volume ratio
        breakouts = sorted(breakouts, key=lambda x: (x["Distance_to_High_pct"], -x["Volume_Ratio"]))
        near_breakouts = sorted(near_breakouts, key=lambda x: (x["Distance_to_High_pct"], -x["Volume_Ratio"]))
        
        # Save to Excel
        with pd.ExcelWriter(excel_path) as writer:
            pd.DataFrame(breakouts).to_excel(writer, sheet_name="Breakouts", index=False)
            pd.DataFrame(near_breakouts).to_excel(writer, sheet_name="Near Breakouts", index=False)
        
        # Return results
        return {
            "universe_choice": universe_choice,
            "universe_name": universe_name,
            "soft_breakout_pct": soft_breakout_pct,
            "proximity_threshold": proximity_threshold,
            "volume_threshold": volume_threshold,
            "lookback_days": lookback_days,
            "breakouts": breakouts,
            "near_breakouts": near_breakouts,
            "excel_path": excel_path
        }
        
    except Exception as e:
        logger.error(f"Error in momentum calculation: {str(e)}")
        logger.error(traceback.format_exc())
        
        if use_sample_data_on_failure:
            logger.warning(f"Using sample data as fallback due to error: {str(e)}")
            data_dict = get_sample_data(lookback_days)
            # Process the sample data
            try:
                close_prices, volumes = process_stock_data(data_dict, lookback_days)
                # Run the rest of the function with the sample data
                # ... [core calculation with sample data]
            except Exception as sample_e:
                logger.error(f"Error processing sample data: {str(sample_e)}")
                return {
                    "error": {
                        "type": "ProcessingError",
                        "description": f"Error processing sample data: {str(sample_e)}",
                        "traceback": traceback.format_exc(),
                        "suggestion": "Try with different parameters"
                    }
                }
        else:
            return {
                "error": {
                    "type": "ProcessingError",
                    "description": f"Error processing data: {str(e)}",
                    "traceback": traceback.format_exc(),
                    "suggestion": "Try with sample data or different parameters"
                }
            }

async def generate_momentum_analysis(result):
    """Generate LLM analysis for momentum screening results"""
    try:
        # Load LLM config
        model = load_llm_config()
        
        # Check if model is properly configured
        if not model:
            logger.error("Failed to load LLM model")
            return {
                "reasoning": {
                    "parameter_validation": "Error initializing LLM model",
                    "hypothesis": "Standard momentum breakout strategy",
                    "calculation_explanation": "No LLM analysis available"
                },
                "result": result
            }
        
        # Prepare data for LLM
        data = {
            "universe_choice": result.get("universe_choice", 0),
            "universe_name": result.get("universe_name", "Unknown"),
            "soft_breakout_pct": result.get("soft_breakout_pct", 0.005),
            "proximity_threshold": result.get("proximity_threshold", 0.05),
            "volume_threshold": result.get("volume_threshold", 1.2),
            "lookback_days": result.get("lookback_days", 365),
            "breakouts_count": len(result.get("breakouts", [])),
            "near_breakouts_count": len(result.get("near_breakouts", [])),
            "top_breakouts": result.get("breakouts", [])[:5] if result.get("breakouts") else []
        }
        
        # Create simplified content for the LLM to reduce processing time
        content = f"""
        Analyze this momentum screening result:
        Universe: {data['universe_name']} (universe_choice: {data['universe_choice']})
        Parameters: 
          - soft_breakout_pct: {data['soft_breakout_pct']}
          - proximity_threshold: {data['proximity_threshold']}
          - volume_threshold: {data['volume_threshold']}
          - lookback_days: {data['lookback_days']}
        
        Results:
          - Found {data['breakouts_count']} breakouts
          - Found {data['near_breakouts_count']} near breakouts
        """
        
        # Add top breakout information if available
        if data['top_breakouts']:
            content += "\nTop breakouts:\n"
            for i, b in enumerate(data['top_breakouts'], 1):
                content += f"{i}. {b['Symbol']}: ${b['Price']} ({b['Distance_to_High_pct']}% from high, {b['Volume_Ratio']}x volume)\n"
        
        # Set a shorter timeout to avoid long waits
        logger.info("Sending request to LLM with 30-second timeout")
        try:
            # Use asyncio.wait_for to set a timeout
            response = await asyncio.wait_for(
                model.generate_content(
                    [SYSTEM_PROMPT, content],
                    generation_config={"temperature": 0.3}
                ),
                timeout=30  # Reduced timeout to 30 seconds
            )
            
            if response and response.text:
                logger.info("LLM response received, extracting JSON")
                json_str = _extract_json_from_markdown(response.text)
                try:
                    analysis_data = json.loads(json_str)
                    # Ensure the analysis data contains all required fields
                    if not all(key in analysis_data for key in ['reasoning', 'analysis', 'suggestions']):
                        logger.warning("LLM response missing required fields, using simplified analysis")
                        return _create_simplified_analysis(data)
                    
                    # Add the original result
                    analysis_data["result"] = result
                    return analysis_data
                    
                except json.JSONDecodeError:
                    logger.error("Failed to parse LLM response as JSON")
                    return _create_simplified_analysis(data)
                    
            else:
                # Create minimal response on failure
                logger.warning("Empty LLM response")
                return _create_simplified_analysis(data)
                
        except asyncio.TimeoutError:
            logger.warning("LLM analysis timed out after 30 seconds")
            return _create_simplified_analysis(data, error_msg="LLM analysis timed out")
            
        except Exception as e:
            logger.error(f"LLM generation error: {str(e)}")
            return _create_simplified_analysis(data, error_msg=f"LLM error: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in generate_momentum_analysis: {str(e)}")
        return {
            "reasoning": {
                "parameter_validation": f"Error: {str(e)}",
                "hypothesis": "Standard momentum breakout strategy",
                "calculation_explanation": "Identifying securities trading near their 52-week high with abnormal volume"
            },
            "analysis": {
                "market_context": f"Analysis of {result.get('universe_name', 'unknown')} using {result.get('lookback_days', 365)} days of historical data",
                "top_breakouts_analysis": f"Found {len(result.get('breakouts', []))} breakouts and {len(result.get('near_breakouts', []))} near breakouts",
                "data_quality_assessment": "Analysis performed with fallback due to LLM error"
            },
            "suggestions": {
                "parameter_adjustments": "Consider adjusting the proximity threshold or volume ratio requirements",
                "additional_filters": "Add filters for uptrending sectors or minimum market capitalization"
            },
            "result": result
        }

def _create_simplified_analysis(data, error_msg=None):
    """Create a simplified analysis when LLM fails"""
    return {
        "reasoning": {
            "parameter_validation": f"Error: {error_msg}" if error_msg else "Using provided parameters directly",
            "hypothesis": "Standard momentum breakout strategy using proximity to 52-week highs with volume confirmation",
            "calculation_explanation": "Identifying securities trading near their 52-week high with abnormal volume"
        },
        "analysis": {
            "market_context": f"Analysis of {data['universe_name']} using {data['lookback_days']} days of historical data",
            "top_breakouts_analysis": f"Found {data['breakouts_count']} breakouts and {data['near_breakouts_count']} near breakouts",
            "data_quality_assessment": "Analysis performed with fallback due to LLM error"
        },
        "suggestions": {
            "parameter_adjustments": "Consider adjusting the proximity threshold or volume ratio requirements",
            "additional_filters": "Add filters for uptrending sectors or minimum market capitalization"
        },
        "result": {
            "breakouts_count": data['breakouts_count'],
            "near_breakouts_count": data['near_breakouts_count'],
            "universe_choice": data['universe_choice'],
            "universe_name": data['universe_name'],
            "soft_breakout_pct": data['soft_breakout_pct'],
            "proximity_threshold": data['proximity_threshold'],
            "volume_threshold": data['volume_threshold'],
            "lookback_days": data['lookback_days'],
            "top_breakouts": data['top_breakouts']
        }
    }

def _extract_json_from_markdown(text):
    """Extract JSON content from markdown code blocks or return the original text if no blocks found."""
    # Pattern to match JSON inside markdown code blocks
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    
    import re
    match = re.search(json_pattern, text)
    if match:
        return match.group(1).strip()
    return text

async def run_momentum_screener(
    universe_choice=0,
    soft_breakout_pct=0.005,
    proximity_threshold=0.05,
    volume_threshold=1.2,
    lookback_days=365,
    use_llm=False,
    data_source="yfinance",
    use_offline_mode=False
):
    """Main function to run the momentum screener with optional LLM analysis"""
    try:
        logger.info("\n=== STARTING MOMENTUM SCREENER ===")
        logger.info(f"Parameters: universe={universe_choice}, soft_breakout={soft_breakout_pct}, " 
                    f"proximity={proximity_threshold}, volume={volume_threshold}, lookback={lookback_days}")
        logger.info(f"Using LLM: {use_llm}, Data source: {data_source}, Offline mode: {use_offline_mode}")
        
        # Run the screener with debug mode
        result = momentum_screener(
            universe_choice, 
            soft_breakout_pct,
            proximity_threshold,
            volume_threshold,
            lookback_days,
            data_source,
            use_sample_data_on_failure=True,
            use_offline_mode=use_offline_mode,
            debug_mode=True  # Enable debug mode
        )
        
        # Handle error case
        if "error" in result:
            logger.error(f"Screener returned an error: {result['error']['description']}")
            return {
                "reasoning": {
                    "parameter_validation": f"Data error: {result['error'].get('description', 'Unknown error')}",
                    "hypothesis": "Could not execute strategy due to data error",
                    "calculation_explanation": "No calculations performed due to data error"
                },
                "result": None,
                "execution_time": 0,
                "timestamp": datetime.now().isoformat()
            }
        
        # Add LLM analysis if requested
        if use_llm:
            try:
                logger.info("Running LLM analysis...")
                start_time = time.time()
                analysis = await generate_momentum_analysis(result)
                execution_time = time.time() - start_time
                analysis["execution_time"] = round(execution_time, 2)
                analysis["timestamp"] = datetime.now().isoformat()
                return analysis
            except Exception as llm_e:
                logger.error(f"LLM analysis failed: {str(llm_e)}")
                execution_time = time.time() - start_time
                return {
                    "reasoning": {
                        "parameter_validation": f"Error during LLM analysis: {str(llm_e)}",
                        "hypothesis": "Standard momentum breakout strategy",
                        "calculation_explanation": "Could not generate explanation due to LLM error"
                    },
                    "result": result,
                    "execution_time": round(execution_time, 2),
                    "timestamp": datetime.now().isoformat()
                }
        else:
            logger.info("Skipping LLM analysis as use_llm=False")
            return {
                "reasoning": {
                    "parameter_validation": "Using provided parameters directly",
                    "hypothesis": "Standard momentum breakout strategy",
                    "calculation_explanation": "Identifying stocks near 52-week highs with volume confirmation"
                },
                "result": result,
                "execution_time": 0,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error in run_momentum_screener: {str(e)}")
        return {
            "reasoning": {
                "parameter_validation": f"Error: {str(e)}",
                "hypothesis": "Failed to execute strategy",
                "calculation_explanation": "An unexpected error occurred"
            },
            "result": None,
            "execution_time": 0,
            "timestamp": datetime.now().isoformat()
        }

# ---------- COMMAND LINE INTERFACE ----------
def print_analysis_report(analysis_data):
    """Print formatted analysis report to console"""
    if "error" in analysis_data.get("result", {}):
        print("\n⚠️ ERROR ⚠️")
        error = analysis_data["result"]["error"]
        print(f"Type: {error['type']}")
        print(f"Description: {error['description']}")
        print(f"Suggestion: {error['suggestion']}")
        return
    
    print("\n=== MOMENTUM SCREENER ANALYSIS ===")
    
    if "reasoning" in analysis_data:
        print("\n🔍 STRATEGY SUMMARY")
        print(analysis_data["reasoning"].get("strategy_summary", "Standard momentum breakout strategy"))
        
        if "calculation_explanation" in analysis_data["reasoning"]:
            print("\n📊 CALCULATION METHOD")
            print(analysis_data["reasoning"]["calculation_explanation"])
    
    if "analysis" in analysis_data:
        print("\n📈 MARKET CONTEXT")
        print(analysis_data["analysis"].get("market_context", "No market context available"))
        
        if "top_breakouts_analysis" in analysis_data["analysis"]:
            print("\n🔝 TOP BREAKOUTS ANALYSIS")
            print(analysis_data["analysis"]["top_breakouts_analysis"])
    
    if "suggestions" in analysis_data:
        print("\n💡 SUGGESTIONS")
        if "parameter_adjustments" in analysis_data["suggestions"]:
            print("Parameter Adjustments:", analysis_data["suggestions"]["parameter_adjustments"])
        if "additional_filters" in analysis_data["suggestions"]:
            print("Additional Filters:", analysis_data["suggestions"]["additional_filters"])
    
    result = analysis_data["result"]
    
    print("\n=== BREAKOUTS ===")
    if not result["breakouts"]:
        print("None found")
    else:
        # Print the first 5 breakouts
        print(f"Found {len(result['breakouts'])} breakouts. Top 5:")
        for i, b in enumerate(result["breakouts"][:5], 1):
            print(f"{i}. {b['Symbol']}: ${b['Price']} ({b['Distance_to_High_pct']}% from high, {b['Volume_Ratio']}x volume)")
    
    print("\n=== NEAR BREAKOUTS ===")
    if not result["near_breakouts"]:
        print("None found")
    else:
        # Print the first 5 near breakouts
        print(f"Found {len(result['near_breakouts'])} near breakouts. Top 5:")
        for i, b in enumerate(result["near_breakouts"][:5], 1):
            print(f"{i}. {b['Symbol']}: ${b['Price']} ({b['Distance_to_High_pct']}% from high, {b['Volume_Ratio']}x volume)")
    
    print(f"\nFull results saved to Excel: {result['excel_path']}")

async def main():
    """Main entry point with CLI interface"""
    print("=== LLM-Enhanced Momentum Screener ===")
    print("This tool identifies stocks breaking out to new highs with volume confirmation")
    print("\n=== Universe Options ===")
    print("0 - SPY (S&P 500)")
    print("1 - S&P1500")
    print("2 - Russell 1000 (CSV required)")
    print("3 - Russell 3000 (CSV required)")
    print("4 - TSX Composite")
    print("5 - Custom (from text file)")
    
    print("\n=== Data Source Options ===")
    print("0 - Yahoo Finance (default)")
    print("1 - Alpha Vantage")
    print("2 - Finnhub")
    
    try:
        # Get user inputs with defaults
        universe_choice = int(input("\nSelect Universe [0-5, default=0]: ") or "0")
        data_source_choice = int(input("Select Data Source [0-2, default=2]: ") or "2")
        
        # Map data source choice to string
        data_source = "yfinance"
        if data_source_choice == 1:
            data_source = "alphavantage"
        elif data_source_choice == 2:
            data_source = "finnhub"
        
        # Other parameter inputs remain the same
        soft_breakout_pct = float(input("Soft Breakout Percentage [0.001-0.02, default=0.005]: ") or "0.005")
        proximity_threshold = float(input("Proximity Threshold [0.01-0.1, default=0.05]: ") or "0.05")
        volume_threshold = float(input("Volume Threshold [1.0-3.0, default=1.2]: ") or "1.2")
        lookback_days = int(input("Lookback Days [90-365, default=365]: ") or "365")
        use_llm_input = input("Use LLM for enhanced analysis? [y/N, default=N]: ").lower()
        use_llm = use_llm_input == 'y' or use_llm_input == 'yes'
        
        # Validate inputs
        if not (0 <= universe_choice <= 5):
            universe_choice = 0
            print("Invalid universe choice. Using S&P 500.")
        
        if not (0.001 <= soft_breakout_pct <= 0.02):
            soft_breakout_pct = 0.005
            print("Invalid soft breakout percentage. Using 0.005.")
        
        if not (0.01 <= proximity_threshold <= 0.1):
            proximity_threshold = 0.05
            print("Invalid proximity threshold. Using 0.05.")
        
        if not (1.0 <= volume_threshold <= 3.0):
            volume_threshold = 1.2
            print("Invalid volume threshold. Using 1.2.")
        
        if not (90 <= lookback_days <= 365):
            lookback_days = 365
            print("Invalid lookback days. Using 365.")
        
        # Run the screener with the selected data source
        analysis = await run_momentum_screener(
            universe_choice,
            soft_breakout_pct,
            proximity_threshold,
            volume_threshold,
            lookback_days,
            use_llm,
            data_source
        )
        
        # Print the results
        print_analysis_report(analysis)
        
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

def process_stock_data(data_dict, lookback_days=365):
    """Process stock data dictionary into DataFrames for analysis"""
    # Create empty DataFrames for close prices and volumes
    close_prices = pd.DataFrame()
    volumes = pd.DataFrame()
    
    if not data_dict or len(data_dict) == 0:
        logger.warning("No data to process")
        return close_prices, volumes
    
    # Process each ticker's data
    for ticker, df in data_dict.items():
        try:
            if 'Close' in df.columns and 'Volume' in df.columns:
                # Add close prices and volumes to respective DataFrames
                close_prices[ticker] = df['Close']
                volumes[ticker] = df['Volume']
            else:
                missing_cols = []
                if 'Close' not in df.columns:
                    missing_cols.append('Close')
                if 'Volume' not in df.columns:
                    missing_cols.append('Volume')
                logger.warning(f"Missing columns for {ticker}: {', '.join(missing_cols)}")
        except Exception as e:
            logger.error(f"Error processing {ticker} data: {str(e)}")
    
    # Log some statistics
    logger.info(f"Processed {len(close_prices.columns)} tickers with valid price data")
    logger.info(f"Processed {len(volumes.columns)} tickers with valid volume data")
    
    return close_prices, volumes

def get_sample_data(lookback_days=365):
    """Generate sample stock data for testing when APIs fail"""
    logger.info("Generating sample stock data for testing")
    
    # Sample tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
    
    # Create date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Dictionary to store sample data
    data_dict = {}
    
    for ticker in tickers:
        # Create a dataframe with random price data
        np.random.seed(hash(ticker) % 10000)  # Different seed for each ticker
        
        # Start with a base price between $50 and $500
        base_price = np.random.uniform(50, 500)
        
        # Generate price fluctuations
        price_series = np.cumprod(1 + np.random.normal(0.0005, 0.015, len(dates)))
        prices = base_price * price_series
        
        # Create DataFrame
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = df['Close'] * np.random.uniform(0.98, 1.02, len(df))
        df['High'] = df[['Open', 'Close']].max(axis=1) * np.random.uniform(1.001, 1.02, len(df))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * np.random.uniform(0.98, 0.999, len(df))
        df['Volume'] = np.random.randint(100000, 10000000, size=len(df))
        
        # Ensure the high of the last 30 days is about 3-5% above current price
        # to create interesting near-breakout situations
        if len(df) > 30:
            last_price = df.iloc[-1]['Close']
            df.iloc[-30:-10]['High'] = last_price * np.random.uniform(1.03, 1.05, 20)
        
        data_dict[ticker] = df
        
    logger.info(f"Generated sample data for {len(tickers)} tickers")
    return data_dict

def load_cached_data(universe_name, lookback_days):
    """Load data from cache if available and not too old"""
    cache_dir = os.path.join("data_cache", f"{universe_name}_{lookback_days}days")
    cache_file = os.path.join(cache_dir, "stock_data.pkl")
    
    if os.path.exists(cache_file):
        # Check if cache is less than 7 days old
        cache_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days
        
        if cache_age_days < 7:
            logger.info(f"Loading cached data from {cache_file} (age: {cache_age_days} days)")
            try:
                with open(cache_file, "rb") as f:
                    data_dict = pickle.load(f)
                    
                # Validate the data
                if data_dict and len(data_dict) > 0:
                    sample_ticker = list(data_dict.keys())[0]
                    sample_df = data_dict[sample_ticker]
                    
                    if not sample_df.empty and 'Close' in sample_df.columns:
                        logger.info(f"Cache hit! Loaded data for {len(data_dict)} tickers")
                        return data_dict
                    else:
                        logger.warning("Cached data is invalid (missing required columns)")
                else:
                    logger.warning("Cached data is empty")
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
        else:
            logger.info(f"Cache is {cache_age_days} days old (> 7 days), will download fresh data")
    else:
        logger.info(f"No cache file found at {cache_file}")
    
    return None

def save_to_cache(data_dict, universe_name, lookback_days):
    """Save data to cache for future use"""
    if not data_dict or len(data_dict) == 0:
        logger.warning("No data to cache")
        return False
    
    try:
        cache_dir = os.path.join("data_cache", f"{universe_name}_{lookback_days}days")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, "stock_data.pkl")
        
        with open(cache_file, "wb") as f:
            pickle.dump(data_dict, f)
            
        logger.info(f"Cached data for {len(data_dict)} tickers to {cache_file}")
        return True
    except Exception as e:
        logger.error(f"Error caching data: {str(e)}")
        return False

def download_yahoo_finance_direct(tickers, start_date, end_date):
    """
    Download data directly from Yahoo Finance API without using yfinance library
    This is more reliable when yfinance has issues with rate limiting
    """
    logger.info(f"Downloading data directly from Yahoo Finance API for {len(tickers)} tickers")
    
    # Import required libraries
    try:
        from curl_cffi import requests as curl_requests
        import pandas as pd
        from datetime import datetime
        import time
        logger.info("Using curl_cffi for direct Yahoo Finance API access")
    except ImportError:
        logger.error("curl_cffi not installed. Installing now...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "curl-cffi"])
            from curl_cffi import requests as curl_requests
            logger.info("Successfully installed curl_cffi")
        except Exception as e:
            logger.error(f"Failed to install curl_cffi: {str(e)}")
            logger.warning("Cannot proceed without curl_cffi for Yahoo Finance")
            return {}
    
    # Create a session with Chrome browser fingerprint
    session = curl_requests.Session(impersonate="chrome")
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    # Convert dates to Unix timestamps
    start_timestamp = int(datetime.timestamp(start_date))
    end_timestamp = int(datetime.timestamp(end_date))
    
    all_data = {}
    
    for i, ticker in enumerate(tickers):
        # Add delay between requests
        if i > 0:
            # More aggressive delays to avoid rate limits
            if i % 20 == 0:
                logger.info(f"Reached 20 requests - sleeping for 30 seconds...")
                time.sleep(30)
            elif i % 5 == 0:
                logger.info(f"Reached 5 requests - sleeping for 10 seconds...")
                time.sleep(10)
            else:
                time.sleep(2)
        
        logger.info(f"Downloading {ticker} ({i+1}/{len(tickers)})")
        
        try:
            # Construct the Yahoo Finance API URL
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_timestamp}&period2={end_timestamp}&interval=1d&events=history&includeAdjustedClose=true"
            
            # Make the request with Chrome fingerprinting
            response = session.get(url)
            
            if response.status_code != 200:
                logger.error(f"Error downloading {ticker}: HTTP {response.status_code}")
                continue
            
            # Parse the JSON response
            data = response.json()
            
            # Check if we have valid data
            if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result'] or data['chart']['result'][0] is None:
                logger.warning(f"No data available for {ticker}")
                continue
            
            # Extract the data
            chart_data = data['chart']['result'][0]
            
            # Check if we have timestamp and quote data
            if ('timestamp' not in chart_data or 
                'indicators' not in chart_data or 
                'quote' not in chart_data['indicators'] or 
                not chart_data['indicators']['quote'] or 
                chart_data['indicators']['quote'][0] is None):
                logger.warning(f"Incomplete data for {ticker}")
                continue
            
            # Extract timestamps and price/volume data
            timestamps = chart_data['timestamp']
            quote_data = chart_data['indicators']['quote'][0]
            
            # Check for minimal required data
            if not timestamps or 'close' not in quote_data or not quote_data['close']:
                logger.warning(f"Missing required data for {ticker}")
                continue
            
            # Get adjusted close if available
            adjclose_data = None
            if 'adjclose' in chart_data['indicators'] and chart_data['indicators']['adjclose']:
                adjclose_data = chart_data['indicators']['adjclose'][0]['adjclose']
            
            # Create a DataFrame with the data
            df = pd.DataFrame()
            df['timestamp'] = timestamps
            df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('Date', inplace=True)
            df.drop('timestamp', axis=1, inplace=True)
            
            # Add price and volume data
            for field in ['open', 'high', 'low', 'close', 'volume']:
                if field in quote_data and quote_data[field]:
                    # Use title case for column names (Open, High, Low, Close, Volume)
                    df[field.title()] = quote_data[field]
            
            # Add adjusted close if available
            if adjclose_data:
                df['Adj Close'] = adjclose_data
            
            # Standard columns for compatibility with yfinance
            if 'Adj Close' not in df.columns and 'Close' in df.columns:
                df['Adj Close'] = df['Close']
            
            # Make sure the standard yfinance columns exist in the DataFrame
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Missing required column {col} for {ticker}")
                    return {}
            
            # Store the data if not empty
            if not df.empty:
                all_data[ticker] = df
                logger.info(f"✓ Successfully downloaded {ticker}: {len(df)} days of data")
            else:
                logger.warning(f"Empty data for {ticker}")
                
        except Exception as e:
            logger.error(f"× Error downloading {ticker}: {str(e)}")
    
    # Log summary statistics
    logger.info(f"\n=== Yahoo Finance Download Summary ===")
    logger.info(f"Total tickers: {len(tickers)}")
    logger.info(f"Successfully downloaded: {len(all_data)}")
    logger.info(f"Failed: {len(tickers) - len(all_data)}")
    
    return all_data

def validate_data(close_prices, volumes):
    """Validate stock data and check for quality issues"""
    # Check if we have valid data
    if close_prices.empty or volumes.empty:
        logger.error("No valid price/volume data found")
        return {
            "error": {
                "type": "DataError",
                "description": "No valid price/volume data found",
                "suggestion": "Try a different universe or check internet connection"
            }
        }
    
    # Check if we have enough tickers
    if len(close_prices.columns) < 5:
        logger.warning(f"Only {len(close_prices.columns)} tickers have valid data")
        logger.warning("This may not be enough for meaningful analysis")
        # Continue anyway but log the warning
    
    # All checks passed
    return None