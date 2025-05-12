"""
memory.py - Memory Module for Stock Analysis Agent

This module handles data storage, retrieval, and preprocessing for stock analysis.
"""
import os
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import yfinance as yf
import requests
import bs4 as bs
from functools import lru_cache

logger = logging.getLogger('stock_agent.memory')

class DataMemory:
    """Class for handling data storage and retrieval for stock analysis"""
    
    def __init__(self):
        """Initialize the data memory module"""
        self.data_cache = {}
        os.makedirs("data_cache", exist_ok=True)
    
    @lru_cache(maxsize=5)
    def get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 tickers from Wikipedia or cached data"""
        try:
            # Check if we have a recent cache
            cache_path = "data_cache/sp500tickers.pickle"
            if os.path.exists(cache_path):
                # Check if cache is less than 7 days old
                if (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))).days < 7:
                    with open(cache_path, "rb") as f:
                        return pickle.load(f)
            
            # Scrape fresh data
            logger.info("Scraping S&P 500 tickers from Wikipedia")
            resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            soup = bs.BeautifulSoup(resp.text, 'lxml')
            
            tables = soup.find_all('table', {'class': 'wikitable'})
            if not tables:
                raise ValueError("No tables with class 'wikitable' found")
            
            table = tables[0]
            tickers = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if cells:
                    ticker = cells[0].text.strip()
                    if ticker:
                        tickers.append(ticker)
            
            if not tickers:
                raise ValueError("No tickers found in the table")
            
            # Save to cache
            with open(cache_path, "wb") as f:
                pickle.dump(tickers, f)
            
            return tickers
            
        except Exception as e:
            logger.error(f"Error getting S&P 500 tickers: {e}")
            # Return fallback list
            return [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "PG",
                "UNH", "XOM", "JNJ", "WMT", "MA", "LLY", "CVX", "HD", "AVGO", "MRK"
            ]
    
    def get_stock_universe(self, universe_choice: int) -> Tuple[List[str], str]:
        """Get stock universe based on user selection"""
        if universe_choice == 0:
            universe_name = "S&P 500"
            tickers = self.get_sp500_tickers()
            universe = self._clean_tickers(tickers)
            
        elif universe_choice == 1:
            universe_name = "S&P 1500"
            try:
                # Get S&P 500
                sp500_tickers = self.get_sp500_tickers()
                
                # Get S&P MidCap 400 and SmallCap 600
                df2 = pd.read_html(
                    'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
                )[0]
                df3 = pd.read_html(
                    'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'
                )[0]
                
                midcap_tickers = df2['Symbol'].tolist()
                smallcap_tickers = df3['Symbol'].tolist()
                
                universe = self._clean_tickers(
                    sp500_tickers + midcap_tickers + smallcap_tickers
                )
            except Exception as e:
                logger.error(f"Error loading S&P 1500: {e}")
                universe = self._clean_tickers(self.get_sp500_tickers())
                universe_name = "S&P 500 (fallback)"
                
        elif universe_choice == 2:
            universe_name = "Russell 1000"
            try:
                universe = self._clean_tickers(
                    pd.read_csv("data_cache/russell1000.csv")['Symbol'].tolist()
                )
            except FileNotFoundError:
                logger.warning("russell1000.csv not found. Falling back to S&P 500")
                universe = self._clean_tickers(self.get_sp500_tickers())
                universe_name = "S&P 500 (fallback)"
                
        elif universe_choice == 3:
            universe_name = "Russell 3000"
            try:
                universe = self._clean_tickers(
                    pd.read_csv("data_cache/russell3000.csv")['Symbol'].tolist()
                )
            except FileNotFoundError:
                logger.warning("russell3000.csv not found. Falling back to S&P 500")
                universe = self._clean_tickers(self.get_sp500_tickers())
                universe_name = "S&P 500 (fallback)"
                
        elif universe_choice == 4:
            universe_name = "TSX Composite"
            try:
                df = pd.read_html(
                    'https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index'
                )[0]
                universe = self._clean_tickers(df['Symbol'].tolist())
            except Exception as e:
                logger.error(f"Error loading TSX Composite: {e}")
                universe = self._clean_tickers(self.get_sp500_tickers())
                universe_name = "S&P 500 (fallback)"
                
        elif universe_choice == 5:
            file_path = input("Enter path to text file with one ticker per line: ")
            universe_name = f"Custom ({os.path.basename(file_path)})"
            try:
                with open(file_path, 'r') as f:
                    universe = self._clean_tickers([line.strip() for line in f.readlines()])
            except FileNotFoundError:
                logger.warning(f"File {file_path} not found. Falling back to S&P 500")
                universe = self._clean_tickers(self.get_sp500_tickers())
                universe_name = "S&P 500 (fallback)"
        else:
            logger.warning("Invalid choice. Using S&P 500")
            universe = self._clean_tickers(self.get_sp500_tickers())
            universe_name = "S&P 500 (fallback)"
        
        return universe, universe_name
    
    def _clean_tickers(self, tickers: List[str]) -> List[str]:
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
    
    def fetch_stock_data(self, tickers: List[str], lookback_days: int) -> Dict[str, pd.DataFrame]:
        """Fetch historical stock data for the given tickers"""
        from datetime import datetime, timedelta
        
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback_days)
        
        logger.info(
            f"Fetching data from {start_date.strftime('%Y-%m-%d')} "
            f"to {end_date.strftime('%Y-%m-%d')}"
        )
        
        return self._download_in_batches(tickers, start_date, end_date)
    
    def _download_in_batches(self, tickers, start_date, end_date, batch_size=100):
        """Download data in batches to avoid timeouts"""
        import time
        
        all_data = {}
        num_batches = len(tickers) // batch_size + (1 if len(tickers) % batch_size != 0 else 0)
        
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(tickers))
            batch_tickers = tickers[batch_start:batch_end]
            
            logger.info(
                f"Downloading batch {i+1}/{num_batches} "
                f"({batch_start+1}-{batch_end} of {len(tickers)} tickers)"
            )
            
            # Try up to 3 times with exponential backoff
            for attempt in range(3):
                try:
                    batch_data = yf.download(
                        batch_tickers, 
                        start=start_date, 
                        end=end_date, 
                        group_by="ticker", 
                        auto_adjust=True,
                        progress=False
                    )
                    
                    # Process the batch data
                    if isinstance(batch_data, pd.DataFrame) and len(batch_tickers) == 1:
                        # Special case for single ticker
                        all_data[batch_tickers[0]] = batch_data
                    else:
                        # For multiple tickers
                        for ticker in batch_tickers:
                            if ticker in batch_data.columns.levels[0]:
                                all_data[ticker] = batch_data[ticker]
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff
                    logger.warning(
                        f"Batch download failed (attempt {attempt+1}/3). Error: {str(e)}"
                    )
                    if attempt < 2:
                        logger.info(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
        
        return all_data
    
    def extract_price_data(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract close prices and volumes into DataFrames"""
        close_prices = pd.DataFrame()
        volumes = pd.DataFrame()
        
        for ticker, ticker_data in data_dict.items():
            try:
                # Handle different data structures
                if 'Close' in ticker_data.columns:
                    close_prices[ticker] = ticker_data['Close']
                    volumes[ticker] = ticker_data['Volume']
                else:
                    # For MultiIndex columns
                    close_prices[ticker] = ticker_data[('Close', ticker)] \
                        if ('Close', ticker) in ticker_data.columns \
                        else ticker_data['Close']
                    volumes[ticker] = ticker_data[('Volume', ticker)] \
                        if ('Volume', ticker) in ticker_data.columns \
                        else ticker_data['Volume']
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
        
        return close_prices, volumes
    
    def save_results(self, result_data: Dict[str, Any], universe_name: str) -> str:
        """Save analysis results to Excel"""
        today = datetime.today().strftime('%Y-%m-%d')
        os.makedirs("outputs", exist_ok=True)
        output_path = f"outputs/{universe_name}_Stock_Analysis_{today}.xlsx"
        
        # Create Excel writer
        with pd.ExcelWriter(output_path) as writer:
            # Write breakouts to sheet
            if "breakouts" in result_data:
                pd.DataFrame(result_data["breakouts"]).to_excel(
                    writer, sheet_name="Breakouts", index=False
                )
                
            # Write near breakouts to sheet
            if "near_breakouts" in result_data:
                pd.DataFrame(result_data["near_breakouts"]).to_excel(
                    writer, sheet_name="Near Breakouts", index=False
                )
        
        logger.info(f"Results saved to: {output_path}")
        return output_path 