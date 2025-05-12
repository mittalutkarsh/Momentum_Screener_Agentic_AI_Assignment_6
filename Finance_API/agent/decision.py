"""
decision.py - Decision Making Module for Stock Analysis Agent

This module processes stock data and makes analytical decisions.
"""
import logging
import random
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger('stock_agent.decision')

class StockAnalyzer:
    """Class for analyzing stock data and making decisions"""
    
    def __init__(self):
        """Initialize the stock analyzer"""
        pass
    
    def analyze_momentum(
        self, 
        close_prices: pd.DataFrame, 
        volumes: pd.DataFrame,
        universe_cleaned: List[str],
        soft_breakout_pct: float = 0.005,
        proximity_threshold: float = 0.05,
        volume_threshold: float = 1.2,
        lookback_days: int = 365
    ) -> Dict[str, Any]:
        """Analyze momentum patterns in stock data"""
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
        
        # Sample verification (random 5 tickers)
        if len(universe_cleaned) > 5:
            sample_tickers = random.sample(universe_cleaned, 5)
            logger.info("=== Sample Data Verification ===")
            for ticker in sample_tickers:
                logger.info(
                    f"{ticker}: latest price: ${current_close[ticker]:.2f}, "
                    f"52-week high: ${rolling_high_today[ticker]:.2f}, "
                    f"distance: {proximity[ticker]*100:.2f}%, "
                    f"volume ratio: {volume_ratio[ticker]:.2f}x"
                )
        
        # Identify breakouts and near breakouts
        high_breakers = (proximity <= soft_breakout_pct) & (volume_ratio > volume_threshold)
        near_highs = (proximity <= proximity_threshold) & (~high_breakers) & \
                    (rolling_high_today > 0)
        
        # Convert to series with index for easier filtering
        high_breakers = pd.Series(high_breakers, index=universe_cleaned).fillna(False)
        near_highs = pd.Series(near_highs, index=universe_cleaned).fillna(False)
        
        # Log breakout statistics
        logger.info("=== Breakout Condition Stats ===")
        close_to_high = (proximity <= proximity_threshold).sum()
        high_volume = (volume_ratio > volume_threshold).sum()
        soft_breakouts = (proximity <= soft_breakout_pct).sum()

        logger.info(
            f"Tickers close to high ({proximity_threshold*100}% threshold): {close_to_high}"
        )
        logger.info(
            f"Tickers with high volume ({volume_threshold}x threshold): {high_volume}"
        )
        logger.info(
            f"Tickers at breakout level ({soft_breakout_pct*100}% threshold): "
            f"{soft_breakouts}"
        )
        logger.info(f"Breakouts with volume confirmation: {high_breakers.sum()}")
        
        # Create results
        breakouts = []
        near_breakouts = []
        
        # Populate breakouts list
        for ticker in high_breakers[high_breakers].index:
            breakouts.append({
                "Symbol": ticker,
                "Price": round(float(current_close[ticker]), 2),
                "52_Week_High": round(float(rolling_high_today[ticker]), 2),
                "Distance_to_High_pct": round(float(proximity[ticker] * 100), 2),
                "Volume_Ratio": round(float(volume_ratio[ticker]), 2)
            })
        
        # Populate near breakouts list
        for ticker in near_highs[near_highs].index:
            near_breakouts.append({
                "Symbol": ticker,
                "Price": round(float(current_close[ticker]), 2),
                "52_Week_High": round(float(rolling_high_today[ticker]), 2),
                "Distance_to_High_pct": round(float(proximity[ticker] * 100), 2),
                "Volume_Ratio": round(float(volume_ratio[ticker]), 2)
            })
        
        # Sort by proximity to high
        breakouts.sort(key=lambda x: x["Distance_to_High_pct"])
        near_breakouts.sort(key=lambda x: x["Distance_to_High_pct"])
        
        return {
            "breakouts": breakouts,
            "near_breakouts": near_breakouts
        }
    
    def validate_data(self, close_prices: pd.DataFrame, volumes: pd.DataFrame) -> Dict[str, Any]:
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
        
        # Data quality verification
        empty_price_count = (close_prices.isna().sum() > close_prices.shape[0] * 0.9).sum()
        if empty_price_count > 0:
            logger.warning(
                f"WARNING: {empty_price_count} tickers have more than 90% missing price data"
            )

        # Recent data check
        from datetime import datetime
        if not close_prices.empty:
            latest_date = close_prices.index[-1].strftime('%Y-%m-%d')
            days_diff = (datetime.today() - close_prices.index[-1]).days

            if days_diff > 5:
                logger.warning(
                    f"WARNING: Most recent data is from {latest_date}, "
                    f"which is {days_diff} days old"
                )
        
        return {"status": "valid"} 