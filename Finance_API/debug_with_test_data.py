import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta

# Create test directory structure
os.makedirs("data_cache", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Create sample data
print("Creating test data...")
num_days = 100
end_date = datetime.today()
dates = [end_date - timedelta(days=i) for i in range(num_days)]
dates.reverse()  # Put in chronological order

test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
test_data = {}

# Create sample price and volume data
for ticker in test_tickers:
    # Create a dataframe with dates
    df = pd.DataFrame(index=dates)
    df.index.name = 'Date'
    
    # Generate some reasonable price data with an uptrend
    base_price = np.random.uniform(100, 500)
    prices = np.cumsum(np.random.normal(0.1, 1, num_days)) + base_price
    df['Close'] = prices
    
    # Generate some volume data
    base_volume = np.random.uniform(1000000, 10000000)
    df['Volume'] = np.random.normal(base_volume, base_volume/5, num_days)
    
    # Add to test data
    test_data[ticker] = df

# Now run the momentum analysis on this test data
print("\nRunning test momentum analysis...")

# Extract close prices and volumes into DataFrames
close_prices = pd.DataFrame()
volumes = pd.DataFrame()

for ticker, data in test_data.items():
    close_prices[ticker] = data['Close']
    volumes[ticker] = data['Volume']

# Get current close prices
current_close = close_prices.iloc[-1]

# Calculate rolling high and proximity to high
lookback_days = 90
rolling_high = close_prices.rolling(window=lookback_days, min_periods=1).max()
rolling_high_today = rolling_high.iloc[-1]
proximity = (rolling_high_today - current_close) / rolling_high_today

# Calculate volume ratio (current vs average)
current_volume = volumes.iloc[-1]
avg_volume = volumes.iloc[-20:].mean()  # 20-day average volume
volume_ratio = current_volume / avg_volume

# Parameters
soft_breakout_pct = 0.02     # 2% from high
proximity_threshold = 0.1    # 10% from high
volume_threshold = 1.0       # Equal to average

# Identify breakouts and near breakouts
high_breakers = (proximity <= soft_breakout_pct) & (volume_ratio > volume_threshold)
near_highs = (proximity <= proximity_threshold) & (~high_breakers) & (rolling_high_today > 0)

# Convert to series with index for easier filtering
high_breakers = pd.Series(high_breakers, index=test_tickers).fillna(False)
near_highs = pd.Series(near_highs, index=test_tickers).fillna(False)

# Create results
breakouts = []
for ticker in high_breakers[high_breakers].index:
    breakouts.append({
        "Symbol": ticker,
        "Price": round(float(current_close[ticker]), 2),
        "52_Week_High": round(float(rolling_high_today[ticker]), 2),
        "Distance_to_High_pct": round(float(proximity[ticker] * 100), 2),
        "Volume_Ratio": round(float(volume_ratio[ticker]), 2)
    })

near_breakouts = []
for ticker in near_highs[near_highs].index:
    near_breakouts.append({
        "Symbol": ticker,
        "Price": round(float(current_close[ticker]), 2),
        "52_Week_High": round(float(rolling_high_today[ticker]), 2),
        "Distance_to_High_pct": round(float(proximity[ticker] * 100), 2),
        "Volume_Ratio": round(float(volume_ratio[ticker]), 2)
    })

# Export to Excel
today = datetime.today().strftime('%Y-%m-%d')
excel_path = f"outputs/TEST_Momentum_Screener_{today}.xlsx"

with pd.ExcelWriter(excel_path) as writer:
    pd.DataFrame(breakouts).to_excel(writer, sheet_name="Breakouts", index=False)
    pd.DataFrame(near_breakouts).to_excel(writer, sheet_name="Near Breakouts", index=False)

# Create result structure
result = {
    "breakouts": breakouts,
    "near_breakouts": near_breakouts,
    "excel_path": excel_path,
    "parameters": {
        "universe": "Test Data",
        "soft_breakout_pct": soft_breakout_pct,
        "proximity_threshold": proximity_threshold,
        "volume_threshold": volume_threshold,
        "lookback_days": lookback_days
    }
}

print("\nResult:")
print(f"Breakouts found: {len(result['breakouts'])}")
print(f"Near breakouts found: {len(result['near_breakouts'])}")
print(f"Excel path: {result.get('excel_path')}")
print(f"Excel file exists: {os.path.exists(result['excel_path'])}")

with open("debug_output.json", "w") as f:
    json.dump(result, f, indent=2)
    print("\nFull result saved to debug_output.json") 