from momentum_screener_llm import momentum_screener
import os
import json

print("Checking directories...")
print(f"Current directory: {os.getcwd()}")
os.makedirs("data_cache", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
print(f"data_cache exists: {os.path.exists('data_cache')}")
print(f"outputs exists: {os.path.exists('outputs')}")

print("\nRunning momentum screener with relaxed parameters...")
result = momentum_screener(
    universe_choice=0,
    soft_breakout_pct=0.02,      # 2% instead of 0.5%
    proximity_threshold=0.1,     # 10% instead of 5%
    volume_threshold=1.0,        # Equal to average instead of 1.2x
    lookback_days=90,            # 3 months instead of 1 year
    use_llm=False                # Test without LLM first
)

print("\nResult:")
print(f"Breakouts found: {len(result.get('breakouts', []))}")
print(f"Near breakouts found: {len(result.get('near_breakouts', []))}")
print(f"Excel path: {result.get('excel_path', 'Not found')}")

if 'excel_path' in result:
    print(f"Excel file exists: {os.path.exists(result['excel_path'])}")

with open("debug_output.json", "w") as f:
    json.dump(result, f, indent=2)
    print(f"\nFull result saved to debug_output.json") 