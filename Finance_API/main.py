#!/usr/bin/env python3
"""
main.py - Stock Analysis Agent

This is the main entry point for the modular stock analysis agent.
It integrates the perception, memory, decision, and action modules.
"""
import os
import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stock_agent')

# Import agent modules
from agent.perception import LLMPerception
from agent.memory import DataMemory
from agent.decision import StockAnalyzer
from agent.action import ActionHandler

class StockAnalysisAgent:
    """Main Stock Analysis Agent class that integrates all modules"""
    
    def __init__(self):
        """Initialize all agent components"""
        self.perception = LLMPerception()
        self.memory = DataMemory()
        self.decision = StockAnalyzer()
        self.action = ActionHandler()
    
    async def run_analysis(
        self,
        universe_choice=0,
        soft_breakout_pct=0.005,
        proximity_threshold=0.05,
        volume_threshold=1.2,
        lookback_days=365,
        use_llm=True
    ):
        """Run the complete stock analysis pipeline"""
        # 1. Get stock universe from memory
        universe, universe_name = self.memory.get_stock_universe(universe_choice)
        logger.info(f"Universe loaded: {universe_name} ({len(universe)} tickers)")
        
        # 2. Fetch historical data
        stock_data = self.memory.fetch_stock_data(universe, lookback_days)
        universe_cleaned = list(stock_data.keys())
        logger.info(f"Data downloaded for {len(universe_cleaned)} tickers")
        
        # 3. Extract price and volume data
        close_prices, volumes = self.memory.extract_price_data(stock_data)
        
        # 4. Validate the data
        validation = self.decision.validate_data(close_prices, volumes)
        if "error" in validation:
            return validation
        
        # 5. Analyze the data with the decision module
        analysis_results = self.decision.analyze_momentum(
            close_prices, 
            volumes,
            universe_cleaned,
            soft_breakout_pct,
            proximity_threshold,
            volume_threshold,
            lookback_days
        )
        
        # 6. Save results to Excel
        excel_path = self.memory.save_results(analysis_results, universe_name)
        
        # 7. Create complete result object
        result_data = {
            **analysis_results,
            "excel_path": excel_path,
            "parameters": {
                "universe": universe_name,
                "soft_breakout_pct": soft_breakout_pct,
                "proximity_threshold": proximity_threshold,
                "volume_threshold": volume_threshold,
                "lookback_days": lookback_days
            }
        }
        
        # 8. Use LLM for enhanced analysis if requested
        if use_llm:
            logger.info("Running LLM analysis...")
            enhanced_analysis = await self.perception.analyze_data(result_data)
            
            # 9. Generate and return the report
            report = self.action.generate_report(enhanced_analysis)
            return {
                "report": report,
                "data": enhanced_analysis
            }
        else:
            # Create simple analysis object
            simple_analysis = {
                "reasoning": {
                    "parameter_check": "Using provided parameters directly",
                    "strategy_summary": "Standard momentum analysis strategy",
                    "verification": (
                        f"Found {len(result_data['breakouts'])} breakouts and "
                        f"{len(result_data['near_breakouts'])} near-breakouts"
                    )
                },
                "result": result_data
            }
            
            # Generate and return the report
            report = self.action.generate_report(simple_analysis)
            return {
                "report": report,
                "data": simple_analysis
            }

async def main():
    """Main entry point with CLI interface"""
    print("=== Stock Analysis Agent ===")
    print("This tool analyzes stocks based on momentum and other technical factors")
    print("\n=== Universe Options ===")
    print("0 - S&P 500")
    print("1 - S&P 1500")
    print("2 - Russell 1000 (CSV required)")
    print("3 - Russell 3000 (CSV required)")
    print("4 - TSX Composite")
    print("5 - Custom (from text file)")
    
    try:
        # Get user inputs with defaults
        universe_choice = int(
            input("\nSelect Universe [0-5, default=0]: ") or "0"
        )
        soft_breakout_pct = float(
            input("Soft Breakout Percentage [0.001-0.02, default=0.005]: ") or "0.005"
        )
        proximity_threshold = float(
            input("Proximity Threshold [0.01-0.1, default=0.05]: ") or "0.05"
        )
        volume_threshold = float(
            input("Volume Threshold [1.0-3.0, default=1.2]: ") or "1.2"
        )
        lookback_days = int(
            input("Lookback Days [90-365, default=365]: ") or "365"
        )
        use_llm_input = input("Use LLM for enhanced analysis? [y/N, default=N]: ").lower()
        use_llm = use_llm_input == 'y' or use_llm_input == 'yes'
        
        # Validate inputs
        if not (0 <= universe_choice <= 5):
            universe_choice = 0
            print("Invalid universe choice. Using S&P 500.")
        
        # Initialize and run the agent
        agent = StockAnalysisAgent()
        result = await agent.run_analysis(
            universe_choice,
            soft_breakout_pct,
            proximity_threshold,
            volume_threshold,
            lookback_days,
            use_llm
        )
        
        # Print the report
        print(result["report"])
        
        # Save report to file
        today = datetime.today().strftime('%Y-%m-%d')
        agent.action.save_report_to_file(result["report"], f"reports/analysis_{today}.txt")
        
    except KeyboardInterrupt:
        print("\nOperation canceled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
