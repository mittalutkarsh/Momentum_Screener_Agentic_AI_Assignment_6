"""
action.py - Action Module for Stock Analysis Agent

This module handles outputs and actions based on analysis results.
"""
import logging
from typing import Dict, Any, List

logger = logging.getLogger('stock_agent.action')

class ActionHandler:
    """Class for handling actions based on stock analysis"""
    
    def __init__(self):
        """Initialize the action handler"""
        pass
    
    def generate_report(self, analysis_data: Dict[str, Any]) -> str:
        """Generate a user-friendly report of the analysis results"""
        report = []
        
        # Check for errors
        if "error" in analysis_data.get("result", {}):
            report.append("⚠️ ERROR ⚠️")
            error = analysis_data["result"]["error"]
            report.append(f"Type: {error['type']}")
            report.append(f"Description: {error['description']}")
            report.append(f"Suggestion: {error['suggestion']}")
            return "\n".join(report)
        
        # Add title
        report.append("=== STOCK ANALYSIS REPORT ===")
        
        # Add strategy summary
        if "reasoning" in analysis_data:
            report.append("\n🔍 STRATEGY SUMMARY")
            report.append(analysis_data["reasoning"].get(
                "strategy_summary", "Standard momentum analysis strategy"
            ))
            
            if "calculation_explanation" in analysis_data["reasoning"]:
                report.append("\n📊 CALCULATION METHOD")
                report.append(analysis_data["reasoning"]["calculation_explanation"])
        
        # Add market context
        if "analysis" in analysis_data:
            report.append("\n📈 MARKET CONTEXT")
            report.append(analysis_data["analysis"].get(
                "market_context", "No market context available"
            ))
            
            if "top_breakouts_analysis" in analysis_data["analysis"]:
                report.append("\n🔝 TOP FINDINGS ANALYSIS")
                report.append(analysis_data["analysis"]["top_breakouts_analysis"])
        
        # Add suggestions
        if "suggestions" in analysis_data:
            report.append("\n💡 SUGGESTIONS")
            if "parameter_adjustments" in analysis_data["suggestions"]:
                report.append("Parameter Adjustments: " + 
                              analysis_data["suggestions"]["parameter_adjustments"])
            if "additional_filters" in analysis_data["suggestions"]:
                report.append("Additional Filters: " + 
                              analysis_data["suggestions"]["additional_filters"])
        
        # Add result details
        result = analysis_data["result"]
        
        # Add breakouts section
        report.append("\n=== BREAKOUTS ===")
        if not result.get("breakouts"):
            report.append("None found")
        else:
            # Print the first 5 breakouts
            breakouts = result["breakouts"]
            report.append(f"Found {len(breakouts)} breakouts. Top 5:")
            for i, b in enumerate(breakouts[:5], 1):
                report.append(
                    f"{i}. {b['Symbol']}: ${b['Price']} "
                    f"({b['Distance_to_High_pct']}% from high, {b['Volume_Ratio']}x volume)"
                )
        
        # Add near breakouts section
        report.append("\n=== NEAR BREAKOUTS ===")
        if not result.get("near_breakouts"):
            report.append("None found")
        else:
            # Print the first 5 near breakouts
            near_breakouts = result["near_breakouts"]
            report.append(f"Found {len(near_breakouts)} near breakouts. Top 5:")
            for i, b in enumerate(near_breakouts[:5], 1):
                report.append(
                    f"{i}. {b['Symbol']}: ${b['Price']} "
                    f"({b['Distance_to_High_pct']}% from high, {b['Volume_Ratio']}x volume)"
                )
        
        # Add Excel file info
        if "excel_path" in result:
            report.append(f"\nFull results saved to Excel: {result['excel_path']}")
        
        return "\n".join(report)
    
    def print_report(self, report: str):
        """Print the report to console"""
        print(report)
    
    def save_report_to_file(self, report: str, filename: str = None) -> str:
        """Save the report to a text file"""
        import os
        from datetime import datetime
        
        os.makedirs("reports", exist_ok=True)
        
        if not filename:
            today = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"reports/stock_analysis_{today}.txt"
        
        with open(filename, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {filename}")
        return filename 