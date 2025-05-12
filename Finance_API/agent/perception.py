"""
perception.py - LLM Integration Module for Stock Analysis Agent

This module handles interactions with language models for analyzing stock data.
"""
import os
import re
import json
import asyncio
import logging
from typing import Dict, Any

from dotenv import load_dotenv
import google.generativeai as genai

logger = logging.getLogger('stock_agent.perception')

# System prompt for the LLM reasoning framework
SYSTEM_PROMPT = """You are a Stock Analysis agent with deep expertise in technical 
analysis and market screening. Follow this framework:

[VALIDATE] Check parameters: universe_choice, soft_breakout_pct, proximity_threshold, 
volume_threshold, lookback_days.
[HYPOTHESIS] Explain your reasoning about what kind of stocks this screening should identify.
[CALCULATION] Explain key calculations in plain language.
[ANALYZE] Interpret the results - what do the found patterns indicate for trading?
[VERIFY] Sanity-check outputs and highlight any anomalies.
[SUGGEST] Recommend 1-3 additional parameters or filters that might improve results.

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

class LLMPerception:
    """Class for handling LLM-based analysis of stock data"""
    
    def __init__(self):
        """Initialize the LLM interface"""
        self.model = None
        
    def load_config(self):
        """Load environment variables and configure the LLM API"""
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        
        logger.info(f"API Key loaded: {'Yes' if api_key else 'No'}")
        
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env file. Please add it.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.0-flash")
        logger.info("Gemini model configured successfully")
        return self.model
    
    async def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM analysis for the stock screening results"""
        if not self.model:
            self.load_config()
            
        if "error" in data:
            logger.info("Skipping LLM due to error in results")
            return data
        
        # Prepare prompt for the LLM
        prompt = f"""
{SYSTEM_PROMPT}

Analyze these stock screening results:
{json.dumps(data, indent=2)}
"""
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Call the LLM with timeout
        try:
            response = await asyncio.wait_for(
                self.model.generate_content_async(prompt), 
                timeout=30
            )
            
            # Process the response - handle markdown code blocks
            json_text = response.text
            json_text = re.sub(r'^```json\s*', '', json_text)
            json_text = re.sub(r'\s*```$', '', json_text)
            
            try:
                analysis = json.loads(json_text)
                
                # Ensure all required fields exist
                analysis.setdefault("reasoning", {})
                analysis.setdefault("analysis", {})
                analysis.setdefault("suggestions", {})
                
                result = {
                    "reasoning": analysis.get("reasoning", {}),
                    "analysis": analysis.get("analysis", {}),
                    "suggestions": analysis.get("suggestions", {}),
                    "result": data
                }
                return result
                
            except json.JSONDecodeError:
                # Try to extract JSON from text if parsing failed
                json_match = re.search(r'(\{.*\})', response.text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    analysis = json.loads(json_str)
                    
                    # Ensure all required fields
                    analysis.setdefault("reasoning", {})
                    analysis.setdefault("analysis", {})
                    analysis.setdefault("suggestions", {})
                    
                    return {
                        "reasoning": analysis.get("reasoning", {}),
                        "analysis": analysis.get("analysis", {}),
                        "suggestions": analysis.get("suggestions", {}),
                        "result": data
                    }
                    
                # Create minimal response on failure
                return self._create_fallback_response(data, "LLM response parsing failed")
                
        except asyncio.TimeoutError:
            return self._create_fallback_response(data, "LLM analysis timed out")
            
        except Exception as e:
            return self._create_fallback_response(data, f"LLM error: {str(e)}")
    
    def _create_fallback_response(self, data: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """Create a fallback response when LLM analysis fails"""
        return {
            "reasoning": {
                "parameter_check": error_msg,
                "strategy_summary": "Standard stock analysis strategy",
                "verification": (
                    f"Found {len(data.get('breakouts', []))} breakouts and "
                    f"{len(data.get('near_breakouts', []))} near-breakouts"
                )
            },
            "result": data
        }

    def _extract_json_from_markdown(self, text):
        """Extract JSON content from markdown code blocks or return the original text if no blocks found."""
        # Pattern to match JSON inside markdown code blocks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        
        # Try to find JSON blocks
        matches = re.findall(json_pattern, text)
        
        if matches:
            # Return the first JSON block found
            return matches[0].strip()
        else:
            # If no code blocks, return the original text
            return text.strip()

    def _parse_llm_response(self, response_text):
        """Parse LLM response text into JSON."""
        try:
            # First clean the text to extract JSON if it's in markdown format
            cleaned_text = self._extract_json_from_markdown(response_text)
            
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response text: {response_text[:500]}...")
            return None 