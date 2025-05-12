import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import re

def extract_json_from_markdown(text):
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

def test_llm():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("API key not found! Check your .env file.")
        return
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    
    # Simple prompt to test LLM
    prompt = """
    Analyze the following fictional stock data:
    
    Stock: DEMO
    Current Price: $150
    52-Week High: $155
    Volume Ratio: 1.8
    
    IMPORTANT: Return your response as plain JSON with no code backticks or markdown formatting.
    
    Provide an analysis with these keys:
    {
        "ticker": "DEMO",
        "analysis": {
            "breakout_potential": "high/medium/low",
            "volume_confirmation": true/false,
            "recommendation": "string"
        }
    }
    """
    
    # Run the LLM
    try:
        response = model.generate_content(prompt)
        print("Raw response:", response.text)
        
        # Extract JSON from markdown if needed
        cleaned_text = extract_json_from_markdown(response.text)
        print("\nCleaned text:", cleaned_text)
        
        # Try to parse as JSON
        try:
            result = json.loads(cleaned_text)
            print("\nParsed JSON successfully:")
            print(json.dumps(result, indent=2))
        except json.JSONDecodeError as e:
            print(f"\nCould not parse response as JSON: {e}")
    
    except Exception as e:
        print(f"Error calling LLM: {str(e)}")

if __name__ == "__main__":
    test_llm() 