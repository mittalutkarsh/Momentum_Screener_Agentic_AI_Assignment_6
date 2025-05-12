import re
import json

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

# Test cases
test_cases = [
    # Case 1: JSON inside markdown code block with language specifier
    '''```json
    {
      "ticker": "DEMO",
      "analysis": {
        "breakout_potential": "low",
        "volume_confirmation": false,
        "recommendation": "Hold or Monitor"
      }
    }
    ```''',
    
    # Case 2: JSON inside markdown code block without language specifier
    '''```
    {
      "ticker": "DEMO",
      "analysis": {
        "breakout_potential": "low"
      }
    }
    ```''',
    
    # Case 3: Raw JSON
    '''{
      "ticker": "DEMO",
      "analysis": {
        "breakout_potential": "low"
      }
    }'''
]

# Test each case
for i, test_case in enumerate(test_cases):
    print(f"Test case {i+1}:")
    print(f"Original: {test_case[:50]}...")
    
    cleaned = extract_json_from_markdown(test_case)
    print(f"Cleaned: {cleaned[:50]}...")
    
    try:
        parsed = json.loads(cleaned)
        print(f"Parsed successfully: {json.dumps(parsed, indent=2)[:50]}...")
    except json.JSONDecodeError as e:
        print(f"Parse error: {e}")
    
    print("\n" + "-"*50 + "\n")

print("All tests completed!") 