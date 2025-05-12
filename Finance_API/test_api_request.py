import requests
import json

def test_api():
    # Set up test parameters
    url = 'http://localhost:5000/api/screener/full'
    payload = {
        "universe_choice": 0,
        "soft_breakout_pct": 0.02,
        "proximity_threshold": 0.1,
        "volume_threshold": 1.0,
        "lookback_days": 90,
        "use_llm": True
    }
    
    # Send the request
    print(f"Sending request to {url} with payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(url, json=payload)
        
        # Check for success
        if response.status_code == 200:
            print(f"Success! Status code: {response.status_code}")
            
            # Parse and examine the response
            data = response.json()
            print("\nResponse data structure:")
            print(f"Keys in response: {list(data.keys())}")
            
            if 'result' in data:
                print(f"Type of result: {type(data['result'])}")
                print(f"Keys in result: {list(data['result'].keys()) if isinstance(data['result'], dict) else 'not a dict'}")
                
                if isinstance(data['result'], dict) and 'result' in data['result']:
                    inner_result = data['result']['result']
                    print(f"Keys in inner result: {list(inner_result.keys()) if isinstance(inner_result, dict) else 'not a dict'}")
                    print(f"Breakouts found: {len(inner_result.get('breakouts', []))}")
        else:
            print(f"Error! Status code: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Request failed: {str(e)}")

if __name__ == "__main__":
    test_api() 