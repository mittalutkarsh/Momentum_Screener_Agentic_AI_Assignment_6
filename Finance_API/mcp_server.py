# Add this near the top of your file, after the existing imports
from flask import Flask, request, jsonify, send_from_directory
import logging
import time
import asyncio

#!/usr/bin/env python3
"""
mcp_server.py - Momentum Screener API Server

This Flask server provides a REST API interface to the momentum screener.
It allows client applications to run momentum screening with custom parameters
and retrieve the results.
"""
import os
import json
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import the screener modules
try:
    from momentum_screener_quick import quick_momentum_screener
    QUICK_MODE_AVAILABLE = True
except ImportError:
    QUICK_MODE_AVAILABLE = False
    print("Quick mode not available. Place momentum_screener_quick.py in the same directory.")

try:
    from momentum_screener_llm import momentum_screener, run_momentum_screener
    FULL_MODE_AVAILABLE = True
except ImportError:
    FULL_MODE_AVAILABLE = False
    print("Full mode not available. Place momentum_screener_llm.py in the same directory.")

# Import the agent modules
from agent.perception import LLMPerception
from agent.memory import DataMemory
from agent.decision import StockAnalyzer
from agent.action import ActionHandler

# Create a global instance of the agent
from main import StockAnalysisAgent
stock_agent = StockAnalysisAgent()

# Initialize Flask app
# app = Flask(__name__, static_folder='../client/build')
app = Flask(__name__) 
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],  # Allow all origins for testing
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})  # Enable CORS for all routes

# Add to the top of mcp_server.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/')
def root():
    return """
    <html>
    <head><title>Momentum Screener API</title></head>
    <body>
        <h1>Momentum Screener API</h1>
        <p>API is running! Try these endpoints:</p>
        <ul>
            <li><a href="/api/status">/api/status</a></li>
            <li><a href="/api/universes">/api/universes</a></li>
        </ul>
    </body>
    </html>
    """

# Serve React static files
# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def serve(path):
#     if path != "" and os.path.exists(app.static_folder + '/' + path):
#         return send_from_directory(app.static_folder, path)
#     else:
#         return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Check which screener modes are available"""
    return jsonify({
        'status': 'online',
        'quick_mode_available': QUICK_MODE_AVAILABLE,
        'full_mode_available': FULL_MODE_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/universes', methods=['GET'])
def get_universes():
    """Get available stock universes"""
    universes = [
        {'id': 0, 'name': 'S&P 500', 'description': 'Standard & Poor\'s 500 Index'},
        {'id': 1, 'name': 'S&P 1500', 'description': 'Combines the S&P 500, S&P MidCap 400, and S&P SmallCap 600'},
        {'id': 2, 'name': 'Russell 1000', 'description': 'Large-cap US stocks (CSV required)'},
        {'id': 3, 'name': 'Russell 3000', 'description': 'Top 3000 US stocks (CSV required)'},
        {'id': 4, 'name': 'TSX Composite', 'description': 'Toronto Stock Exchange Composite Index'},
        {'id': 5, 'name': 'Custom', 'description': 'Custom universe from text file'}
    ]
    return jsonify(universes)

@app.route('/api/screener/quick', methods=['POST'])
def run_quick_screener():
    """Run the quick version of the momentum screener"""
    if not QUICK_MODE_AVAILABLE:
        return jsonify({
            'error': 'Quick mode not available',
            'message': 'momentum_screener_quick.py is not available'
        }), 500
    
    # Get parameters from request
    data = request.json
    ticker_count = int(data.get('ticker_count', 30))
    lookback_days = int(data.get('lookback_days', 90))
    soft_breakout_pct = float(data.get('soft_breakout_pct', 0.005))
    proximity_threshold = float(data.get('proximity_threshold', 0.05))
    volume_threshold = float(data.get('volume_threshold', 1.2))
    
    # Run the screener
    try:
        result = quick_momentum_screener(
            ticker_count=ticker_count,
            lookback_days=lookback_days,
            soft_breakout_pct=soft_breakout_pct,
            proximity_threshold=proximity_threshold,
            volume_threshold=volume_threshold
        )
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'result': result
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error running quick screener'
        }), 500

@app.route('/api/screener/full', methods=['POST'])
def run_full_screener():
    """Run the full version of the momentum screener"""
    if not FULL_MODE_AVAILABLE:
        return jsonify({
            "error": "Full mode not available"
        }), 400
    
    # Get parameters from request
    data = request.json
    universe_choice = int(data.get('universe_choice', 0))
    soft_breakout_pct = float(data.get('soft_breakout_pct', 0.005))
    proximity_threshold = float(data.get('proximity_threshold', 0.05))
    volume_threshold = float(data.get('volume_threshold', 1.2))
    lookback_days = int(data.get('lookback_days', 365))
    use_llm = bool(data.get('use_llm', False))
    data_source = data.get('data_source', 'yfinance')
    use_offline_mode = bool(data.get('use_offline_mode', False))
    
    logger.info(f"Running full screener with parameters: {data}")
    
    try:
        # Run the screener
        result = asyncio.run(run_momentum_screener(
            universe_choice=universe_choice,
            soft_breakout_pct=soft_breakout_pct,
            proximity_threshold=proximity_threshold,
            volume_threshold=volume_threshold,
            lookback_days=lookback_days,
            use_llm=use_llm,
            data_source=data_source,
            use_offline_mode=use_offline_mode
        ))
        
        # Ensure near_breakouts and breakouts arrays are present for frontend compatibility
        if 'result' in result and result['result'] is not None:
            if 'breakouts' not in result['result']:
                result['result']['breakouts'] = []
            if 'near_breakouts' not in result['result']:
                result['result']['near_breakouts'] = []
        else:
            # Ensure result has a fallback structure if needed
            if 'result' not in result or result['result'] is None:
                result['result'] = {
                    'breakouts': [],
                    'near_breakouts': [],
                    'universe_name': 'Unknown',
                    'lookback_days': lookback_days,
                    'soft_breakout_pct': soft_breakout_pct,
                    'proximity_threshold': proximity_threshold,
                    'volume_threshold': volume_threshold
                }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error running screener: {str(e)}")
        return jsonify({
            "error": f"Error running screener: {str(e)}"
        }), 500

@app.route('/api/screener/agent', methods=['POST'])
def run_agent_screener():
    """Run the agent-based stock analysis"""
    data = request.json
    # Extract your parameters here, e.g.:
    universe_choice     = int(data.get('universe_choice', 0))
    soft_breakout_pct   = float(data.get('soft_breakout_pct', 0.005))
    proximity_threshold = float(data.get('proximity_threshold', 0.05))
    volume_threshold    = float(data.get('volume_threshold', 1.2))
    lookback_days       = int(data.get('lookback_days', 365))
    use_llm             = bool(data.get('use_llm', False))

    try:
        import asyncio
        result = asyncio.run(stock_agent.run_analysis(
            universe_choice=universe_choice,
            soft_breakout_pct=soft_breakout_pct,
            proximity_threshold=proximity_threshold,
            volume_threshold=volume_threshold,
            lookback_days=lookback_days,
            use_llm=use_llm
        ))
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'result': result.get('data', {})\
                                .get('result', {'breakouts': [], 'near_breakouts': []})
        })
    except Exception as e:
        # now we actually have a statement in the except-block
        return jsonify({
            'error': str(e),
            'message': 'Error running agent screener',
            'result': {'breakouts': [], 'near_breakouts': []}
        }), 500

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get a list of available reports"""
    try:
        reports = []
        if os.path.exists('outputs'):
            for file in os.listdir('outputs'):
                if file.endswith('.xlsx'):
                    report_path = os.path.join('outputs', file)
                    report_time = datetime.fromtimestamp(os.path.getmtime(report_path))
                    reports.append({
                        'filename': file,
                        'path': report_path,
                        'created': report_time.isoformat(),
                        'size': os.path.getsize(report_path)
                    })
        return jsonify(reports)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error retrieving reports'
        }), 500

@app.route('/api/reports/<path:filename>', methods=['GET'])
def download_report(filename):
    """Download a specific report file"""
    try:
        directory = os.path.abspath("outputs")
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error downloading report'
        }), 500

@app.route('/api/test_results', methods=['GET'])
def get_test_results():
    """Return test results for debugging"""
    # Load the test data we created with debug_with_test_data.py
    try:
        with open("debug_output.json", "r") as f:
            test_data = json.load(f)
            
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'result': test_data
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Error loading test data',
            'result': {'breakouts': [], 'near_breakouts': []}
        }), 500

@app.route('/api/debug/llm_check', methods=['GET'])
def llm_check():
    """Check if the LLM API is working properly"""
    try:
        from momentum_screener_llm import load_llm_config
        import google.generativeai as genai
        
        # Try to load the config
        load_llm_config()
        
        # Try a simple model call
        result = {
            "api_key_set": bool(os.getenv("GEMINI_API_KEY")),
            "config_loaded": True,
            "test_call": None,
            "error": None
        }
        
        try:
            # Attempt a simple model call
            model = genai.GenerativeModel("models/gemini-1.5-flash")
            response = model.generate_content("Hello, this is a test from the momentum screener app.")
            result["test_call"] = "success" if response else "no response"
        except Exception as e:
            result["test_call"] = "failed"
            result["error"] = str(e)
            
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "api_key_set": False,
            "config_loaded": False,
            "test_call": None,
            "error": str(e)
        })

@app.route('/api/debug/llm_analyze', methods=['POST'])
def debug_llm_analyze():
    """Debug endpoint to test LLM analysis directly"""
    try:
        # Get simple test data
        data = request.get_json() or {}
        test_text = data.get('text', 'Test analysis of AAPL stock with bullish momentum')
        
        from agent.perception import LLMPerception
        llm = LLMPerception()
        
        # Run a simple test analysis
        result = llm.analyze({
            'test': True,
            'prompt': test_text,
            'parameters': {
                'universe': 'Test',
                'soft_breakout_pct': 0.02,
                'proximity_threshold': 0.1,
                'volume_threshold': 1.0,
                'lookback_days': 90
            }
        })
        
        return jsonify({
            'success': True,
            'message': 'LLM analysis completed',
            'result': result
        })
    except Exception as e:
        logger.exception(f"Error in debug_llm_analyze: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Error running LLM analysis',
        }), 500

@app.route('/api/debug/structure', methods=['GET'])
def debug_structure():
    """Return a simple data structure for testing"""
    sample_data = {
        'breakouts': [
            {'Symbol': 'AAPL', 'Price': 150.0, 'Distance_to_High_pct': 2.5, 'Volume_Ratio': 1.5},
            {'Symbol': 'MSFT', 'Price': 280.0, 'Distance_to_High_pct': 1.8, 'Volume_Ratio': 1.3}
        ],
        'near_breakouts': [
            {'Symbol': 'GOOGL', 'Price': 120.0, 'Distance_to_High_pct': 4.2, 'Volume_Ratio': 1.1}
        ]
    }
    
    return jsonify({
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'result': sample_data
    })

@app.route('/api/ping', methods=['GET'])
def ping():
    """Simple endpoint to test connection"""
    return jsonify({
        'success': True,
        'message': 'API is connected!',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test/data', methods=['GET'])
def test_data():
    """Return test data for frontend development"""
    return jsonify({
        "reasoning": {
            "parameter_validation": "Valid parameters",
            "hypothesis": "Test data for frontend development"
        },
        "result": {
            "breakouts": [
                {"Symbol": "AAPL", "Price": 150.0, "52_Week_High": 155.0, "Distance_to_High_pct": 3.2, "Volume_Ratio": 1.5},
                {"Symbol": "MSFT", "Price": 280.0, "52_Week_High": 290.0, "Distance_to_High_pct": 3.4, "Volume_Ratio": 1.2}
            ],
            "near_breakouts": [
                {"Symbol": "GOOGL", "Price": 120.0, "52_Week_High": 130.0, "Distance_to_High_pct": 7.7, "Volume_Ratio": 1.1}
            ],
            "universe_name": "Test Data",
            "universe_choice": 0
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


# Add this right after the CORS(app) line
