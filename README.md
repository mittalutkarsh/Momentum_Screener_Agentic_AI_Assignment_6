# Momentum Stock Screener with AI Analysis

A full-stack application that identifies momentum breakouts and stocks approaching new 52-week highs with volume confirmation, enhanced with AI analysis capabilities.

## Table of Contents

- [Overview](#overview)  
- [Key Features](#key-features)  
- [Project Structure](#project-structure)  
- [Core Algorithm](#core-algorithm)  
- [LLM Integration](#llm-integration)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
- [API Reference](#api-reference)  
- [Frontend Interface](#frontend-interface)  
- [Performance Notes](#performance-notes)  
- [License](#license)  
- [Contributing](#contributing)  

## Overview

This application scans various stock universes (S&P 500, S&P 1500, Russell indices, etc.) to identify stocks that are breaking out to new highs with strong volume. It uses a momentum-based approach to find potential trading opportunities based on proximity to 52-week highs and volume surges, with AI-enhanced analysis.

## Key Features

- **Multiple Stock Universes**: S&P 500, S&P 1500, Russell 1000/3000, TSX, custom lists
- **Robust Data Handling**: Error-resilient data downloading with retry mechanisms and batch processing
- **Technical Analysis Engine**: Breakout detection based on proximity to 52-week highs and volume confirmation
- **AI-Enhanced Analysis**: Integration with Google's Gemini API for intelligent interpretation
- **REST API**: Flask-based API server for integration with any frontend
- **React Frontend**: Modern UI for interacting with the screener
- **Detailed Reporting**: Excel export with comprehensive analysis

## Project Structure

```
├── Finance_API/               # Backend Python code
│   ├── agent/                 # Modular agent architecture
│   │   ├── perception.py      # LLM analysis module
│   │   ├── memory.py          # Data storage and retrieval
│   │   ├── decision.py        # Technical analysis algorithms
│   │   └── action.py          # Report generation
│   ├── data_sources/          # Stock data providers
│   │   ├── alphavantage.py    # Alpha Vantage integration
│   │   ├── finnhub.py         # Finnhub integration
│   │   └── iex.py             # IEX Cloud integration
│   ├── momentum_screener_llm.py  # Main screener implementation
│   ├── mcp_server.py          # Flask API server
│   └── main.py                # CLI application entry point
├── my-app/                    # React frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── App.tsx            # Main application component
│   │   └── mcp_client.ts      # API client functions
└── requirements.txt           # Python dependencies
```

## Core Algorithm

1. **Average Volume Analysis**  
   - Calculate 50-day average volume for each stock
   - Identify stocks with abnormally high current volume relative to their average

2. **Rolling High Calculation**  
   - Determine the 52-week (or custom lookback period) rolling high for each stock
   - Focus on the highest price in the window to identify potential breakout levels

3. **Volume Ratio Calculation**  
   - Compare today's volume to the 50-day average volume
   - Higher ratios indicate stronger interest in the stock

4. **Proximity Calculation**  
   - Measure how close each stock is to its 52-week high
   - Smaller proximity values indicate the stock is closer to breaking out to new highs

5. **Breakout Classification**  
   - **Breakouts**: Stocks within the specified percentage of their 52-week high **and** showing volume surge above threshold
   - **Near Breakouts**: Stocks approaching their 52-week high but not yet breaking out or lacking volume confirmation

## LLM Integration

The screener integrates Google's Gemini 2.0 Flash model for enhanced analysis of screening results.

### LLM Prompt Framework

A structured system prompt guides the AI analysis through specific steps:

- **[VALIDATE]** Check parameters and input data quality
- **[HYPOTHESIS]** Explain expected stock characteristics based on parameters
- **[CALCULATION]** Explain key calculations in plain language
- **[ANALYZE]** Interpret the results—what do the found breakouts indicate
- **[VERIFY]** Sanity-check outputs and highlight anomalies
- **[SUGGEST]** Recommend improvements to the screening process

## Installation

### Backend Setup

1. Clone the repository:
   ```
   git clone https://github.com/mittalutkarsh/Momentum_Screener_Agentic_AI_Assignment_6.git
   cd Momentum_Screener_Agentic_AI_Assignment_6
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the Finance_API directory:
   ```
   GEMINI_API_KEY=your_gemini_api_key
   FINNHUB_API_KEY=your_finnhub_api_key  # Optional
   ALPHA_VANTAGE_API_KEY=your_av_api_key  # Optional
   IEX_API_KEY=your_iex_api_key  # Optional
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd my-app
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Install Bootstrap (if not included in package.json):
   ```
   npm install react-bootstrap bootstrap
   ```

## Configuration

The application can be configured through environment variables and command-line parameters. Key configuration options:

- **Stock Universe**: Choose from S&P 500, S&P 1500, Russell 1000/3000, TSX, or custom lists
- **Breakout Parameters**: Customize proximity threshold, volume threshold, and lookback period
- **Data Source**: Select from available data providers (Yahoo Finance, Finnhub, Alpha Vantage, IEX)

## Usage

### Running the Backend API

```
cd Finance_API
python mcp_server.py
```

The server will start on http://localhost:5000 by default.

### Running the Frontend

```
cd my-app
npm start
```

The frontend will be available at http://localhost:3000.

### Using the CLI Directly

```
cd Finance_API
python main.py
```

This will launch the CLI interface for the stock analyzer.

## API Reference

### Main Endpoints

- `GET /api/status` - Check which screener modes are available
- `GET /api/universes` - Get available stock universes
- `POST /api/screener/full` - Run the full screener with LLM analysis
- `POST /api/screener/quick` - Run a faster version without LLM analysis
- `POST /api/screener/agent` - Run the agent-based stock analysis
- `GET /api/test/data` - Get sample test data for frontend development

### Request Format

Example request to `/api/screener/full`:

```json
{
  "universe_choice": 0,
  "soft_breakout_pct": 0.005,
  "proximity_threshold": 0.05,
  "volume_threshold": 1.2,
  "lookback_days": 365,
  "use_llm": true
}
```

## Frontend Interface

The React frontend provides an intuitive interface for interacting with the momentum screener:

- **Form Component**: Configure screening parameters
- **Results Component**: View breakouts and near-breakouts with key statistics
- **Analysis Panel**: Read the AI-enhanced analysis and suggestions

## Performance Notes

- Data downloading is batched to avoid timeouts and rate limits
- Browser fingerprinting is used for Yahoo Finance to improve reliability
- The system handles missing data gracefully
- For large universes, consider using the "quick" mode without LLM analysis

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

```mermaid
flowchart TB
  %%=== Subgraphs for logical layers ===%%
  subgraph CLI & Config
    direction TB
    U[User] --> |runs| Main[<b>momentum_screener_llm.py</b>]
    ENV[.env<br/>GEMINI_API_KEY] --- Main
  end

  subgraph Data_Fetching ["1. Data Fetching"]
    direction LR
    Main --> |calls| Fetcher[data_fetcher.py]
    Fetcher --> |yfinance / requests| RawData[Raw Price & Volume Data]
  end

  subgraph Processing ["2. Core Processing"]
    direction TB
    RawData --> VolCalc[VolumeCalculator<br/>(50-day avg)]
    RawData --> HighCalc[RollingHighCalculator<br/>(52-week high)]
    RawData --> ProxCalc[ProximityCalculator]
    RawData --> VolRatio[VolumeRatioCalculator]
    VolCalc & HighCalc & ProxCalc & VolRatio --> Screener[screener.py<br/>ScreenerEngine]
    Screener --> ScreenerResult[ScreenerResult JSON]
  end

  subgraph LLM_Analysis ["3. AI-Enhanced Analysis"]
    direction TB
    ScreenerResult --> LLMClient[llm_client.py]
    LLMClient --> |builds system prompt| Prompt[Structured Prompt]
    Prompt --> |API call| GeminiAPI[Gemini 2.0 Flash Model]
    GeminiAPI --> |returns| LLMRaw[LLM JSON Response]
    LLMRaw --> AnalysisProc[analysis.py<br/>AnalysisProcessor]
    AnalysisProc --> AIResult[Enhanced Analysis JSON]
  end

  subgraph Output ["4. Output & Export"]
    direction LR
    ScreenerResult --> Exporter[excel_exporter.py]
    AIResult --> Exporter
    Exporter --> |writes| ExcelFile[Excel Workbook]
    Main --> |prints| Console[Console Summary & Logs]
  end

  %%=== Styling ===%%
  classDef subgraphTitle fill:#efefef,stroke:#ccc,stroke-width:1px,font-weight:bold;
  class Data_Fetching,Processing,LLM_Analysis,Output subgraphTitle;