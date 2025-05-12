import React, { useState, FormEvent } from 'react';
import { API_BASE_URL } from '../mcp_client';

// Define prop types
interface ScreenerFormProps {
  onSubmitSuccess: (data: any) => void;
  onSubmit?: (formData: any) => Promise<void>;  // Make it optional with ?
}

const ScreenerForm: React.FC<ScreenerFormProps> = ({ onSubmitSuccess }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Form state
  const [universeChoice, setUniverseChoice] = useState(0);
  const [softBreakoutPct, setSoftBreakoutPct] = useState(0.02);  // More relaxed
  const [proximityThreshold, setProximityThreshold] = useState(0.1);  // More relaxed
  const [volumeThreshold, setVolumeThreshold] = useState(1.2);
  const [lookbackDays, setLookbackDays] = useState(365);
  const [useLLM, setUseLLM] = useState(false);
  const [debugInfo, setDebugInfo] = useState<any>(null);
  const [useLocalData, setUseLocalData] = useState(false);
  const [dataSource, setDataSource] = useState("finnhub");
  const [useOfflineMode, setUseOfflineMode] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setDebugInfo(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/screener/full`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          universe_choice: universeChoice,
          soft_breakout_pct: softBreakoutPct,
          proximity_threshold: proximityThreshold,
          volume_threshold: volumeThreshold,
          lookback_days: lookbackDays,
          use_llm: useLLM,
          data_source: dataSource,
          use_offline_mode: useOfflineMode
        }),
      });
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Full response data:", data);
      
      setDebugInfo({
        rawData: data,
        resultKeys: data.result ? Object.keys(data.result) : [],
        hasBreakouts: data.result?.breakouts?.length > 0,
        hasNearBreakouts: data.result?.near_breakouts?.length > 0,
        excelPath: data.result?.excel_path
      });
      
      // Special handling for LLM results which might have a different structure
      if (useLLM && data.result && typeof data.result === 'object') {
        // If using LLM and result is nested further, extract it properly
        if (data.result.result) {
          console.log("Using LLM format result");
          // Data from LLM has nested result
          const formattedData = {
            ...data,
            result: data.result.result
          };
          onSubmitSuccess(formattedData);
        } else {
          // Normal format - use as is
          onSubmitSuccess(data);
        }
      } else {
        // Not using LLM or result already has the right structure
        onSubmitSuccess(data);
      }
    } catch (error: any) {
      console.error("Error running screener:", error);
      setError(error.message || "Failed to run screener");
    } finally {
      setIsLoading(false);
    }
  };

  const handleTestData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/test_results`);
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Test data loaded:", data);
      
      // Call the function that handles results
      onSubmitSuccess(data);
    } catch (error: any) {
      console.error("Error loading test data:", error);
      setError(error.message || "Failed to load test data");
    } finally {
      setIsLoading(false);
    }
  };

  const loadTestData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/debug/structure`);
      
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }
      
      const data = await response.json();
      console.log("Test data loaded:", data);
      
      if (data.success) {
        onSubmitSuccess(data);
      } else {
        setError(data.error || "Unknown error occurred");
      }
    } catch (err: any) {
      console.error("Error loading test data:", err);
      setError(err.message || "An error occurred while fetching test data");
    } finally {
      setIsLoading(false);
    }
  };

  const pingServer = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/ping`);
      const data = await response.json();
      alert(`Server connection: ${data.success ? 'SUCCESS' : 'FAILED'}\n${data.message}`);
    } catch (err) {
      alert(`Server connection failed: ${err}`);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="mb-4">
      <h2>Screener Parameters</h2>
      
      {error && (
        <div className="alert alert-danger">{error}</div>
      )}
      
      <div className="form-group mb-3">
        <label htmlFor="universeChoice">Stock Universe</label>
        <select 
          id="universeChoice"
          className="form-control"
          value={universeChoice}
          onChange={(e) => setUniverseChoice(Number(e.target.value))}
        >
          <option value={0}>S&P 500</option>
          <option value={1}>S&P 1500</option>
          <option value={2}>Russell 1000</option>
          <option value={3}>Russell 3000</option>
          <option value={4}>TSX Composite</option>
        </select>
      </div>
      
      <div className="form-group mb-3">
        <label htmlFor="softBreakoutPct">Breakout Percentage (%)</label>
        <input 
          type="number" 
          step="0.1"
          min="0.1" 
          max="5"
          id="softBreakoutPct"
          className="form-control"
          value={softBreakoutPct * 100}
          onChange={(e) => setSoftBreakoutPct(Number(e.target.value) / 100)}
        />
        <small className="form-text text-muted">
          Maximum distance from 52-week high to consider as a breakout (lower = stronger breakout)
        </small>
      </div>
      
      <div className="form-group mb-3">
        <label htmlFor="proximityThreshold">Proximity Threshold (%)</label>
        <input 
          type="number"
          step="1"
          min="1" 
          max="20"
          id="proximityThreshold"
          className="form-control"
          value={proximityThreshold * 100}
          onChange={(e) => setProximityThreshold(Number(e.target.value) / 100)}
        />
        <small className="form-text text-muted">
          Maximum distance from 52-week high to consider a stock "near" breakout
        </small>
      </div>
      
      <div className="form-group mb-3">
        <label htmlFor="volumeThreshold">Volume Threshold</label>
        <input 
          type="number"
          step="0.1"
          min="0.5" 
          max="5"
          id="volumeThreshold"
          className="form-control"
          value={volumeThreshold}
          onChange={(e) => setVolumeThreshold(Number(e.target.value))}
        />
        <small className="form-text text-muted">
          Minimum volume ratio compared to 20-day average
        </small>
      </div>
      
      <div className="form-group mb-3">
        <label htmlFor="lookbackDays">Lookback Period (days)</label>
        <input 
          type="number"
          step="30"
          min="30" 
          max="365"
          id="lookbackDays"
          className="form-control"
          value={lookbackDays}
          onChange={(e) => setLookbackDays(Number(e.target.value))}
        />
      </div>

      <div className="form-check mb-3">
        <input
          type="checkbox"
          className="form-check-input"
          id="useLLM"
          checked={useLLM}
          onChange={(e) => setUseLLM(e.target.checked)}
        />
        <label className="form-check-label" htmlFor="useLLM">
          Use AI analysis
        </label>
        <small className="form-text text-muted d-block">
          Enhances results with AI reasoning (slower but more insightful)
        </small>
      </div>
      
      <div className="form-group mb-3">
        <div className="form-check">
          <input
            className="form-check-input"
            type="checkbox"
            id="useLocalData"
            checked={useLocalData}
            onChange={(e) => setUseLocalData(e.target.checked)}
          />
          <label className="form-check-label" htmlFor="useLocalData">
            Use locally cached data (faster, works offline)
          </label>
        </div>
      </div>
      
      <div className="form-group mb-3">
        <label htmlFor="dataSource">Data Source</label>
        <select
          className="form-select"
          id="dataSource"
          value={dataSource}
          onChange={(e) => setDataSource(e.target.value)}
        >
          <option value="finnhub">Finnhub (recommended)</option>
          <option value="alphavantage">Alpha Vantage</option>
          <option value="yfinance">Yahoo Finance (may hit rate limits)</option>
        </select>
        <small className="form-text text-muted">
          Finnhub recommended for better reliability.
        </small>
      </div>
      
      <div className="mb-3 form-check">
        <input
          type="checkbox"
          className="form-check-input"
          id="useOfflineMode"
          checked={useOfflineMode}
          onChange={(e) => setUseOfflineMode(e.target.checked)}
        />
        <label className="form-check-label" htmlFor="useOfflineMode">
          Offline Mode
        </label>
        <small className="form-text text-muted d-block">
          Use cached data or samples (works without internet)
        </small>
      </div>
      
      <div className="form-group">
        <button 
          type="button" 
          className="btn btn-info" 
          onClick={handleTestData}
          style={{ marginRight: '10px' }}
          disabled={isLoading}
        >
          Use Test Data
        </button>
        <button 
          type="button" 
          className="btn btn-secondary me-2" 
          onClick={loadTestData}
          disabled={isLoading}
        >
          Load Test Data
        </button>
        <button 
          type="button" 
          className="btn btn-outline-secondary me-2" 
          onClick={pingServer}
        >
          Test Connection
        </button>
        <button 
          type="submit" 
          className="btn btn-primary"
          disabled={isLoading}
        >
          {isLoading ? 'Running...' : 'Run Screener'}
        </button>
      </div>

      {debugInfo && (
        <div className="debug-info p-3 mb-3 bg-light border">
          <h5>Debug Info:</h5>
          <pre style={{fontSize: '0.8rem', maxHeight: '200px', overflow: 'auto'}}>
            {JSON.stringify(debugInfo, null, 2)}
          </pre>
        </div>
      )}
    </form>
  );
};

export default ScreenerForm; 