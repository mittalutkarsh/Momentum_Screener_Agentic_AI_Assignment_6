import React, { useState, useEffect } from 'react';
import { API_BASE_URL } from '../mcp_client';

const ApiTest: React.FC = () => {
  const [status, setStatus] = useState<string>('Loading...');
  const [apiStatus, setApiStatus] = useState<any>(null);
  const [llmStatus, setLlmStatus] = useState<string>('Not tested');
  const [llmDetails, setLlmDetails] = useState<any>(null);

  useEffect(() => {
    const testConnection = async () => {
      try {
        setStatus('Testing connection...');
        const response = await fetch(`${API_BASE_URL}/status`);
        
        if (!response.ok) {
          throw new Error(`Server responded with status: ${response.status}`);
        }
        
        const data = await response.json();
        setApiStatus(data);
        setStatus('Connection successful');
      } catch (error) {
        console.error('API connection test failed:', error);
        setStatus(`Connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    };

    testConnection();
  }, []);

  const testLLM = async () => {
    try {
      setLlmStatus('Testing LLM API...');
      const response = await fetch(`${API_BASE_URL}/debug/llm_check`);
      
      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }
      
      const data = await response.json();
      setLlmDetails(data);
      
      if (data.test_call === "success") {
        setLlmStatus('LLM API working correctly');
      } else if (data.error) {
        setLlmStatus(`LLM API error: ${data.error}`);
      } else {
        setLlmStatus('LLM API test inconclusive');
      }
    } catch (error) {
      console.error('LLM API test failed:', error);
      setLlmStatus(`LLM test failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  return (
    <div className="api-test">
      <div className="card">
        <div className="card-header">
          API Connection Status
        </div>
        <div className="card-body">
          <h5 className="card-title">Status: {status}</h5>
          
          {apiStatus && (
            <div>
              <p>Available modes:</p>
              <ul>
                <li>
                  Quick Mode: 
                  <span className={apiStatus.quick_mode_available ? 'text-success' : 'text-danger'}>
                    {apiStatus.quick_mode_available ? ' Available' : ' Unavailable'}
                  </span>
                </li>
                <li>
                  Full Mode: 
                  <span className={apiStatus.full_mode_available ? 'text-success' : 'text-danger'}>
                    {apiStatus.full_mode_available ? ' Available' : ' Unavailable'}
                  </span>
                </li>
              </ul>
            </div>
          )}
        </div>
      </div>
      <div className="mt-3">
        <button 
          className="btn btn-secondary" 
          onClick={testLLM}
        >
          Test LLM API
        </button>
        <div className="mt-2">
          <strong>LLM Status:</strong> {llmStatus}
          {llmDetails && (
            <pre className="mt-2 bg-light p-2">
              {JSON.stringify(llmDetails, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
};

export default ApiTest; 