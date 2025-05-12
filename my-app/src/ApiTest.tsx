import React, { useEffect, useState } from 'react';

function ApiTest() {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  // Try different API endpoints
  const apiEndpoints = [
    'http://192.168.1.79:5000/api/status',
    'http://localhost:5000/api/status',
    '/api/status'  // Relative URL (if you set up a proxy)
  ];
  
  useEffect(() => {
    // Try each endpoint until one works
    const tryEndpoints = async () => {
      for (const endpoint of apiEndpoints) {
        try {
          console.log(`Trying to connect to: ${endpoint}`);
          const response = await fetch(endpoint);
          if (response.ok) {
            const data = await response.json();
            setStatus(data);
            setLoading(false);
            console.log(`Successfully connected to: ${endpoint}`);
            return; // Exit after success
          }
        } catch (err) {
          console.error(`Failed to connect to ${endpoint}:`, err);
          // Continue to the next endpoint
        }
      }
      
      // If we get here, all endpoints failed
      setError("Failed to connect to any API endpoint. Check if your server is running.");
      setLoading(false);
    };
    
    tryEndpoints();
  }, []);
  
  return (
    <div className="api-test-container">
      <h2>API Connection Test</h2>
      {loading && <p>Trying to connect to API server...</p>}
      
      {error && (
        <div className="error-container">
          <p style={{color: 'red'}}>Error connecting to API: {error}</p>
          <p>Make sure your server is running at http://192.168.1.79:5000</p>
          <p>Tried these endpoints: {apiEndpoints.join(', ')}</p>
        </div>
      )}
      
      {status && (
        <div className="success-container">
          <p style={{color: 'green'}}>✅ Connected successfully!</p>
          <h3>API Response:</h3>
          <pre>{JSON.stringify(status, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default ApiTest;