import React, { useState } from 'react';
import './App.css';
import ScreenerForm from './components/ScreenerForm';
import ScreenerResults from './components/ScreenerResults';
import ApiTest from './components/ApiTest';

// Define the interface for our result data structure
interface ScreenerResult {
  rawData?: any;
}

function App() {
  // Update the type here
  const [result, setResult] = useState<ScreenerResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleScreenerResult = (data: any) => {
    console.log("Got screener result:", data);
    setResult({ rawData: data });
  };

  const handleFormSubmit = async (formData: any) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Call API
      const response = await fetch('http://localhost:5000/api/screener/full', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });
      
      const data = await response.json();
      console.log("API Response:", data);
      
      // Check if data contains an error field
      if (data.error) {
        setError(data.error);
        setResult(null);
      } else {
        // Wrap the API response in an object that ScreenerResults expects
        setResult({ rawData: data });
      }
    } catch (err: any) {
      console.error("Error submitting form:", err);
      setError(err.message || "An error occurred while fetching data");
      setResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Add a test function
  const testWithSimpleData = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5000/api/test/data');
      const data = await response.json();
      console.log("Test data response:", data);
      
      // Wrap the API response in the expected structure
      setResult({ rawData: data });
    } catch (err: any) {
      console.error("Error fetching test data:", err);
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Stock Momentum Screener</h1>
        <p className="lead">Find stocks breaking out to new highs with volume confirmation</p>
      </header>
      
      <main className="container mt-4">
        <div className="row">
          <div className="col-md-6">
            <ScreenerForm onSubmitSuccess={handleFormSubmit} />
          </div>
          <div className="col-md-6">
            {result && <ScreenerResults result={result} isLoading={isLoading} />}
          </div>
        </div>
        
        <div className="row mt-5">
          <div className="col">
            <h3>API Connection Diagnostic</h3>
            <ApiTest />
          </div>
        </div>

        {/* Add a test button in your component */}
        <button 
          className="btn btn-secondary mt-3" 
          onClick={testWithSimpleData}
        >
          Test with Sample Data
        </button>
      </main>
    </div>
  );
}

export default App;