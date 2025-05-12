import React, { useEffect, useState } from 'react';

interface BreakoutResult {
  Symbol: string;
  Price: number;
  '52_Week_High': number;
  Distance_to_High_pct: number;
  Volume_Ratio: number;
}

interface ScreenerResultProps {
  result: any;
  isLoading: boolean;
}

const ScreenerResults: React.FC<ScreenerResultProps> = ({ result, isLoading }) => {
  const [hasBreakouts, setHasBreakouts] = useState(false);
  const [hasNearBreakouts, setHasNearBreakouts] = useState(false);
  const [breakouts, setBreakouts] = useState<BreakoutResult[]>([]);
  const [nearBreakouts, setNearBreakouts] = useState<BreakoutResult[]>([]);

  useEffect(() => {
    console.log("ScreenerResults received result:", result);
    
    if (result?.rawData?.result) {
      const resultData = result.rawData.result;
      console.log("Processing result data:", resultData);
      
      // Check for arrays directly from the API response
      if (Array.isArray(resultData.breakouts)) {
        setBreakouts(resultData.breakouts);
        setHasBreakouts(resultData.breakouts.length > 0);
        console.log("Found breakouts:", resultData.breakouts);
      } else {
        console.log("No breakouts array found in data");
      }
      
      if (Array.isArray(resultData.near_breakouts)) {
        setNearBreakouts(resultData.near_breakouts);
        setHasNearBreakouts(resultData.near_breakouts.length > 0);
        console.log("Found near breakouts:", resultData.near_breakouts);
      } else {
        console.log("No near_breakouts array found in data");
      }
    } else {
      console.log("No valid result.rawData.result found");
    }
  }, [result]);

  // Show loading indicator
  if (isLoading) {
    return (
      <div className="text-center my-5">
        <div className="spinner-border" role="status">
          <span className="visually-hidden">Loading...</span>
        </div>
        <p className="mt-3">Analyzing breakouts...</p>
      </div>
    );
  }

  // No result yet
  if (!result) {
    return null;
  }

  // Handle errors or no data
  if (result.error) {
    return <div className="alert alert-danger">{result.error}</div>;
  }

  // Handle cases where the analysis was performed but no breakouts were found
  const noBreakoutsMessage = (
    <div className="alert alert-info my-3">
      No breakouts found with the current parameters. Try adjusting your parameters.
    </div>
  );

  return (
    <div className="container mt-4">
      <h2>Screener Results</h2>
      {result.rawData?.reasoning?.parameter_validation?.startsWith('Error') && (
        <div className="alert alert-warning">
          {result.rawData.reasoning.parameter_validation}
        </div>
      )}

      {/* Analysis Section */}
      <div className="card mb-4">
        <div className="card-header">
          <h5>Analysis</h5>
        </div>
        <div className="card-body">
          <div className="mb-3">
            <h6>Market Context</h6>
            <p>{result.rawData?.analysis?.market_context || "No market context available"}</p>
          </div>
          <div className="mb-3">
            <h6>Strategy Logic</h6>
            <p>{result.rawData?.reasoning?.hypothesis || "No strategy explanation available"}</p>
          </div>
          <div className="mb-3">
            <h6>Calculation Method</h6>
            <p>{result.rawData?.reasoning?.calculation_explanation || "No calculation explanation available"}</p>
          </div>
        </div>
      </div>

      {/* Breakouts Section */}
      <div className="card mb-4">
        <div className="card-header">
          <h5>Breakouts <span className="badge bg-primary">{breakouts.length}</span></h5>
        </div>
        <div className="card-body">
          {hasBreakouts ? (
            <table className="table table-striped table-bordered table-hover">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Price</th>
                  <th>52 Week High</th>
                  <th>% From High</th>
                  <th>Volume Ratio</th>
                </tr>
              </thead>
              <tbody>
                {breakouts.map((item, index) => (
                  <tr key={index}>
                    <td>{item.Symbol}</td>
                    <td>${item.Price.toFixed(2)}</td>
                    <td>${item['52_Week_High'].toFixed(2)}</td>
                    <td>{item.Distance_to_High_pct.toFixed(2)}%</td>
                    <td>{item.Volume_Ratio.toFixed(2)}x</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            noBreakoutsMessage
          )}
        </div>
      </div>

      {/* Near Breakouts Section */}
      <div className="card mb-4">
        <div className="card-header">
          <h5>Near Breakouts <span className="badge bg-secondary">{nearBreakouts.length}</span></h5>
        </div>
        <div className="card-body">
          {hasNearBreakouts ? (
            <table className="table table-striped table-bordered table-hover">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Price</th>
                  <th>52 Week High</th>
                  <th>% From High</th>
                  <th>Volume Ratio</th>
                </tr>
              </thead>
              <tbody>
                {nearBreakouts.map((item, index) => (
                  <tr key={index}>
                    <td>{item.Symbol}</td>
                    <td>${item.Price.toFixed(2)}</td>
                    <td>${item['52_Week_High'].toFixed(2)}</td>
                    <td>{item.Distance_to_High_pct.toFixed(2)}%</td>
                    <td>{item.Volume_Ratio.toFixed(2)}x</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="alert alert-info">No near breakouts found with the current parameters.</div>
          )}
        </div>
      </div>

      {/* Suggestions Section */}
      <div className="card mb-4">
        <div className="card-header">
          <h5>Suggestions</h5>
        </div>
        <div className="card-body">
          <h6>Parameter Adjustments</h6>
          <p>{result.rawData?.suggestions?.parameter_adjustments || "No suggestions available"}</p>
          
          <h6>Additional Filters</h6>
          <p>{result.rawData?.suggestions?.additional_filters || "No additional filter suggestions available"}</p>
        </div>
      </div>
    </div>
  );
};

export default ScreenerResults; 