/**
 * mcp_client.ts - TypeScript client for the Momentum Screener API
 * 
 * This client provides a simple interface for interacting with the momentum screener API.
 * It handles API requests and response parsing for a React frontend application.
 */

// API base URL - use environment variable if defined, or fallback to localhost
export const API_BASE_URL = 'http://localhost:5000/api';

// Type definitions
export interface ScreenerParameters {
  universe_choice?: number;
  soft_breakout_pct?: number;
  proximity_threshold?: number;
  volume_threshold?: number;
  lookback_days?: number;
  use_llm?: boolean;
}

export interface QuickScreenerParameters {
  ticker_count?: number;
  lookback_days?: number;
  soft_breakout_pct?: number;
  proximity_threshold?: number;
  volume_threshold?: number;
}

export interface StockUniverse {
  id: number;
  name: string;
  description: string;
}

export interface BreakoutStock {
  Symbol: string;
  Price: number;
  High?: number;
  "52_Week_High"?: number;
  Distance?: number;
  "Distance_to_High_pct"?: number;
  Volume_Ratio: number;
}

export interface ScreenerResult {
  breakouts: BreakoutStock[];
  near_breakouts: BreakoutStock[];
  excel_path: string;
  parameters?: {
    universe: string;
    soft_breakout_pct: number;
    proximity_threshold: number;
    volume_threshold: number;
    lookback_days: number;
  };
}

export interface ScreenerResponse {
  success: boolean;
  timestamp: string;
  result: ScreenerResult;
}

export interface ApiError {
  error: string;
  message: string;
}

export interface Report {
  filename: string;
  path: string;
  created: string;
  size: number;
}

export interface ApiStatus {
  status: string;
  quick_mode_available: boolean;
  full_mode_available: boolean;
  timestamp: string;
}

/**
 * Momentum Screener API Client class
 */
export class MomentumScreenerClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Check API status and available modes
   */
  async getStatus(): Promise<ApiStatus> {
    const response = await fetch(`${this.baseUrl}/status`);
    
    if (!response.ok) {
      const errorData: ApiError = await response.json();
      throw new Error(errorData.message || 'Failed to get API status');
    }
    
    return response.json();
  }

  /**
   * Get available stock universes
   */
  async getUniverses(): Promise<StockUniverse[]> {
    const response = await fetch(`${this.baseUrl}/universes`);
    
    if (!response.ok) {
      const errorData: ApiError = await response.json();
      throw new Error(errorData.message || 'Failed to get universes');
    }
    
    return response.json();
  }

  /**
   * Run the quick version of the momentum screener
   */
  async runQuickScreener(params: QuickScreenerParameters): Promise<ScreenerResponse> {
    const response = await fetch(`${this.baseUrl}/screener/quick`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(params)
    });
    
    if (!response.ok) {
      const errorData: ApiError = await response.json();
      throw new Error(errorData.message || 'Failed to run quick screener');
    }
    
    return response.json();
  }

  /**
   * Run the full version of the momentum screener
   */
  async runFullScreener(params: ScreenerParameters): Promise<ScreenerResponse> {
    const response = await fetch(`${this.baseUrl}/screener/full`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(params)
    });
    
    if (!response.ok) {
      const errorData: ApiError = await response.json();
      throw new Error(errorData.message || 'Failed to run full screener');
    }
    
    const data = await response.json();
    
    // Ensure we always have breakouts and near_breakouts arrays
    if (data && data.result) {
      if (!data.result.breakouts) data.result.breakouts = [];
      if (!data.result.near_breakouts) data.result.near_breakouts = [];
    }
    
    return data;
  }

  /**
   * Get a list of available reports
   */
  async getReports(): Promise<Report[]> {
    const response = await fetch(`${this.baseUrl}/reports`);
    
    if (!response.ok) {
      const errorData: ApiError = await response.json();
      throw new Error(errorData.message || 'Failed to get reports');
    }
    
    return response.json();
  }

  /**
   * Get the download URL for a specific report
   */
  getReportDownloadUrl(filename: string): string {
    return `${this.baseUrl}/reports/${filename}`;
  }

  /**
   * Download a specific report
   */
  async downloadReport(filename: string): Promise<Blob> {
    const response = await fetch(this.getReportDownloadUrl(filename));
    
    if (!response.ok) {
      const errorData: ApiError = await response.json();
      throw new Error(errorData.message || 'Failed to download report');
    }
    
    return response.blob();
  }
}

// Default export of client instance
export default new MomentumScreenerClient();