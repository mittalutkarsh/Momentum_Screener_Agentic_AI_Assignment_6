import React from 'react';
import { 
  Paper, 
  Typography, 
  Box, 
  Divider, 
  Chip, 
  Card, 
  CardContent,
  Button,
  Link,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails 
} from '@mui/material';
import { 
  DataGrid, 
  GridColDef
} from '@mui/x-data-grid';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import DownloadIcon from '@mui/icons-material/Download';
import InsightsIcon from '@mui/icons-material/Insights';
import momentumClient from './mcp_client';

// Types
interface BreakoutStock {
  Symbol: string;
  Price: number;
  "52_Week_High"?: number;
  "Distance_to_High_pct"?: number;
  Volume_Ratio: number;
}

interface ScreenerResult {
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

interface ScreenerResponse {
  success: boolean;
  timestamp: string;
  result: ScreenerResult;
}

interface ScreenerAnalysis {
  reasoning?: {
    parameter_check?: string;
    strategy_summary?: string;
    calculation_explanation?: string;
    verification?: string;
  };
  analysis?: {
    market_context?: string;
    top_breakouts_analysis?: string;
    pattern_recognition?: string;
  };
  suggestions?: {
    parameter_adjustments?: string;
    additional_filters?: string;
  };
}

interface ScreenerResultsProps {
  data: any; // The full response from the screener API
  onReset: () => void;
}

const ScreenerResults: React.FC<ScreenerResultsProps> = ({ data, onReset }) => {
  // Early return if no data
  if (!data || !data.result) {
    return (
      <Alert severity="error">
        No valid screening results available
      </Alert>
    );
  }

  const screeningResult: ScreenerResult = data.result;
  const analysis: ScreenerAnalysis = {
    reasoning: data.reasoning || {},
    analysis: data.analysis || {},
    suggestions: data.suggestions || {}
  };
  
  // Check if there are any breakouts or near breakouts
  const hasResults = 
    screeningResult.breakouts.length > 0 || 
    screeningResult.near_breakouts.length > 0;
  
  // Check if we have LLM analysis
  const hasAnalysis = 
    Object.keys(analysis.reasoning || {}).length > 0 ||
    Object.keys(analysis.analysis || {}).length > 0 ||
    Object.keys(analysis.suggestions || {}).length > 0;

  // Handle download
  const handleDownload = () => {
    const filename = screeningResult.excel_path.split('/').pop() || 'momentum_screener_results.xlsx';
    const link = momentumClient.getReportDownloadUrl(filename);
    window.open(link, '_blank');
  };

  // DataGrid columns with simpler format functions
  const columns: GridColDef[] = [
    { field: 'Symbol', headerName: 'Symbol', width: 120 },
    { 
      field: 'Price', 
      headerName: 'Price', 
      width: 120,
      // Simpler approach to formatting
      renderCell: (params) => `$${(params.value as number).toFixed(2)}`
    },
    { 
      field: '52_Week_High', 
      headerName: '52-Week High', 
      width: 150,
      renderCell: (params) => {
        const value = params.value as number | undefined;
        return value ? `$${value.toFixed(2)}` : 'N/A';
      }
    },
    { 
      field: 'Distance_to_High_pct', 
      headerName: 'Distance to High', 
      width: 160,
      renderCell: (params) => {
        const value = params.value as number | undefined;
        return value ? `${value.toFixed(2)}%` : 'N/A';
      }
    },
    { 
      field: 'Volume_Ratio', 
      headerName: 'Volume Ratio', 
      width: 150,
      renderCell: (params) => `${(params.value as number).toFixed(2)}x`
    },
  ];

  return (
    <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" component="h2">
          Momentum Screener Results
        </Typography>
        <Box>
          <Button 
            variant="outlined" 
            startIcon={<DownloadIcon />}
            onClick={handleDownload}
            sx={{ mr: 1 }}
          >
            Download Excel
          </Button>
          <Button 
            variant="outlined" 
            color="secondary"
            onClick={onReset}
          >
            New Screening
          </Button>
        </Box>
      </Box>
      
      {/* Screening Parameters */}
      {screeningResult.parameters && (
        <Box mb={3}>
          <Typography variant="subtitle1" fontWeight="bold">
            Screening Parameters
          </Typography>
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mt: 1 }}>
            <Box sx={{ minWidth: '180px' }}>
              <Typography variant="body2" color="text.secondary">Universe:</Typography>
              <Typography variant="body1">{screeningResult.parameters.universe}</Typography>
            </Box>
            
            <Box sx={{ minWidth: '180px' }}>
              <Typography variant="body2" color="text.secondary">Lookback Period:</Typography>
              <Typography variant="body1">{screeningResult.parameters.lookback_days} days</Typography>
            </Box>
            
            <Box sx={{ minWidth: '180px' }}>
              <Typography variant="body2" color="text.secondary">Breakout Threshold:</Typography>
              <Typography variant="body1">{(screeningResult.parameters.soft_breakout_pct * 100).toFixed(1)}%</Typography>
            </Box>
            
            <Box sx={{ minWidth: '180px' }}>
              <Typography variant="body2" color="text.secondary">Proximity Threshold:</Typography>
              <Typography variant="body1">{(screeningResult.parameters.proximity_threshold * 100).toFixed(1)}%</Typography>
            </Box>
            
            <Box sx={{ minWidth: '180px' }}>
              <Typography variant="body2" color="text.secondary">Volume Threshold:</Typography>
              <Typography variant="body1">{screeningResult.parameters.volume_threshold.toFixed(1)}x</Typography>
            </Box>
          </Box>
        </Box>
      )}
      
      <Divider sx={{ my: 3 }} />
      
      {!hasResults ? (
        <Alert severity="info" sx={{ mb: 3 }}>
          No breakouts or near-breakouts found with the current parameters.
          Try adjusting your parameters and run the screener again.
        </Alert>
      ) : (
        <>
          {/* Breakouts Section */}
          <Typography variant="h6" component="h3" gutterBottom>
            Breakouts {screeningResult.breakouts.length > 0 && (
              <Chip 
                label={screeningResult.breakouts.length} 
                color="primary" 
                size="small" 
                sx={{ ml: 1 }} 
              />
            )}
          </Typography>
          
          {screeningResult.breakouts.length === 0 ? (
            <Alert severity="info" sx={{ mb: 3 }}>
              No breakouts found with the current parameters.
            </Alert>
          ) : (
            <Box sx={{ height: 400, width: '100%', mb: 4 }}>
              <DataGrid
                rows={screeningResult.breakouts.map((row, index) => ({ id: index, ...row }))}
                columns={columns}
                initialState={{
                  pagination: { paginationModel: { pageSize: 10 } },
                }}
                pageSizeOptions={[5, 10, 25, 50]}
                disableRowSelectionOnClick
              />
            </Box>
          )}
          
          {/* Near Breakouts Section */}
          <Typography variant="h6" component="h3" gutterBottom>
            Near Breakouts {screeningResult.near_breakouts.length > 0 && (
              <Chip 
                label={screeningResult.near_breakouts.length} 
                color="secondary" 
                size="small" 
                sx={{ ml: 1 }} 
              />
            )}
          </Typography>
          
          {screeningResult.near_breakouts.length === 0 ? (
            <Alert severity="info" sx={{ mb: 3 }}>
              No near breakouts found with the current parameters.
            </Alert>
          ) : (
            <Box sx={{ height: 400, width: '100%', mb: 4 }}>
              <DataGrid
                rows={screeningResult.near_breakouts.map((row, index) => ({ id: index, ...row }))}
                columns={columns}
                initialState={{
                  pagination: { paginationModel: { pageSize: 10 } },
                }}
                pageSizeOptions={[5, 10, 25, 50]}
                disableRowSelectionOnClick
              />
            </Box>
          )}
        </>
      )}
      
      {/* LLM Analysis Section */}
      {hasAnalysis && (
        <Box mt={4}>
          <Divider sx={{ my: 3 }} />
          <Box display="flex" alignItems="center" mb={2}>
            <InsightsIcon sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6" component="h3">
              LLM-Enhanced Analysis
            </Typography>
          </Box>
          
          {/* Strategy Summary */}
          {analysis.reasoning?.strategy_summary && (
            <Card variant="outlined" sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  Strategy Summary
                </Typography>
                <Typography variant="body1">
                  {analysis.reasoning.strategy_summary}
                </Typography>
              </CardContent>
            </Card>
          )}
          
          {/* Market Context */}
          {analysis.analysis?.market_context && (
            <Card variant="outlined" sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                  Market Context
                </Typography>
                <Typography variant="body1">
                  {analysis.analysis.market_context}
                </Typography>
              </CardContent>
            </Card>
          )}
          
          {/* Accordions for additional details */}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography fontWeight="bold">Top Breakouts Analysis</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography>
                {analysis.analysis?.top_breakouts_analysis || "No analysis available"}
              </Typography>
            </AccordionDetails>
          </Accordion>
          
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography fontWeight="bold">Pattern Recognition</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography>
                {analysis.analysis?.pattern_recognition || "No patterns identified"}
              </Typography>
            </AccordionDetails>
          </Accordion>
          
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography fontWeight="bold">Suggestions for Improvement</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="subtitle2">Parameter Adjustments:</Typography>
              <Typography paragraph>
                {analysis.suggestions?.parameter_adjustments || "No suggestions available"}
              </Typography>
              
              <Typography variant="subtitle2">Additional Filters:</Typography>
              <Typography>
                {analysis.suggestions?.additional_filters || "No additional filters suggested"}
              </Typography>
            </AccordionDetails>
          </Accordion>
        </Box>
      )}
    </Paper>
  );
};

export default ScreenerResults;