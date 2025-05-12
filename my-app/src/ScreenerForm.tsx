import React, { useState, useEffect } from 'react';
import { 
  Button, 
  FormControl, 
  FormLabel, 
  InputLabel, 
  MenuItem, 
  Select, 
  Slider, 
  Typography, 
  Paper, 
  Box, 
  CircularProgress, 
  Switch, 
  FormControlLabel 
} from '@mui/material';
import momentumClient from './mcp_client';

// Types
interface Universe {
  id: number;
  name: string;
  description: string;
}

interface ScreenerFormProps {
  onScreeningComplete: (result: any) => void;
  onError: (error: string) => void;
}

const ScreenerForm: React.FC<ScreenerFormProps> = ({ onScreeningComplete, onError }) => {
  // Universe options
  const [universes, setUniverses] = useState<Universe[]>([]);
  const [loadingUniverses, setLoadingUniverses] = useState(true);
  
  // Form state
  const [universeChoice, setUniverseChoice] = useState<number>(0);
  const [softBreakoutPct, setSoftBreakoutPct] = useState<number>(0.005);
  const [proximityThreshold, setProximityThreshold] = useState<number>(0.05);
  const [volumeThreshold, setVolumeThreshold] = useState<number>(1.2);
  const [lookbackDays, setLookbackDays] = useState<number>(90);
  const [useLLM, setUseLLM] = useState<boolean>(false);
  const [useQuickMode, setUseQuickMode] = useState<boolean>(true);
  const [tickerCount, setTickerCount] = useState<number>(30);
  
  // UI state
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [apiStatus, setApiStatus] = useState<any>(null);

  // Load universes and API status on component mount
  useEffect(() => {
    const initialize = async () => {
      try {
        setLoadingUniverses(true);
        // Get API status
        const status = await momentumClient.getStatus();
        setApiStatus(status);
        
        // Get universe options
        const universesData = await momentumClient.getUniverses();
        setUniverses(universesData);
      } catch (error) {
        onError(`Failed to initialize: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        setLoadingUniverses(false);
      }
    };
    
    initialize();
  }, [onError]);

  // Handle form submission
  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsSubmitting(true);
    
    try {
      let result;
      
      if (useQuickMode) {
        // Run quick screener
        result = await momentumClient.runQuickScreener({
          ticker_count: tickerCount,
          lookback_days: lookbackDays,
          soft_breakout_pct: softBreakoutPct,
          proximity_threshold: proximityThreshold,
          volume_threshold: volumeThreshold
        });
      } else {
        // Run full screener
        result = await momentumClient.runFullScreener({
          universe_choice: universeChoice,
          soft_breakout_pct: softBreakoutPct,
          proximity_threshold: proximityThreshold,
          volume_threshold: volumeThreshold,
          lookback_days: lookbackDays,
          use_llm: useLLM
        });
      }
      
      onScreeningComplete(result);
    } catch (error) {
      onError(`Screening failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Format percentage value for display
  const formatPct = (value: number) => `${(value * 100).toFixed(1)}%`;

  if (loadingUniverses) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="200px">
        <CircularProgress />
        <Typography variant="body1" sx={{ ml: 2 }}>
          Loading...
        </Typography>
      </Box>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
      <Typography variant="h5" component="h2" gutterBottom>
        Momentum Screener Settings
      </Typography>
      
      {apiStatus && (
        <Box mb={3}>
          <Typography variant="body2" color="text.secondary">
            API Status: <span style={{ color: apiStatus.status === 'online' ? 'green' : 'red' }}>{apiStatus.status}</span>
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Available Modes: 
            {apiStatus.quick_mode_available && ' Quick Mode'} 
            {apiStatus.full_mode_available && ' Full Mode'}
          </Typography>
        </Box>
      )}
      
      <form onSubmit={handleSubmit}>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {/* Quick/Full Mode Toggle */}
          <Box>
            <FormControlLabel
              control={
                <Switch
                  checked={useQuickMode}
                  onChange={(e) => setUseQuickMode(e.target.checked)}
                  color="primary"
                />
              }
              label="Use Quick Mode (Faster, fewer tickers)"
            />
          </Box>
          
          {useQuickMode ? (
            // Quick Mode Options
            <Box sx={{ width: '100%', maxWidth: 500 }}>
              <FormControl fullWidth>
                <FormLabel>Number of Tickers</FormLabel>
                <Slider
                  value={tickerCount}
                  onChange={(_, value) => setTickerCount(value as number)}
                  min={5}
                  max={100}
                  step={5}
                  marks={[
                    { value: 5, label: '5' },
                    { value: 30, label: '30' },
                    { value: 50, label: '50' },
                    { value: 100, label: '100' }
                  ]}
                  valueLabelDisplay="auto"
                />
              </FormControl>
            </Box>
          ) : (
            // Full Mode Options
            <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 2 }}>
              <Box sx={{ flex: 1 }}>
                <FormControl fullWidth>
                  <InputLabel>Stock Universe</InputLabel>
                  <Select
                    value={universeChoice}
                    onChange={(e) => setUniverseChoice(Number(e.target.value))}
                    label="Stock Universe"
                  >
                    {universes.map((universe) => (
                      <MenuItem key={universe.id} value={universe.id}>
                        {universe.name} - {universe.description}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>
              
              <Box sx={{ flex: 1, display: 'flex', alignItems: 'center' }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={useLLM}
                      onChange={(e) => setUseLLM(e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Use LLM for enhanced analysis"
                />
              </Box>
            </Box>
          )}
          
          {/* Common Parameters */}
          <Box sx={{ width: '100%' }}>
            <FormControl fullWidth>
              <FormLabel>Lookback Period (Days)</FormLabel>
              <Slider
                value={lookbackDays}
                onChange={(_, value) => setLookbackDays(value as number)}
                min={30}
                max={365}
                step={5}
                marks={[
                  { value: 30, label: '30' },
                  { value: 90, label: '90' },
                  { value: 180, label: '180' },
                  { value: 365, label: '365' }
                ]}
                valueLabelDisplay="auto"
              />
            </FormControl>
          </Box>
          
          <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 2 }}>
            <Box sx={{ flex: 1 }}>
              <FormControl fullWidth>
                <FormLabel>Soft Breakout Threshold</FormLabel>
                <Slider
                  value={softBreakoutPct}
                  onChange={(_, value) => setSoftBreakoutPct(value as number)}
                  min={0.001}
                  max={0.02}
                  step={0.001}
                  marks={[
                    { value: 0.001, label: '0.1%' },
                    { value: 0.005, label: '0.5%' },
                    { value: 0.01, label: '1%' },
                    { value: 0.02, label: '2%' }
                  ]}
                  valueLabelDisplay="auto"
                  valueLabelFormat={formatPct}
                />
              </FormControl>
            </Box>
            
            <Box sx={{ flex: 1 }}>
              <FormControl fullWidth>
                <FormLabel>Proximity Threshold</FormLabel>
                <Slider
                  value={proximityThreshold}
                  onChange={(_, value) => setProximityThreshold(value as number)}
                  min={0.01}
                  max={0.1}
                  step={0.01}
                  marks={[
                    { value: 0.01, label: '1%' },
                    { value: 0.05, label: '5%' },
                    { value: 0.1, label: '10%' }
                  ]}
                  valueLabelDisplay="auto"
                  valueLabelFormat={formatPct}
                />
              </FormControl>
            </Box>
            
            <Box sx={{ flex: 1 }}>
              <FormControl fullWidth>
                <FormLabel>Volume Threshold</FormLabel>
                <Slider
                  value={volumeThreshold}
                  onChange={(_, value) => setVolumeThreshold(value as number)}
                  min={1.0}
                  max={3.0}
                  step={0.1}
                  marks={[
                    { value: 1.0, label: '1.0x' },
                    { value: 1.5, label: '1.5x' },
                    { value: 2.0, label: '2.0x' },
                    { value: 3.0, label: '3.0x' }
                  ]}
                  valueLabelDisplay="auto"
                  valueLabelFormat={(value) => `${value}x`}
                />
              </FormControl>
            </Box>
          </Box>
          
          <Box sx={{ mt: 2 }}>
            <Button 
              type="submit" 
              variant="contained" 
              color="primary" 
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <>
                  <CircularProgress size={24} sx={{ mr: 1 }} />
                  Running Screener...
                </>
              ) : (
                'Run Momentum Screener'
              )}
            </Button>
          </Box>
        </Box>
      </form>
    </Paper>
  );
};

export default ScreenerForm;