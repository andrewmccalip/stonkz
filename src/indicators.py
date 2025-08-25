"""Technical indicators calculation module"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class TechnicalIndicators:
    """Calculate various technical indicators for stock data"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize with configuration parameters
        
        Args:
            config: Dictionary with indicator parameters
        """
        self.config = config or {}
        
        # Default parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.sma_periods = self.config.get('sma_periods', [20, 50])
        self.ema_periods = self.config.get('ema_periods', [12, 26])
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)
        self.stoch_periods = self.config.get('stoch_periods', (14, 3, 3))
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and add them to the dataframe
        
        Args:
            df: DataFrame with OHLCV data (normalized or regular)
            
        Returns:
            DataFrame with all indicators added
        """
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Determine if we're working with normalized data
        if 'close_norm' in df.columns:
            close_col = 'close_norm'
            high_col = 'high_norm'
            low_col = 'low_norm'
            open_col = 'open_norm'
        else:
            close_col = 'close'
            high_col = 'high'
            low_col = 'low'
            open_col = 'open'
        
        # Calculate each indicator
        result_df = self._add_rsi(result_df, close_col)
        result_df = self._add_sma(result_df, close_col)
        result_df = self._add_ema(result_df, close_col)
        result_df = self._add_macd(result_df, close_col)
        result_df = self._add_bollinger_bands(result_df, close_col)
        result_df = self._add_obv(result_df, close_col)
        result_df = self._add_vwap(result_df, high_col, low_col, close_col)
        result_df = self._add_stochastic(result_df, high_col, low_col, close_col)
        
        return result_df
    
    def _add_rsi(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """Add RSI (Relative Strength Index)"""
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_sma(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """Add Simple Moving Averages"""
        for period in self.sma_periods:
            df[f'sma_{period}'] = df[close_col].rolling(window=period).mean()
        return df
    
    def _add_ema(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """Add Exponential Moving Averages"""
        for period in self.ema_periods:
            df[f'ema_{period}'] = df[close_col].ewm(span=period, adjust=False).mean()
        return df
    
    def _add_macd(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)"""
        fast = self.ema_periods[0] if len(self.ema_periods) >= 1 else 12
        slow = self.ema_periods[1] if len(self.ema_periods) >= 2 else 26
        signal = 9
        
        exp1 = df[close_col].ewm(span=fast, adjust=False).mean()
        exp2 = df[close_col].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """Add Bollinger Bands"""
        sma = df[close_col].rolling(window=self.bb_period).mean()
        std = df[close_col].rolling(window=self.bb_period).std()
        
        df['bb_upper'] = sma + (std * self.bb_std)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std * self.bb_std)
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent'] = (df[close_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def _add_obv(self, df: pd.DataFrame, close_col: str) -> pd.DataFrame:
        """Add On-Balance Volume"""
        # Calculate price direction
        price_diff = df[close_col].diff()
        
        # Calculate OBV
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0] if price_diff.iloc[0] > 0 else -df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if price_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif price_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        df['obv'] = obv
        return df
    
    def _add_vwap(self, df: pd.DataFrame, high_col: str, low_col: str, close_col: str) -> pd.DataFrame:
        """Add Volume Weighted Average Price"""
        # VWAP resets daily, so we need to group by date
        df_copy = df.copy()
        df_copy['date'] = df_copy.index.date
        
        vwap_values = []
        for date, group in df_copy.groupby('date'):
            # Calculate typical price
            typical_price = (group[high_col] + group[low_col] + group[close_col]) / 3
            
            # Calculate VWAP for the day
            cumulative_tpv = (typical_price * group['volume']).cumsum()
            cumulative_volume = group['volume'].cumsum()
            vwap = cumulative_tpv / cumulative_volume
            vwap_values.extend(vwap.tolist())
        
        df['vwap'] = vwap_values
        return df
    
    def _add_stochastic(self, df: pd.DataFrame, high_col: str, low_col: str, close_col: str) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        k_period, d_period, smooth_k = self.stoch_periods
        
        # Calculate %K
        lowest_low = df[low_col].rolling(window=k_period).min()
        highest_high = df[high_col].rolling(window=k_period).max()
        
        k_percent = 100 * ((df[close_col] - lowest_low) / (highest_high - lowest_low))
        
        # Smooth %K if specified
        if smooth_k > 1:
            k_percent = k_percent.rolling(window=smooth_k).mean()
        
        # Calculate %D (moving average of %K)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        df['stoch_k'] = k_percent
        df['stoch_d'] = d_percent
        
        return df
    
    def get_indicator_summary(self, df: pd.DataFrame, timestamp: Optional[pd.Timestamp] = None) -> Dict:
        """
        Get a summary of all indicators at a specific timestamp
        
        Args:
            df: DataFrame with calculated indicators
            timestamp: Specific timestamp to get values for (default: latest)
            
        Returns:
            Dictionary with indicator values and signals
        """
        if timestamp is None:
            row = df.iloc[-1]
        else:
            row = df.loc[timestamp]
        
        # Determine price column
        close_col = 'close_norm' if 'close_norm' in df.columns else 'close'
        
        summary = {
            'price': row[close_col],
            'volume': row['volume'],
            'indicators': {}
        }
        
        # RSI
        if 'rsi' in row and not pd.isna(row['rsi']):
            rsi_value = row['rsi']
            summary['indicators']['rsi'] = {
                'value': rsi_value,
                'signal': 'oversold' if rsi_value < 30 else 'overbought' if rsi_value > 70 else 'neutral'
            }
        
        # MACD
        if 'macd' in row and 'macd_signal' in row:
            if not pd.isna(row['macd']) and not pd.isna(row['macd_signal']):
                summary['indicators']['macd'] = {
                    'value': row['macd'],
                    'signal_line': row['macd_signal'],
                    'histogram': row.get('macd_histogram', 0),
                    'signal': 'bullish' if row['macd'] > row['macd_signal'] else 'bearish'
                }
        
        # Bollinger Bands
        if 'bb_upper' in row and 'bb_lower' in row:
            if not pd.isna(row['bb_upper']) and not pd.isna(row['bb_lower']):
                price = row[close_col]
                summary['indicators']['bollinger_bands'] = {
                    'upper': row['bb_upper'],
                    'middle': row.get('bb_middle', 0),
                    'lower': row['bb_lower'],
                    'position': 'above' if price > row['bb_upper'] else 'below' if price < row['bb_lower'] else 'within'
                }
        
        # Stochastic
        if 'stoch_k' in row and 'stoch_d' in row:
            if not pd.isna(row['stoch_k']) and not pd.isna(row['stoch_d']):
                stoch_k = row['stoch_k']
                summary['indicators']['stochastic'] = {
                    'k': stoch_k,
                    'd': row['stoch_d'],
                    'signal': 'oversold' if stoch_k < 20 else 'overbought' if stoch_k > 80 else 'neutral'
                }
        
        # Moving Averages
        for period in self.sma_periods:
            col_name = f'sma_{period}'
            if col_name in row and not pd.isna(row[col_name]):
                summary['indicators'][col_name] = row[col_name]
        
        for period in self.ema_periods:
            col_name = f'ema_{period}'
            if col_name in row and not pd.isna(row[col_name]):
                summary['indicators'][col_name] = row[col_name]
        
        # VWAP
        if 'vwap' in row and not pd.isna(row['vwap']):
            summary['indicators']['vwap'] = {
                'value': row['vwap'],
                'position': 'above' if row[close_col] > row['vwap'] else 'below'
            }
        
        # OBV
        if 'obv' in row and not pd.isna(row['obv']):
            summary['indicators']['obv'] = row['obv']
        
        return summary


def add_all_indicators(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Convenience function to add all indicators to a dataframe
    
    Args:
        df: DataFrame with OHLCV data
        config: Optional configuration dictionary
        
    Returns:
        DataFrame with all indicators added
    """
    indicators = TechnicalIndicators(config)
    return indicators.calculate_all_indicators(df)