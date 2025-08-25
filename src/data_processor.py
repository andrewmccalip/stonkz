"""Data processing module for loading and normalizing stock market data"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pathlib import Path
from typing import Optional, Tuple, Dict
import pickle
import hashlib
import json
import os

class DataProcessor:
    """Handles loading, normalization, and processing of market data"""
    
    def __init__(self, data_path: str = "databento/SPY/", cache_dir: str = "cache/"):
        self.data_path = Path(data_path)
        self.cache_dir = Path(cache_dir)
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.market_close_time = "16:00"
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_key(self, filename: str, date: str, include_indicators: bool = True) -> str:
        """Generate a unique cache key for the processed data"""
        key_data = {
            'filename': filename,
            'date': date,
            'include_indicators': include_indicators,
            'version': '1.0'  # Increment this to invalidate all caches
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the full path for a cache file"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load processed data from cache if available"""
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                # Check if cache is newer than source file
                source_path = self.data_path / self.filename_in_use
                if source_path.exists():
                    source_mtime = source_path.stat().st_mtime
                    cache_mtime = cache_path.stat().st_mtime
                    
                    if cache_mtime > source_mtime:
                        with open(cache_path, 'rb') as f:
                            data = pickle.load(f)
                            print(f"Loaded data from cache: {cache_key}")
                            return data
                    else:
                        print(f"Cache is older than source file, regenerating...")
                        
            except Exception as e:
                print(f"Error loading cache: {e}")
                
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """Save processed data to cache"""
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved data to cache: {cache_key}")
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def clear_cache(self):
        """Clear all cached files"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print("Cache cleared")
    
    def load_csv_data(self, filename: str) -> pd.DataFrame:
        """
        Load Databento CSV file and parse timestamps
        
        Args:
            filename: Name of the CSV file to load
            
        Returns:
            DataFrame with parsed timestamps and price data
        """
        # Store filename for cache checking
        self.filename_in_use = filename
        
        filepath = self.data_path / filename
        
        # Load CSV with specific dtypes
        df = pd.read_csv(
            filepath,
            parse_dates=['ts_event'],
            date_format='ISO8601'
        )
        
        # Convert timestamp to Eastern timezone
        df['ts_event'] = pd.to_datetime(df['ts_event']).dt.tz_convert(self.eastern_tz)
        
        # Set timestamp as index
        df.set_index('ts_event', inplace=True)
        
        # Keep only relevant columns
        columns_to_keep = ['open', 'high', 'low', 'close', 'volume', 'symbol']
        df = df[columns_to_keep]
        
        # Filter for only 4-character symbols (actual futures contracts, not spreads)
        print(f"Original data: {len(df)} rows, {df['symbol'].nunique()} unique symbols")
        df = df[df['symbol'].str.len() == 4]
        print(f"After filtering for 4-char symbols: {len(df)} rows, {df['symbol'].nunique()} unique symbols")
        
        # Filter for ESM5 only
        df = df[df['symbol'] == 'ESM5']
        print(f"After filtering for ESM5: {len(df)} rows")
        
        if df.empty:
            print("WARNING: No ESM5 data found!")
        else:
            date_range = f"{df.index.min()} to {df.index.max()}"
            print(f"ESM5 data range: {date_range}")
        
        return df
    
    def get_trading_day_data(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        Extract data for a specific trading day (4:01PM previous day to 4PM current day)
        
        Args:
            df: DataFrame with market data
            date: Date string in format 'YYYY-MM-DD'
            
        Returns:
            DataFrame containing data for the specified trading day
        """
        # Parse the target date
        target_date = pd.to_datetime(date)
        target_date = self.eastern_tz.localize(target_date.replace(hour=16, minute=0, second=0))
        
        # Define the trading day boundaries
        # Start at 4:01 PM previous day (market close + 1 minute)
        start_time = target_date - timedelta(days=1) + timedelta(minutes=1)
        end_time = target_date
        
        # Filter data for the trading day (inclusive of start time)
        mask = (df.index >= start_time) & (df.index <= end_time)
        trading_day_data = df.loc[mask].copy()
        
        # If we don't have data from 4:01 PM previous day, use earliest available data
        if len(trading_day_data) == 0 or trading_day_data.index[0] > start_time:
            # Get all data for the target date up to 4 PM
            date_only = pd.to_datetime(date).date()
            mask = (df.index.date == date_only) & (df.index <= end_time)
            trading_day_data = df.loc[mask].copy()
            
            if len(trading_day_data) > 0:
                actual_start = trading_day_data.index[0].strftime('%I:%M %p %Z')
                print(f"Note: No data from 4:01 PM previous day. Using earliest available data from {actual_start}")
        
        return trading_day_data
    
    def normalize_prices(self, df: pd.DataFrame, base_price: Optional[float] = None) -> pd.DataFrame:
        """
        Normalize price data to 1.00 based on previous day's closing price
        
        Args:
            df: DataFrame with price data
            base_price: Optional base price for normalization. If None, uses first close price
            
        Returns:
            DataFrame with normalized prices
        """
        # Make a copy to avoid modifying original
        normalized_df = df.copy()
        
        # Determine base price
        if base_price is None:
            # Find the previous day's 4PM close
            # For now, use the first available close price
            base_price = df['close'].iloc[0]
        
        # Normalize price columns
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            normalized_df[f'{col}_norm'] = df[col] / base_price
        
        # Keep volume as is
        normalized_df['volume'] = df['volume']
        
        return normalized_df
    
    def get_previous_close(self, df: pd.DataFrame, current_date: str) -> float:
        """
        Get the previous trading day's closing price at 4PM
        
        Args:
            df: DataFrame with market data
            current_date: Current date string
            
        Returns:
            Previous day's closing price
        """
        current = pd.to_datetime(current_date)
        current = self.eastern_tz.localize(current.replace(hour=16, minute=0, second=0))
        
        # Look for previous day's 4PM close
        prev_close_time = current - timedelta(days=1)
        
        # Find the closest price to 4PM
        time_window = pd.Timedelta(minutes=30)
        mask = (df.index >= prev_close_time - time_window) & (df.index <= prev_close_time + time_window)
        close_prices = df.loc[mask]
        
        if not close_prices.empty:
            # Get the price closest to 4PM
            time_diffs = pd.Series(close_prices.index - prev_close_time)
            closest_idx = time_diffs.abs().argmin()
            return close_prices.iloc[closest_idx]['close']
        else:
            # Fallback to last available price before target time
            before_close = df[df.index < prev_close_time]
            if not before_close.empty:
                return before_close.iloc[-1]['close']
            else:
                # Use first available price
                return df.iloc[0]['close']
    
    def slice_data_to_current(self, df: pd.DataFrame, current_time: str) -> pd.DataFrame:
        """
        Slice data from start up to current moment
        
        Args:
            df: DataFrame with market data
            current_time: Current time string (ISO format or 'HH:MM' for today)
            
        Returns:
            DataFrame containing data up to current time
        """
        if ':' in current_time and len(current_time) <= 5:
            # Time only provided, assume today
            today = datetime.now(self.eastern_tz).date()
            hour, minute = map(int, current_time.split(':'))
            current_dt = self.eastern_tz.localize(datetime.combine(today, datetime.min.time()).replace(hour=hour, minute=minute))
        else:
            # Full datetime provided
            current_dt = pd.to_datetime(current_time)
            if current_dt.tzinfo is None:
                current_dt = self.eastern_tz.localize(current_dt)
            else:
                current_dt = current_dt.astimezone(self.eastern_tz)
        
        # Return data up to current time
        return df[df.index <= current_dt].copy()
    
    def get_available_dates(self, df: pd.DataFrame) -> list:
        """
        Get list of available trading dates in the dataset
        
        Args:
            df: DataFrame with market data
            
        Returns:
            List of date strings
        """
        # Get unique dates
        dates = df.index.normalize().unique()
        
        # Convert to strings
        return [date.strftime('%Y-%m-%d') for date in dates]
    
    def prepare_data_for_prediction(self, df: pd.DataFrame, lookback_periods: Optional[int] = None) -> Dict:
        """
        Prepare data for AI model prediction
        
        Args:
            df: Normalized DataFrame with technical indicators
            lookback_periods: Number of periods to include for context (None = all data)
            
        Returns:
            Dictionary with prepared data for model input
        """
        # Use all data if lookback_periods is None, otherwise get the most recent data
        if lookback_periods is not None:
            recent_data = df.tail(lookback_periods)
        else:
            recent_data = df
        
        # Prepare time series data
        data_dict = {
            'timestamps': recent_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'prices': {
                'open': recent_data['open_norm'].tolist(),
                'high': recent_data['high_norm'].tolist(),
                'low': recent_data['low_norm'].tolist(),
                'close': recent_data['close_norm'].tolist()
            },
            'volume': recent_data['volume'].tolist(),
            'current_price': recent_data['close_norm'].iloc[-1],
            'current_time': recent_data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
            'market_close': recent_data.index[-1].replace(hour=16, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add technical indicators if they exist
        indicator_columns = [col for col in recent_data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'open_norm', 'high_norm', 'low_norm', 'close_norm']]
        if indicator_columns:
            data_dict['indicators'] = {}
            for col in indicator_columns:
                if not recent_data[col].isna().all():
                    data_dict['indicators'][col] = recent_data[col].fillna(0).tolist()
        
        return data_dict
    
    def get_processed_trading_day(self, filename: str, target_date: str, include_indicators: bool = True) -> pd.DataFrame:
        """
        Get fully processed trading day data with caching support
        
        Args:
            filename: CSV filename
            target_date: Date string 'YYYY-MM-DD'
            include_indicators: Whether to calculate technical indicators
            
        Returns:
            Processed DataFrame with normalized prices and optionally indicators
        """
        # Generate cache key
        cache_key = self._get_cache_key(filename, target_date, include_indicators)
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # If not in cache, process the data
        print(f"Processing data for {target_date}...")
        
        # Load raw data
        df = self.load_csv_data(filename)
        
        # Get trading day data
        trading_day = self.get_trading_day_data(df, target_date)
        
        # Get previous close for normalization
        prev_close = self.get_previous_close(df, target_date)
        
        # Normalize prices
        normalized = self.normalize_prices(trading_day, prev_close)
        
        # Add technical indicators if requested
        if include_indicators:
            from src.indicators import add_all_indicators
            normalized = add_all_indicators(normalized)
        
        # Save to cache
        self._save_to_cache(normalized, cache_key)
        
        return normalized


# Utility functions for standalone use
def quick_load_and_normalize(filename: str, target_date: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Quick helper function to load and normalize data for a specific date
    
    Args:
        filename: CSV filename
        target_date: Date string 'YYYY-MM-DD'
        use_cache: Whether to use caching
        
    Returns:
        Normalized DataFrame for the trading day
    """
    processor = DataProcessor()
    
    if use_cache:
        return processor.get_processed_trading_day(filename, target_date, include_indicators=True)
    else:
        # Original implementation without caching
        df = processor.load_csv_data(filename)
        trading_day = processor.get_trading_day_data(df, target_date)
        prev_close = processor.get_previous_close(df, target_date)
        normalized = processor.normalize_prices(trading_day, prev_close)
        
        from src.indicators import add_all_indicators
        return add_all_indicators(normalized)