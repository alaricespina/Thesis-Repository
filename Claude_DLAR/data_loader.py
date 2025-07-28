import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    def __init__(self, data_source_path="../Data Source Files/"):
        self.data_source_path = Path(data_source_path)
        
    def load_historical_data(self, start_year=2001, end_year=2024):
        """Load and concatenate historical data from 2001-2024"""
        dfs = []
        
        for year in range(start_year, end_year + 1):
            file_path = self.data_source_path / f"{year}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['datetime'] = pd.to_datetime(df['datetime'])
                dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
        
        return combined_df
    
    def load_test_data(self, year=2025):
        """Load 2025 data for testing"""
        file_path = self.data_source_path / f"{year}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            return df
        return None
    
    def prepare_target_variables(self, df):
        """Extract temperature, humidity, and precipitation columns"""
        target_cols = ['temp', 'humidity', 'precip']
        
        # Ensure all target columns exist
        available_cols = []
        for col in target_cols:
            if col in df.columns:
                available_cols.append(col)
        
        if not available_cols:
            raise ValueError("No target columns found in data")
            
        return df[['datetime'] + available_cols].copy()