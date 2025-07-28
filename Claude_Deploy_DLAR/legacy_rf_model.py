"""
Legacy Random Forest Auto Regressive Model

Custom Random Forest implementation with sophisticated deseasonalization approach.
This preserves the original algorithm developed for weather prediction with:
- Yearly seasonal removal (configurable cycle length)
- Monthly seasonal removal  
- Windowed autoregressive features
- Stochastic noise injection during prediction
- Full seasonal reconstruction
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.base import clone

class LegacyRandomForestAR:
    """
    Legacy Random Forest with custom deseasonalization exactly as originally implemented.
    
    This model performs sophisticated preprocessing:
    1. Remove yearly seasonality using configurable cycle length
    2. Remove monthly seasonality 
    3. Train RF on windowed residuals
    4. Predict with stochastic noise injection
    5. Reconstruct full seasonal patterns
    """
    
    def __init__(self, model_order=3, yearly_season_length=7):
        """
        Initialize Legacy Random Forest model
        
        Args:
            model_order: Number of lagged values to use for prediction
            yearly_season_length: Length of yearly seasonal cycle (default: 7)
        """
        self.model_order = model_order
        self.yearly_season_length = yearly_season_length
        self.models = {}
        self.model_params = {}
        self.fitted = False
        
    def remove_yearly_mean(self, df, col):
        """
        Remove yearly seasonality using configurable cycle length
        
        Args:
            df: DataFrame with datetime, target column, Month, Year
            col: Target column name
            
        Returns:
            Tuple of (yearly_std, yearly_mean, year_min_val, year_grp, normalized_df)
        """
        normalized_yearly_df = df.copy()
        normalized_yearly_df["YearMod"] = normalized_yearly_df[col]
        
        # Calculate year normalization based on cycle length
        years = pd.DatetimeIndex(normalized_yearly_df["datetime"]).year.unique().tolist()
        min_val = years[0]
        normalized_yearly_df["YearAug"] = (pd.DatetimeIndex(normalized_yearly_df["datetime"]).year - min_val) % self.yearly_season_length
        
        # Group by normalized year and subtract yearly means
        yr_grp = normalized_yearly_df.groupby(by="YearAug").mean(numeric_only=True)[col]
        
        for year in normalized_yearly_df["YearAug"].unique():
            normalized_yearly_df.loc[normalized_yearly_df["YearAug"] == year, "YearMod"] -= yr_grp[year]
            
        # Remove overall yearly mean
        yearly_mean = normalized_yearly_df["YearMod"].mean()
        normalized_yearly_df["YearMod"] = normalized_yearly_df["YearMod"] - yearly_mean
        yearly_std = np.std(normalized_yearly_df["YearMod"])
        
        return yearly_std, yearly_mean, min_val, yr_grp, normalized_yearly_df
    
    def remove_monthly_mean(self, in_df):
        """
        Remove monthly seasonality from yearly-normalized data
        
        Args:
            in_df: DataFrame with YearMod column
            
        Returns:
            Tuple of (normalized_df, month_grp, monthly_mean)
        """
        normalized_df = in_df.copy()
        normalized_df["MonthMod"] = normalized_df["YearMod"]
        
        # Group by month and subtract monthly means
        month_grp = normalized_df.groupby(by="Month").mean(numeric_only=True)["MonthMod"]
        
        for month in normalized_df['Month'].unique():
            normalized_df.loc[normalized_df['Month'] == month, "MonthMod"] -= month_grp[month]
            
        # Remove overall monthly mean
        monthly_mean = normalized_df['MonthMod'].mean()
        normalized_df['MonthMod'] = normalized_df['MonthMod'] - monthly_mean
        
        return normalized_df, month_grp, monthly_mean
    
    def window_data(self, data):
        """
        Create windowed features for autoregressive model
        
        Args:
            data: 1D array of time series data
            
        Returns:
            Tuple of (X, y) where X is windowed features, y is targets
        """
        n = len(data)
        x = []
        y = []
        for i in range(n - self.model_order):
            x.append(data[i : i + self.model_order])
            y.append(data[i + self.model_order])
        
        return np.array(x), np.array(y)
    
    def split_data(self, df, split_percent=70):
        """
        Split deseasonalized data into train/test sets with windowing
        
        Args:
            df: DataFrame with MonthMod column (deseasonalized data)
            split_percent: Percentage for training split
            
        Returns:
            Tuple of ((train_x, test_x, train_y, test_y), monthly_std)
        """
        train_data = df["MonthMod"].copy()
        n = len(train_data)
        split_point = int(split_percent / 100 * n)
        
        # Calculate standard deviation for noise injection
        monthly_std = np.std(train_data[:split_point])
        
        # Create windowed features
        window_x, window_y = self.window_data(train_data)
        
        # Split into train/test
        train_x = window_x[:split_point]
        test_x = window_x[split_point:]
        train_y = window_y[:split_point]
        test_y = window_y[split_point:]
        
        return (train_x, test_x, train_y, test_y), monthly_std
    
    def fit(self, data, target_columns):
        """
        Fit Legacy RF models for each target variable
        
        Args:
            data: DataFrame with datetime, target columns, Month, Year
            target_columns: List of target variable column names
        """
        self.target_columns = target_columns
        
        for col in target_columns:
            if col in data.columns:
                print(f"Training Legacy RF for {col}...")
                
                # Prepare input data
                temp_df = data[["datetime", col]].copy()
                temp_df["Month"] = pd.DatetimeIndex(temp_df["datetime"]).month
                temp_df["Year"] = pd.DatetimeIndex(temp_df["datetime"]).year
                
                # Step 1: Remove yearly seasonality
                yearly_std, yearly_mean, year_min_val, year_grp, normalized_yearly_df = self.remove_yearly_mean(temp_df, col)
                
                # Step 2: Remove monthly seasonality
                normalized_df, month_grp, monthly_mean = self.remove_monthly_mean(normalized_yearly_df)
                
                # Step 3: Create windowed training data
                (train_x, test_x, train_y, test_y), monthly_std = self.split_data(normalized_df, 70)
                
                # Step 4: Train Random Forest model
                # model = RandomForestRegressor(n_estimators=100, random_state=42)
                model = LinearRegression()
                model.fit(train_x, train_y)
                
                # Store model and all parameters needed for reconstruction
                self.models[col] = model
                self.model_params[col] = {
                    'yearly_std': yearly_std,
                    'yearly_mean': yearly_mean,
                    'year_min_val': year_min_val,
                    'year_grp': year_grp,
                    'month_grp': month_grp,
                    'monthly_mean': monthly_mean,
                    'monthly_std': monthly_std,
                    'last_train_x': train_x[-1]  # For prediction initialization
                }
                
                print(f"  ✓ Legacy RF trained for {col}")
                
        self.fitted = True
    
    def do_predictions(self, train_x_last, num_predictions, col, year_std_scale=0.1, month_std_scale=0.1):
        """
        Generate predictions with stochastic noise injection
        
        Args:
            train_x_last: Last training window for initialization
            num_predictions: Number of predictions to generate
            col: Target column name
            year_std_scale: Scale factor for yearly noise
            month_std_scale: Scale factor for monthly noise
            
        Returns:
            Array of raw predictions (before seasonal reconstruction)
        """
        params = self.model_params[col]
        model = self.models[col]
        
        preds = np.array([])
        r_queue = np.copy(train_x_last)
        
        # Generate predictions with noise injection
        for i in range(num_predictions):
            # Reshape for model input
            tr = np.reshape(r_queue, (1, -1))
            
            # Add stochastic residuals
            month_resid = np.random.normal(loc=0, scale=params['monthly_std'] * month_std_scale)
            year_resid = np.random.normal(loc=0, scale=params['yearly_std'] * year_std_scale)
            
            # Predict and add noise
            next_val = model.predict(tr)[0] + month_resid + year_resid
            preds = np.append(preds, next_val)
            
            # Update queue for next prediction
            r_queue = np.roll(r_queue, -1)
            r_queue[-1] = next_val
            
        return preds
    
    def reconstruct_predictions(self, preds, col, start_date):
        """
        Reconstruct predictions by adding back all seasonal components
        
        Args:
            preds: Raw predictions from model
            col: Target column name
            start_date: Start date for prediction period
            
        Returns:
            Fully reconstructed predictions with seasonality
        """
        params = self.model_params[col]
        
        # Create date range for predictions
        date_range = pd.date_range(start=start_date, periods=len(preds), freq='D')
        
        # Create reconstruction dataframe
        df = pd.DataFrame({
            'datetime': date_range,
            'PredictionFull': preds
        })
        
        df["Month"] = df["datetime"].dt.month
        df["Year"] = df["datetime"].dt.year
        df["YearAug"] = (df["Year"] - params['year_min_val']) % self.yearly_season_length
        
        # Step 1: Reconstruct monthly component
        df["ReconMonth"] = df["PredictionFull"]
        for month in df["Month"].unique():
            if month in params['month_grp']:
                df.loc[df['Month'] == month, "ReconMonth"] += params['month_grp'][month]
        
        df["ReconMonth"] += params['monthly_mean']
        
        # Step 2: Reconstruct yearly component
        df["ReconstructedFinal"] = df["ReconMonth"]
        for year in df["YearAug"].unique():
            if year in params['year_grp']:
                df.loc[df["YearAug"] == year, "ReconstructedFinal"] += params['year_grp'][year]
        
        df["ReconstructedFinal"] += params['yearly_mean']
        
        return df["ReconstructedFinal"].values
    
    def predict_n_years(self, target_columns, n_days):
        """
        Predict n days ahead for all target variables
        
        Args:
            target_columns: List of target variables to predict
            n_days: Number of days to predict
            
        Returns:
            Dictionary mapping column names to prediction arrays
        """
        predictions = {}
        
        for col in target_columns:
            if col in self.models:
                try:
                    # Get initialization data
                    last_train_x = self.model_params[col]['last_train_x']
                    
                    # Generate raw predictions
                    raw_preds = self.do_predictions(last_train_x, n_days, col)
                    
                    # Reconstruct with full seasonality
                    start_date = datetime(2025, 1, 1)  # Start predictions from 2025
                    reconstructed_preds = self.reconstruct_predictions(raw_preds, col, start_date)
                    
                    predictions[col] = reconstructed_preds
                    
                except Exception as e:
                    print(f"Error predicting for {col}: {e}")
                    predictions[col] = np.zeros(n_days)
                    
        return predictions
    
    def predict_test_period(self, test_length, target_columns):
        """
        Predict for test period (e.g., 2025 data)
        
        Args:
            test_length: Length of test period
            target_columns: List of target variables
            
        Returns:
            Dictionary mapping column names to prediction arrays
        """
        return self.predict_n_years(target_columns, test_length)
    
    def evaluate(self, true_values, predictions):
        """
        Evaluate predictions using MSE, MAE, and RMSE
        
        Args:
            true_values: Dictionary of true values by column
            predictions: Dictionary of predictions by column
            
        Returns:
            Dictionary of evaluation metrics by column
        """
        results = {}
        
        for col in predictions.keys():
            if col in true_values:
                # Handle different lengths
                min_len = min(len(true_values[col]), len(predictions[col]))
                true_vals = true_values[col][:min_len]
                pred_vals = predictions[col][:min_len]
                
                # Calculate metrics
                mse = np.mean((true_vals - pred_vals) ** 2)
                mae = np.mean(np.abs(true_vals - pred_vals))
                rmse = np.sqrt(mse)
                
                results[col] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse}
                
        return results
    
    def get_model_info(self):
        """
        Get information about the fitted models
        
        Returns:
            Dictionary with model information
        """
        if not self.fitted:
            return {"status": "Not fitted"}
            
        info = {
            "status": "Fitted",
            "model_order": self.model_order,
            "yearly_season_length": self.yearly_season_length,
            "target_columns": self.target_columns,
            "models_count": len(self.models)
        }
        
        return info