import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class VanillaAutoRegressive:
    def __init__(self, order=5):
        self.order = order
        self.models = {}
        self.seasonal_components = {}
        self.fitted = False
        
    def create_lagged_features(self, series, order):
        """Create lagged features for autoregressive model"""
        X = []
        y = []
        
        for i in range(order, len(series)):
            X.append(series[i-order:i])
            y.append(series[i])
            
        return np.array(X), np.array(y)
    
    def decompose_series(self, series, period=365):
        """Decompose time series into trend, seasonal, and residual components"""
        try:
            # Ensure series is long enough for decomposition
            if len(series) < 2 * period:
                period = min(30, len(series) // 4)  # Use monthly if not enough data
                
            decomposition = seasonal_decompose(
                series, 
                model='additive', 
                period=period,
                extrapolate_trend='freq'
            )
            
            return {
                'trend': decomposition.trend.fillna(method='bfill').fillna(method='ffill'),
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid.fillna(0),
                'period': period
            }
        except:
            # Fallback: create simple seasonal pattern
            n = len(series)
            trend = np.linspace(series.iloc[0], series.iloc[-1], n)
            seasonal = np.tile(np.sin(2 * np.pi * np.arange(period) / period), n // period + 1)[:n]
            residual = series - trend - seasonal
            
            return {
                'trend': pd.Series(trend, index=series.index),
                'seasonal': pd.Series(seasonal, index=series.index),
                'residual': pd.Series(residual, index=series.index),
                'period': period
            }
    
    def fit(self, data, target_columns):
        """Fit AR models for each target variable with seasonal decomposition"""
        self.target_columns = target_columns
        
        for col in target_columns:
            if col in data.columns:
                series = pd.Series(data[col].values, index=range(len(data[col])))
                
                # Decompose the series
                decomp = self.decompose_series(series)
                self.seasonal_components[col] = decomp
                
                # Fit AR model on detrended and deseasonalized residuals
                residual_values = decomp['residual'].values
                X, y = self.create_lagged_features(residual_values, self.order)
                
                model = LinearRegression()
                model.fit(X, y)
                self.models[col] = model
                
        self.fitted = True
        
    def predict_single_step(self, last_values, target_col):
        """Predict next value given last 'order' values"""
        if not self.fitted or target_col not in self.models:
            raise ValueError(f"Model not fitted for {target_col}")
            
        return self.models[target_col].predict([last_values])[0]
    
    def predict_recursive(self, initial_values, steps, target_col):
        """Recursively predict multiple steps ahead with seasonal reconstruction"""
        if target_col not in self.seasonal_components:
            # Fallback to original method
            predictions = []
            current_values = initial_values[-self.order:].copy()
            
            for _ in range(steps):
                next_pred = self.predict_single_step(current_values, target_col)
                # Add small noise to prevent convergence
                noise = np.random.normal(0, np.std(initial_values) * 0.01)
                next_pred += noise
                predictions.append(next_pred)
                current_values = np.append(current_values[1:], next_pred)
                
            return np.array(predictions)
        
        # Use seasonal decomposition approach
        decomp = self.seasonal_components[target_col]
        period = decomp['period']
        
        # Get last residual values for AR prediction
        last_residuals = decomp['residual'].values[-self.order:]
        
        # Predict residuals
        residual_predictions = []
        current_residuals = last_residuals.copy()
        
        for step in range(steps):
            next_residual = self.predict_single_step(current_residuals, target_col)
            # Add controlled noise
            noise_std = np.std(decomp['residual'].values) * 0.05
            noise = np.random.normal(0, noise_std)
            next_residual += noise
            
            residual_predictions.append(next_residual)
            current_residuals = np.append(current_residuals[1:], next_residual)
        
        # Reconstruct predictions with trend and seasonality
        predictions = []
        last_trend = decomp['trend'].values[-1] if not pd.isna(decomp['trend'].values[-1]) else np.mean(decomp['trend'].dropna().values)
        seasonal_pattern = decomp['seasonal'].values
        
        for step in range(steps):
            # Linear trend continuation
            trend_value = last_trend + (step + 1) * 0.001  # Very small trend
            
            # Repeat seasonal pattern
            seasonal_idx = (len(decomp['seasonal']) + step) % period
            seasonal_value = seasonal_pattern[seasonal_idx]
            
            # Combine components
            prediction = trend_value + seasonal_value + residual_predictions[step]
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_50_years(self, data, target_columns):
        """Predict 50 years (18250 days) ahead for all target variables"""
        return self.predict_n_years(data, target_columns, 18250)
    
    def predict_n_years(self, data, target_columns, n_days):
        """Predict n days ahead for all target variables"""
        predictions = {}
        
        for col in target_columns:
            if col in data.columns and col in self.models:
                series = data[col].values
                
                if col in self.seasonal_components:
                    # Use full series for seasonal reconstruction
                    initial_values = series
                else:
                    # Fallback: use last 'order' values
                    initial_values = series[-self.order:]
                
                # Predict n days
                pred_n_days = self.predict_recursive(initial_values, n_days, col)
                predictions[col] = pred_n_days
                
        return predictions
    
    def evaluate(self, true_values, predictions):
        """Evaluate predictions using MSE and MAE"""
        results = {}
        
        for col in predictions.keys():
            if col in true_values:
                mse = mean_squared_error(true_values[col], predictions[col])
                mae = mean_absolute_error(true_values[col], predictions[col])
                results[col] = {'MSE': mse, 'MAE': mae, 'RMSE': np.sqrt(mse)}
                
        return results