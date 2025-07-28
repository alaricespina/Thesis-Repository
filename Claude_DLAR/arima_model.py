import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.models = {}
        self.fitted = False
        
    def check_stationarity(self, series):
        """Check if series is stationary using ADF test"""
        result = adfuller(series.dropna())
        return result[1] <= 0.05  # p-value <= 0.05 indicates stationarity
    
    def make_stationary(self, series, max_diff=2):
        """Make series stationary through differencing"""
        diff_series = series.copy()
        d = 0
        
        while d <= max_diff and not self.check_stationarity(diff_series):
            diff_series = diff_series.diff().dropna()
            d += 1
            
        return diff_series, d
    
    def fit(self, data, target_columns):
        """Fit ARIMA models for each target variable"""
        self.target_columns = target_columns
        
        for col in target_columns:
            if col in data.columns:
                series = data[col].dropna()
                
                try:
                    # Automatically determine differencing order
                    stationary_series, d_order = self.make_stationary(series)
                    
                    # Update ARIMA order with determined d
                    arima_order = (self.order[0], d_order, self.order[2])
                    
                    # Fit ARIMA model
                    model = ARIMA(series, order=arima_order)
                    fitted_model = model.fit()
                    
                    self.models[col] = {
                        'model': fitted_model,
                        'order': arima_order,
                        'original_series': series
                    }
                    
                except Exception as e:
                    print(f"Error fitting ARIMA for {col}: {e}")
                    # Fallback to simple ARIMA(1,1,1)
                    try:
                        model = ARIMA(series, order=(1, 1, 1))
                        fitted_model = model.fit()
                        self.models[col] = {
                            'model': fitted_model,
                            'order': (1, 1, 1),
                            'original_series': series
                        }
                    except:
                        print(f"Failed to fit any ARIMA model for {col}")
                        
        self.fitted = True
        
    def predict_50_years(self, target_columns):
        """Predict 50 years (18250 days) ahead for all target variables"""
        return self.predict_n_years(target_columns, 18250)
    
    def predict_n_years(self, target_columns, n_days):
        """Predict n days ahead for all target variables"""
        predictions = {}
        
        for col in target_columns:
            if col in self.models:
                try:
                    # Predict n days
                    forecast = self.models[col]['model'].forecast(steps=n_days)
                    predictions[col] = forecast.values
                    
                except Exception as e:
                    print(f"Error predicting for {col}: {e}")
                    predictions[col] = np.zeros(n_days)
                    
        return predictions
    
    def predict_test_period(self, test_length, target_columns):
        """Predict for test period (e.g., 2025 data)"""
        predictions = {}
        
        for col in target_columns:
            if col in self.models:
                try:
                    forecast = self.models[col]['model'].forecast(steps=test_length)
                    predictions[col] = forecast.values
                except Exception as e:
                    print(f"Error predicting test period for {col}: {e}")
                    predictions[col] = np.zeros(test_length)
                    
        return predictions
    
    def evaluate(self, true_values, predictions):
        """Evaluate predictions using MSE and MAE"""
        results = {}
        
        for col in predictions.keys():
            if col in true_values:
                # Handle different lengths
                min_len = min(len(true_values[col]), len(predictions[col]))
                true_vals = true_values[col][:min_len]
                pred_vals = predictions[col][:min_len]
                
                mse = mean_squared_error(true_vals, pred_vals)
                mae = mean_absolute_error(true_vals, pred_vals)
                results[col] = {'MSE': mse, 'MAE': mae, 'RMSE': np.sqrt(mse)}
                
        return results
    
    def get_model_summary(self, target_col):
        """Get model summary for a specific target variable"""
        if target_col in self.models:
            return self.models[target_col]['model'].summary()
        return None