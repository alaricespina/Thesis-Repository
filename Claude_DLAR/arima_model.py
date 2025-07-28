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
    
    def extract_seasonal_pattern(self, series, period):
        """Extract seasonal pattern from time series"""
        try:
            if len(series) < period:
                # Create simple sinusoidal pattern
                return np.sin(2 * np.pi * np.arange(period) / period)
            
            # Calculate average pattern for each position in the cycle
            pattern = np.zeros(period)
            counts = np.zeros(period)
            
            for i, value in enumerate(series):
                if not pd.isna(value):
                    pos = i % period
                    pattern[pos] += value
                    counts[pos] += 1
            
            # Average and normalize
            pattern = np.where(counts > 0, pattern / counts, 0)
            pattern = pattern - np.mean(pattern)  # Center around zero
            
            return pattern
        except:
            # Fallback: simple sinusoidal pattern
            return np.sin(2 * np.pi * np.arange(period) / period)
    
    def predict_n_years(self, target_columns, n_days):
        """Predict n days ahead for all target variables with improved approach"""
        predictions = {}
        
        for col in target_columns:
            if col in self.models:
                try:
                    model_info = self.models[col]
                    fitted_model = model_info['model']
                    original_series = model_info['original_series']
                    
                    # For very long forecasts, use a chunked approach
                    if n_days > 1000:
                        # Break into smaller chunks and add seasonal patterns
                        chunk_size = 365  # One year chunks
                        all_forecasts = []
                        
                        # Calculate yearly seasonal pattern from historical data
                        yearly_pattern = self.extract_seasonal_pattern(original_series, 365)
                        
                        current_model = fitted_model
                        for chunk_start in range(0, n_days, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, n_days)
                            chunk_length = chunk_end - chunk_start
                            
                            # Forecast this chunk
                            chunk_forecast = current_model.forecast(steps=chunk_length)
                            
                            # Add seasonal pattern to prevent flat lines
                            seasonal_adjustment = yearly_pattern[:chunk_length] * 0.1  # Subtle seasonal effect
                            adjusted_forecast = chunk_forecast.values + seasonal_adjustment
                            
                            # Add small trend component
                            trend_adjustment = np.linspace(0, 0.01 * chunk_length, chunk_length)
                            adjusted_forecast += trend_adjustment
                            
                            all_forecasts.extend(adjusted_forecast)
                            
                            # Re-fit model with extended data for next chunk
                            if chunk_end < n_days:
                                try:
                                    extended_series = pd.concat([
                                        original_series,
                                        pd.Series(adjusted_forecast, index=range(len(original_series), len(original_series) + chunk_length))
                                    ])
                                    new_model = ARIMA(extended_series, order=model_info['order'])
                                    current_model = new_model.fit()
                                except:
                                    # Keep using original model if refit fails
                                    pass
                        
                        predictions[col] = np.array(all_forecasts)
                    else:
                        # For shorter forecasts, use direct approach with seasonal adjustment
                        forecast = fitted_model.forecast(steps=n_days)
                        
                        # Add seasonal pattern
                        seasonal_pattern = self.extract_seasonal_pattern(original_series, min(365, len(original_series) // 3))
                        seasonal_cycle = np.tile(seasonal_pattern, (n_days // len(seasonal_pattern)) + 1)[:n_days]
                        
                        # Combine forecast with subtle seasonal adjustment
                        adjusted_forecast = forecast.values + seasonal_cycle * 0.05
                        predictions[col] = adjusted_forecast
                    
                except Exception as e:
                    print(f"Error predicting for {col}: {e}")
                    # Fallback: create realistic seasonal pattern
                    if col in self.models and 'original_series' in self.models[col]:
                        mean_val = self.models[col]['original_series'].mean()
                        std_val = self.models[col]['original_series'].std()
                        seasonal_pattern = self.extract_seasonal_pattern(self.models[col]['original_series'], 365)
                        seasonal_cycle = np.tile(seasonal_pattern, (n_days // len(seasonal_pattern)) + 1)[:n_days]
                        predictions[col] = mean_val + seasonal_cycle * std_val * 0.1
                    else:
                        predictions[col] = np.zeros(n_days)
                    
        return predictions
    
    def predict_test_period(self, test_length, target_columns):
        """Predict for test period (e.g., 2025 data)"""
        predictions = {}
        
        for col in target_columns:
            if col in self.models:
                try:
                    model_info = self.models[col]
                    fitted_model = model_info['model']
                    original_series = model_info['original_series']
                    
                    # Get base forecast
                    forecast = fitted_model.forecast(steps=test_length)
                    
                    # Add seasonal pattern for more realistic predictions
                    seasonal_pattern = self.extract_seasonal_pattern(original_series, min(365, len(original_series) // 3))
                    seasonal_cycle = np.tile(seasonal_pattern, (test_length // len(seasonal_pattern)) + 1)[:test_length]
                    
                    # Combine with subtle seasonal effect
                    adjusted_forecast = forecast.values + seasonal_cycle * 0.1
                    predictions[col] = adjusted_forecast
                    
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