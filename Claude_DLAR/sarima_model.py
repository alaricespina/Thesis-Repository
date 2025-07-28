import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import signal
import time
warnings.filterwarnings('ignore')

class SARIMAModel:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.models = {}
        self.fitted = False
        
    def check_stationarity(self, series):
        """Check if series is stationary using ADF test"""
        result = adfuller(series.dropna())
        return result[1] <= 0.05
    
    def detect_seasonality(self, series, max_periods=[12, 24, 365]):
        """Simple seasonality detection by checking autocorrelation"""
        best_period = 12  # Default to monthly seasonality
        
        try:
            from statsmodels.tsa.stattools import acf
            autocorr = acf(series.dropna(), nlags=min(365, len(series)//4))
            
            max_autocorr = 0
            for period in max_periods:
                if period < len(autocorr):
                    if abs(autocorr[period]) > max_autocorr:
                        max_autocorr = abs(autocorr[period])
                        best_period = period
        except:
            pass
            
        return best_period
    
    def fit_with_timeout(self, series, order, seasonal_order, timeout=120):
        """Fit SARIMA model with timeout protection"""
        def timeout_handler(signum, frame):
            raise TimeoutError("SARIMA fitting timed out")
            
        # Set timeout alarm
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            model = SARIMAX(
                series, 
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False, maxiter=100)  # Limit iterations
            signal.alarm(0)  # Cancel timeout
            return fitted_model
        except TimeoutError:
            signal.alarm(0)
            print(f"SARIMA fitting timed out after {timeout} seconds")
            return None
        except Exception as e:
            signal.alarm(0)
            print(f"SARIMA fitting failed: {e}")
            return None

    def fit(self, data, target_columns):
        """Fit SARIMA models for each target variable"""
        self.target_columns = target_columns
        
        for col in target_columns:
            if col in data.columns:
                print(f"Fitting SARIMA for {col}...")
                series = data[col].dropna()
                
                # Limit data size for faster fitting (use last 3 years)
                if len(series) > 1095:  # 3 years
                    series = series.tail(1095)
                    print(f"  Using last {len(series)} data points for faster fitting")
                
                fitted_model = None
                
                # Try simpler SARIMA configurations in order of complexity
                configs_to_try = [
                    ((1, 1, 1), (0, 0, 0, 0)),        # ARIMA only, no seasonality
                    ((1, 1, 1), (1, 0, 1, 12)),       # Simple seasonal
                    ((1, 1, 1), (1, 1, 1, 12)),       # Original config
                    ((2, 1, 2), (1, 1, 1, 12)),       # More complex
                ]
                
                for order, seasonal_order in configs_to_try:
                    print(f"  Trying SARIMA{order}x{seasonal_order}...")
                    
                    fitted_model = self.fit_with_timeout(series, order, seasonal_order, timeout=60)
                    
                    if fitted_model is not None:
                        self.models[col] = {
                            'model': fitted_model,
                            'order': order,
                            'seasonal_order': seasonal_order,
                            'original_series': series
                        }
                        print(f"  ✓ Successfully fitted SARIMA{order}x{seasonal_order} for {col}")
                        break
                    else:
                        print(f"  ✗ Failed to fit SARIMA{order}x{seasonal_order}")
                
                if fitted_model is None:
                    print(f"  Failed to fit any SARIMA configuration for {col}")
                        
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
                    forecast_result = self.models[col]['model'].forecast(steps=n_days)
                    predictions[col] = forecast_result.values
                    
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
                    forecast_result = self.models[col]['model'].forecast(steps=test_length)
                    predictions[col] = forecast_result.values
                except Exception as e:
                    print(f"Error predicting test period for {col}: {e}")
                    predictions[col] = np.zeros(test_length)
                    
        return predictions
    
    def predict_with_confidence_intervals(self, steps, target_columns, alpha=0.05):
        """Predict with confidence intervals"""
        predictions = {}
        
        for col in target_columns:
            if col in self.models:
                try:
                    forecast_result = self.models[col]['model'].get_forecast(steps=steps)
                    mean_forecast = forecast_result.predicted_mean
                    conf_int = forecast_result.conf_int(alpha=alpha)
                    
                    predictions[col] = {
                        'mean': mean_forecast.values,
                        'lower_ci': conf_int.iloc[:, 0].values,
                        'upper_ci': conf_int.iloc[:, 1].values
                    }
                except Exception as e:
                    print(f"Error predicting with CI for {col}: {e}")
                    predictions[col] = {
                        'mean': np.zeros(steps),
                        'lower_ci': np.zeros(steps),
                        'upper_ci': np.zeros(steps)
                    }
                    
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