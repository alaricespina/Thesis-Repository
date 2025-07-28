import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SimpleSARIMA:
    """
    Simplified SARIMA implementation using seasonal differencing and linear regression
    Fallback when statsmodels SARIMA hangs or fails
    """
    def __init__(self, ar_order=3, seasonal_period=12):
        self.ar_order = ar_order
        self.seasonal_period = seasonal_period
        self.models = {}
        self.fitted = False
        self.seasonal_means = {}
        
    def seasonal_difference(self, series, period):
        """Apply seasonal differencing"""
        return series.diff(period).dropna()
    
    def first_difference(self, series):
        """Apply first-order differencing"""
        return series.diff().dropna()
    
    def create_ar_features(self, series, order):
        """Create autoregressive features"""
        X = []
        y = []
        
        for i in range(order, len(series)):
            X.append(series[i-order:i])
            y.append(series[i])
            
        return np.array(X), np.array(y)
    
    def fit(self, data, target_columns):
        """Fit Simple SARIMA models"""
        self.target_columns = target_columns
        
        for col in target_columns:
            if col in data.columns:
                print(f"Fitting Simple SARIMA for {col}...")
                series = pd.Series(data[col].dropna().values)
                
                try:
                    # Store original series statistics
                    self.seasonal_means[col] = {}
                    
                    # Calculate seasonal means for reconstruction
                    for month in range(self.seasonal_period):
                        mask = np.arange(len(series)) % self.seasonal_period == month
                        if np.any(mask):
                            self.seasonal_means[col][month] = series[mask].mean()
                        else:
                            self.seasonal_means[col][month] = series.mean()
                    
                    # Apply seasonal differencing
                    seasonal_diff = self.seasonal_difference(series, self.seasonal_period)
                    
                    if len(seasonal_diff) > 0:
                        # Apply first differencing
                        stationary = self.first_difference(seasonal_diff)
                        
                        if len(stationary) >= self.ar_order:
                            # Create AR features
                            X, y = self.create_ar_features(stationary, self.ar_order)
                            
                            if len(X) > 0:
                                # Fit linear regression model
                                model = LinearRegression()
                                model.fit(X, y)
                                
                                self.models[col] = {
                                    'ar_model': model,
                                    'original_series': series,
                                    'seasonal_diff': seasonal_diff,
                                    'stationary': stationary,
                                    'last_values': stationary[-self.ar_order:].values
                                }
                                print(f"  ✓ Simple SARIMA fitted for {col}")
                            else:
                                print(f"  ✗ Not enough data for AR features for {col}")
                        else:
                            print(f"  ✗ Not enough stationary data for {col}")
                    else:
                        print(f"  ✗ Seasonal differencing failed for {col}")
                        
                except Exception as e:
                    print(f"  ✗ Error fitting Simple SARIMA for {col}: {e}")
                    
        self.fitted = True
        
    def predict_recursive(self, last_values, steps, col):
        """Recursively predict on stationary data"""
        if col not in self.models:
            return np.zeros(steps)
            
        predictions = []
        current_values = last_values.copy()
        model = self.models[col]['ar_model']
        
        for _ in range(steps):
            next_pred = model.predict([current_values])[0]
            predictions.append(next_pred)
            
            # Update for next prediction
            current_values = np.append(current_values[1:], next_pred)
            
        return np.array(predictions)
    
    def reconstruct_seasonal(self, stationary_pred, col, start_idx=0):
        """Reconstruct seasonal and trend components"""
        if col not in self.models:
            return stationary_pred
            
        # Add seasonal component
        seasonal_component = []
        for i in range(len(stationary_pred)):
            month = (start_idx + i) % self.seasonal_period
            seasonal_component.append(self.seasonal_means[col][month])
            
        seasonal_component = np.array(seasonal_component)
        
        # Simple trend component (linear)
        original_series = self.models[col]['original_series']
        trend_slope = (original_series.iloc[-1] - original_series.iloc[-100]) / 100 if len(original_series) >= 100 else 0
        trend_component = np.arange(len(stationary_pred)) * trend_slope
        
        # Combine components
        final_pred = stationary_pred + seasonal_component + trend_component
        
        return final_pred
    
    def predict_50_years(self, target_columns):
        """Predict 50 years ahead"""
        return self.predict_n_years(target_columns, 18250)
    
    def predict_n_years(self, target_columns, n_days):
        """Predict n days ahead"""
        predictions = {}
        n_years = n_days / 365
        
        for col in target_columns:
            if col in self.models:
                print(f"Simple SARIMA predicting {n_years:.1f} years for {col}...")
                
                last_values = self.models[col]['last_values']
                
                # Predict on stationary data
                stationary_pred = self.predict_recursive(last_values, n_days, col)
                
                # Reconstruct with seasonal components
                final_pred = self.reconstruct_seasonal(stationary_pred, col)
                
                predictions[col] = final_pred
                
        return predictions
    
    def predict_test_period(self, test_length, target_columns):
        """Predict for test period"""
        predictions = {}
        
        for col in target_columns:
            if col in self.models:
                last_values = self.models[col]['last_values']
                
                stationary_pred = self.predict_recursive(last_values, test_length, col)
                final_pred = self.reconstruct_seasonal(stationary_pred, col)
                
                predictions[col] = final_pred
                
        return predictions
    
    def evaluate(self, true_values, predictions):
        """Evaluate predictions"""
        results = {}
        
        for col in predictions.keys():
            if col in true_values:
                min_len = min(len(true_values[col]), len(predictions[col]))
                true_vals = true_values[col][:min_len]
                pred_vals = predictions[col][:min_len]
                
                mse = mean_squared_error(true_vals, pred_vals)
                mae = mean_absolute_error(true_vals, pred_vals)
                results[col] = {'MSE': mse, 'MAE': mae, 'RMSE': np.sqrt(mse)}
                
        return results