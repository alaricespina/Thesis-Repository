import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

class FastDeepAR:
    """
    Fast Deep Learning AR fallback using pattern recognition
    Avoids recursive prediction loops that can hang
    """
    def __init__(self, pattern_length=365):
        self.pattern_length = pattern_length
        self.models = {}
        self.scalers = {}
        self.patterns = {}
        self.fitted = False
        
    def extract_patterns(self, series, pattern_length):
        """Extract recurring patterns from time series"""
        patterns = []
        
        # Extract yearly patterns
        for i in range(len(series) - pattern_length + 1):
            pattern = series[i:i + pattern_length]
            patterns.append(pattern)
            
        return np.array(patterns)
    
    def fit(self, data, target_columns):
        """Fit Fast Deep AR models"""
        self.target_columns = target_columns
        
        for col in target_columns:
            if col in data.columns:
                print(f"Training Fast Deep AR for {col}...")
                
                series = data[col].dropna().values
                
                try:
                    # Extract patterns
                    patterns = self.extract_patterns(series, self.pattern_length)
                    
                    if len(patterns) > 2:
                        # Use last few patterns as templates
                        recent_patterns = patterns[-5:] if len(patterns) >= 5 else patterns
                        avg_pattern = np.mean(recent_patterns, axis=0)
                        
                        # Store pattern and series info
                        self.patterns[col] = {
                            'avg_pattern': avg_pattern,
                            'recent_patterns': recent_patterns,
                            'last_values': series[-self.pattern_length:],
                            'trend': (series[-1] - series[-365]) / 365 if len(series) >= 365 else 0
                        }
                        
                        print(f"  ✓ Fast Deep AR trained for {col}")
                    else:
                        print(f"  ✗ Not enough patterns for {col}")
                        
                except Exception as e:
                    print(f"  ✗ Error training Fast Deep AR for {col}: {e}")
                    
        self.fitted = True
        
    def predict_50_years(self, data, target_columns):
        """Predict 50 years using pattern repetition"""
        return self.predict_n_years(data, target_columns, 18250)
    
    def predict_n_years(self, data, target_columns, n_days):
        """Predict n days using pattern repetition"""
        predictions = {}
        n_years = n_days / 365
        
        for col in target_columns:
            if col in self.patterns:
                print(f"Fast Deep AR predicting {n_years:.1f} years for {col}...")
                
                pattern_info = self.patterns[col]
                avg_pattern = pattern_info['avg_pattern']
                trend = pattern_info['trend']
                
                # Calculate how many full patterns we need
                full_patterns = n_days // self.pattern_length
                remainder = n_days % self.pattern_length
                
                # Create base prediction by repeating pattern
                full_prediction = np.tile(avg_pattern, full_patterns)
                
                # Add remainder
                if remainder > 0:
                    remainder_pattern = avg_pattern[:remainder]
                    full_prediction = np.concatenate([full_prediction, remainder_pattern])
                
                # Add trend component
                trend_component = np.arange(n_days) * trend
                
                # Add some variation
                variation = np.random.normal(0, np.std(avg_pattern) * 0.1, n_days)
                
                # Combine components
                final_prediction = full_prediction + trend_component + variation
                
                predictions[col] = final_prediction
                print(f"  ✓ Fast prediction completed for {col}")
                
        return predictions
    
    def predict_test_period(self, data, test_length, target_columns):
        """Predict for test period"""
        predictions = {}
        
        for col in target_columns:
            if col in self.patterns:
                pattern_info = self.patterns[col]
                avg_pattern = pattern_info['avg_pattern']
                trend = pattern_info['trend']
                
                # Calculate patterns needed
                full_patterns = test_length // self.pattern_length
                remainder = test_length % self.pattern_length
                
                # Create prediction
                if full_patterns > 0:
                    full_prediction = np.tile(avg_pattern, full_patterns)
                else:
                    full_prediction = np.array([])
                
                if remainder > 0:
                    remainder_pattern = avg_pattern[:remainder]
                    if len(full_prediction) > 0:
                        full_prediction = np.concatenate([full_prediction, remainder_pattern])
                    else:
                        full_prediction = remainder_pattern
                
                # Add trend
                trend_component = np.arange(test_length) * trend
                final_prediction = full_prediction + trend_component
                
                predictions[col] = final_prediction
                
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