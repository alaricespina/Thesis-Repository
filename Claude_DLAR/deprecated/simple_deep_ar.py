import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SimpleDeepAR:
    """
    Simple Deep Learning AR using sklearn MLPRegressor
    Fallback when TensorFlow LSTM hangs or fails
    """
    def __init__(self, sequence_length=30, hidden_layers=(100, 50)):
        self.sequence_length = sequence_length
        self.hidden_layers = hidden_layers
        self.models = {}
        self.scalers = {}
        self.seasonal_means = {}
        self.fitted = False
        
    def create_sequences(self, data, sequence_length):
        """Create sequences for neural network training"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
            
        return np.array(X), np.array(y)
    
    def remove_seasonality(self, series, period=365):
        """Simple seasonal decomposition"""
        seasonal_means = {}
        deseasonalized = series.copy()
        
        for i in range(period):
            mask = np.arange(len(series)) % period == i
            if np.any(mask):
                seasonal_mean = series[mask].mean()
                seasonal_means[i] = seasonal_mean
                deseasonalized[mask] -= seasonal_mean
            else:
                seasonal_means[i] = 0
                
        return deseasonalized, seasonal_means
    
    def fit(self, data, target_columns):
        """Fit Simple Deep AR models"""
        self.target_columns = target_columns
        
        for col in target_columns:
            if col in data.columns:
                print(f"Training Simple Deep AR for {col}...")
                
                series = data[col].dropna().values
                
                # Limit data for faster training
                if len(series) > 1000:
                    series = series[-1000:]
                    print(f"  Using last {len(series)} data points")
                
                try:
                    # Remove seasonality
                    deseasonalized, seasonal_means = self.remove_seasonality(series)
                    self.seasonal_means[col] = seasonal_means
                    
                    # Scale data
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(deseasonalized.reshape(-1, 1)).flatten()
                    self.scalers[col] = scaler
                    
                    # Create sequences
                    X, y = self.create_sequences(scaled_data, self.sequence_length)
                    
                    if len(X) > 10:
                        # Train MLP model
                        model = MLPRegressor(
                            hidden_layer_sizes=self.hidden_layers,
                            max_iter=200,
                            early_stopping=True,
                            validation_fraction=0.2,
                            n_iter_no_change=10,
                            random_state=42
                        )
                        
                        model.fit(X, y)
                        
                        self.models[col] = {
                            'model': model,
                            'last_sequence': scaled_data[-self.sequence_length:],
                            'original_series': series
                        }
                        
                        print(f"  ✓ Simple Deep AR trained for {col}")
                    else:
                        print(f"  ✗ Not enough data for {col}")
                        
                except Exception as e:
                    print(f"  ✗ Error training Simple Deep AR for {col}: {e}")
                    
        self.fitted = True
        
    def predict_recursive(self, model, last_sequence, steps):
        """Recursively predict using MLP"""
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            next_pred = model.predict([current_sequence])[0]
            predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)
            
        return np.array(predictions)
    
    def add_seasonality(self, predictions, col, start_idx=0):
        """Add back seasonal component"""
        if col not in self.seasonal_means:
            return predictions
            
        seasonal_component = []
        period = len(self.seasonal_means[col])
        
        for i in range(len(predictions)):
            season_idx = (start_idx + i) % period
            seasonal_component.append(self.seasonal_means[col][season_idx])
            
        return predictions + np.array(seasonal_component)
    
    def predict_50_years(self, data, target_columns):
        """Predict 50 years ahead"""
        return self.predict_n_years(data, target_columns, 18250)
    
    def predict_n_years(self, data, target_columns, n_days):
        """Predict n days ahead"""
        predictions = {}
        n_years = n_days / 365
        
        for col in target_columns:
            if col in self.models:
                print(f"Simple Deep AR predicting {n_years:.1f} years for {col}...")
                
                model = self.models[col]['model']
                last_sequence = self.models[col]['last_sequence']
                
                # Predict on scaled data
                scaled_predictions = self.predict_recursive(model, last_sequence, n_days)
                
                # Inverse scale
                scaler = self.scalers[col]
                unscaled_predictions = scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()
                
                # Add seasonality back
                final_predictions = self.add_seasonality(unscaled_predictions, col)
                
                predictions[col] = final_predictions
                
        return predictions
    
    def predict_test_period(self, data, test_length, target_columns):
        """Predict for test period"""
        predictions = {}
        
        for col in target_columns:
            if col in self.models:
                model = self.models[col]['model']
                last_sequence = self.models[col]['last_sequence']
                
                scaled_predictions = self.predict_recursive(model, last_sequence, test_length)
                
                scaler = self.scalers[col]
                unscaled_predictions = scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).flatten()
                
                final_predictions = self.add_seasonality(unscaled_predictions, col)
                
                predictions[col] = final_predictions
                
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