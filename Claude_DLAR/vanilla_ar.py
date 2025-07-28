import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class VanillaAutoRegressive:
    def __init__(self, order=5):
        self.order = order
        self.models = {}
        self.fitted = False
        
    def create_lagged_features(self, series, order):
        """Create lagged features for autoregressive model"""
        X = []
        y = []
        
        for i in range(order, len(series)):
            X.append(series[i-order:i])
            y.append(series[i])
            
        return np.array(X), np.array(y)
    
    def fit(self, data, target_columns):
        """Fit AR models for each target variable"""
        self.target_columns = target_columns
        
        for col in target_columns:
            if col in data.columns:
                series = data[col].values
                X, y = self.create_lagged_features(series, self.order)
                
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
        """Recursively predict multiple steps ahead"""
        predictions = []
        current_values = initial_values[-self.order:].copy()
        
        for _ in range(steps):
            next_pred = self.predict_single_step(current_values, target_col)
            predictions.append(next_pred)
            
            # Update current values for next prediction
            current_values = np.append(current_values[1:], next_pred)
            
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
                
                # Use last 'order' values as initial conditions
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