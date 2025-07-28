import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import signal
import time
import os
import pickle
import json

# Disable GPU to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

class DeepLearningAutoRegressive:
    def __init__(self, sequence_length=60, lstm_units=50, dropout_rate=0.2, batch_size_pred=1000):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.batch_size_pred = batch_size_pred  # Configurable batch size for predictions
        self.models = {}
        self.scalers = {}
        self.seasonal_components = {}
        self.fitted = False
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def seasonal_decompose_simple(self, series, period=365):
        """Simple seasonal decomposition using moving averages"""
        # Calculate trend using centered moving average
        trend = series.rolling(window=period, center=True).mean()
        
        # Calculate seasonal component
        detrended = series - trend
        seasonal = detrended.groupby(detrended.index % period).mean()
        seasonal_full = np.tile(seasonal.values, len(series) // period + 1)[:len(series)]
        
        # Calculate residual (stationary component)
        residual = series - trend - seasonal_full
        
        return trend, seasonal_full, residual
    
    def deseasonalize(self, series, period=365):
        """Remove seasonal component through differencing and decomposition"""
        # First, apply seasonal differencing
        seasonal_diff = series.diff(period).dropna()
        
        # Store the seasonal pattern for later reconstruction
        seasonal_pattern = series.iloc[:period].values
        
        # Apply additional first-order differencing to make it stationary
        stationary = seasonal_diff.diff().dropna()
        
        return stationary, seasonal_pattern, seasonal_diff
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for LSTM training"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
            
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def save_model_components(self, col):
        """Save model, scaler, and seasonal components for a variable"""
        if col in self.models:
            # Save Keras model
            model_path = os.path.join(self.model_dir, f"{col}_lstm_model.keras")
            self.models[col].save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"{col}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[col], f)
            
            # Save seasonal components
            seasonal_path = os.path.join(self.model_dir, f"{col}_seasonal.pkl")
            with open(seasonal_path, 'wb') as f:
                pickle.dump(self.seasonal_components[col], f)
                
            print(f"       💾 Model components saved for {col}")
            
    def load_model_components(self, col):
        """Load model, scaler, and seasonal components for a variable"""
        model_path = os.path.join(self.model_dir, f"{col}_lstm_model.keras")
        scaler_path = os.path.join(self.model_dir, f"{col}_scaler.pkl")
        seasonal_path = os.path.join(self.model_dir, f"{col}_seasonal.pkl")
        
        if all(os.path.exists(p) for p in [model_path, scaler_path, seasonal_path]):
            try:
                # Load Keras model
                self.models[col] = load_model(model_path)
                
                # Load scaler
                with open(scaler_path, 'rb') as f:
                    self.scalers[col] = pickle.load(f)
                
                # Load seasonal components
                with open(seasonal_path, 'rb') as f:
                    self.seasonal_components[col] = pickle.load(f)
                    
                print(f"       📁 Model components loaded for {col}")
                return True
            except Exception as e:
                print(f"       ❌ Error loading model for {col}: {e}")
                return False
        return False
    
    def model_exists(self, col):
        """Check if saved model exists for a variable"""
        model_path = os.path.join(self.model_dir, f"{col}_lstm_model.keras")
        scaler_path = os.path.join(self.model_dir, f"{col}_scaler.pkl")
        seasonal_path = os.path.join(self.model_dir, f"{col}_seasonal.pkl")
        return all(os.path.exists(p) for p in [model_path, scaler_path, seasonal_path])
    
    def fit_single_model(self, X, y, col, epochs=50, batch_size=32, validation_split=0.2):
        """Fit model without timeout"""
        try:
            # Build and train model
            model = self.build_lstm_model((X.shape[1], 1))
            
            # Train with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            # Reduce epochs for problematic variables
            if col == 'precip':
                epochs = min(epochs, 30)  # Limit epochs for precipitation
                print(f"       Using reduced epochs ({epochs}) for precipitation due to sparsity")
            
            print(f"       🏋️ Training LSTM model ({epochs} epochs max)...")
            history = model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
            
            return model, history
            
        except Exception as e:
            print(f"       ❌ Training failed for {col}: {e}")
            return None, None

    def fit(self, data, target_columns, epochs=50, batch_size=32, validation_split=0.2, force_retrain=False):
        """Fit deep learning AR models for each target variable"""
        self.target_columns = target_columns
        
        for col in target_columns:
            if col in data.columns:
                print(f"Processing model for {col}...")
                
                # Check if model already exists
                if not force_retrain and self.model_exists(col):
                    print(f"  📁 Found existing model for {col}, loading...")
                    if self.load_model_components(col):
                        print(f"  ✅ Model loaded successfully for {col}")
                        continue
                    else:
                        print(f"  ⚠️ Failed to load existing model, will retrain...")
                
                print(f"  🏋️ Training new model for {col}...")
                series = data[col].dropna()
                
                # Limit data size for faster training (last 2 years)
                if len(series) > 730:
                    series = series.tail(730)
                    print(f"       Using last {len(series)} data points for faster training")
                
                try:
                    print(f"       🔄 Deseasonalizing data...")
                    # Deseasonalize the data
                    stationary_data, seasonal_pattern, seasonal_diff = self.deseasonalize(series)
                    
                    if len(stationary_data) < self.sequence_length + 10:
                        print(f"       ❌ Not enough stationary data for {col}")
                        continue
                    
                    # Store seasonal components for reconstruction
                    self.seasonal_components[col] = {
                        'seasonal_pattern': seasonal_pattern,
                        'original_series': series,
                        'seasonal_diff': seasonal_diff
                    }
                    
                    print(f"       📊 Scaling data...")
                    # Scale the stationary data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(stationary_data.values.reshape(-1, 1))
                    self.scalers[col] = scaler
                    
                    print(f"       🔗 Creating sequences...")
                    # Create sequences
                    X, y = self.create_sequences(scaled_data.flatten(), self.sequence_length)
                    print(f"       Created {len(X)} training sequences")
                    
                    if len(X) > 10:  # Need minimum samples
                        # Reshape for LSTM [samples, time steps, features]
                        X = X.reshape((X.shape[0], X.shape[1], 1))
                        
                        # Train model
                        model, history = self.fit_single_model(
                            X, y, col, epochs, batch_size, validation_split
                        )
                        
                        if model is not None and history is not None:
                            self.models[col] = model
                            final_loss = history.history['loss'][-1] if history.history['loss'] else 0
                            print(f"       ✅ Training completed - Final loss: {final_loss:.4f}")
                            
                            # Save the trained model
                            self.save_model_components(col)
                            print(f"  ✅ Model trained and saved for {col}")
                        else:
                            print(f"  ❌ Failed to train model for {col}")
                    else:
                        print(f"  ❌ Not enough sequences created for {col}")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {col}: {e}")
                    
        self.fitted = True
    
    def predict_recursive_stationary(self, model, scaler, last_sequence, steps):
        """Recursively predict on stationary data with progress tracking"""
        predictions = []
        current_sequence = last_sequence.copy()
        
        # For very long predictions, use batch prediction to speed up
        if steps > 1000:
            batch_size = self.batch_size_pred
            print(f"    Predicting {steps} steps in batches of {batch_size}...")
            print(f"    Each batch processes {batch_size} individual predictions")
            
            for batch_start in range(0, steps, batch_size):
                batch_end = min(batch_start + batch_size, steps)
                batch_steps = batch_end - batch_start
                
                print(f"    📦 Processing batch {batch_start//batch_size + 1}/{(steps-1)//batch_size + 1}")
                print(f"       Batch range: {batch_start} to {batch_end-1} ({batch_steps} predictions)")
                
                batch_predictions = []
                for i in range(batch_steps):
                    # Show progress within batch every 100 steps
                    if i > 0 and i % 100 == 0:
                        batch_progress = (i / batch_steps) * 100
                        print(f"       Within batch: {batch_progress:.1f}% ({i}/{batch_steps})")
                    
                    next_pred = model.predict(current_sequence.reshape(1, self.sequence_length, 1), verbose=0)[0, 0]
                    batch_predictions.append(next_pred)
                    current_sequence = np.append(current_sequence[1:], next_pred)
                
                predictions.extend(batch_predictions)
                
                # Overall progress update
                progress = (batch_end / steps) * 100
                print(f"    ✅ Batch {batch_start//batch_size + 1} completed!")
                print(f"    📊 Overall Progress: {progress:.1f}% ({batch_end}/{steps})")
                print(f"    ⏱️  Predictions completed so far: {len(predictions)}")
                
                # Estimated time remaining (rough estimate)
                if batch_end > batch_size:  # After first batch
                    batches_completed = batch_end // batch_size
                    total_batches = (steps - 1) // batch_size + 1
                    batches_remaining = total_batches - batches_completed
                    print(f"    ⏳ Estimated batches remaining: {batches_remaining}")
                
                print()  # Empty line for readability
                
        else:
            for _ in range(steps):
                next_pred = model.predict(current_sequence.reshape(1, self.sequence_length, 1), verbose=0)[0, 0]
                predictions.append(next_pred)
                current_sequence = np.append(current_sequence[1:], next_pred)
            
        return np.array(predictions)
    
    def reseasonalize(self, stationary_predictions, col, steps):
        """Add back seasonal component and noise to get final predictions"""
        print(f"       🔄 Reseasonalization substeps:")
        
        seasonal_info = self.seasonal_components[col]
        seasonal_pattern = seasonal_info['seasonal_pattern']
        original_series = seasonal_info['original_series']
        
        print(f"       📈 Inverse scaling stationary predictions...")
        # Inverse transform stationary predictions
        scaler = self.scalers[col]
        stationary_pred_scaled = scaler.inverse_transform(stationary_predictions.reshape(-1, 1)).flatten()
        print(f"          ✅ {len(stationary_pred_scaled)} values inverse scaled")
        
        print(f"       🔄 Reconstructing seasonal component...")
        # Reconstruct seasonal component
        period = len(seasonal_pattern)
        seasonal_component = np.tile(seasonal_pattern, steps // period + 1)[:steps]
        print(f"          ✅ Seasonal pattern (period={period}) tiled for {steps} days")
        
        print(f"       📊 Adding trend component...")
        # Add trend (simple linear extrapolation from last known values)
        last_values = original_series.tail(period).values
        trend_slope = (last_values[-1] - last_values[0]) / period
        trend_component = np.arange(steps) * trend_slope + last_values[-1]
        print(f"          ✅ Trend slope: {trend_slope:.6f} per day")
        
        print(f"       🔗 Combining all components...")
        # Combine components
        final_predictions = stationary_pred_scaled + seasonal_component + trend_component
        print(f"          ✅ Stationary + Seasonal + Trend combined")
        
        print(f"       🎲 Adding white noise for realism...")
        # Add white noise for realism
        noise_std = np.std(original_series.diff().dropna()) * 0.1  # Small noise
        white_noise = np.random.normal(0, noise_std, steps)
        final_predictions += white_noise
        print(f"          ✅ White noise added (std={noise_std:.4f})")
        
        return final_predictions
    
    def predict_50_years(self, data, target_columns):
        """Predict 50 years (18250 days) ahead for all target variables"""
        return self.predict_n_years(data, target_columns, 18250)
    
    def predict_n_years(self, data, target_columns, n_days):
        """Predict n days ahead for all target variables"""
        predictions = {}
        n_years = n_days / 365
        
        for col in target_columns:
            if col in self.models and col in self.scalers:
                print(f"Predicting {n_years:.1f} years ({n_days} days) for {col}...")
                
                try:
                    print(f"  🔄 Step 1/4: Preparing data for {col}...")
                    # Get the last sequence from training data
                    series = data[col].dropna()
                    print(f"     Original series length: {len(series)} days")
                    
                    print(f"  🔄 Step 2/4: Deseasonalizing data for {col}...")
                    stationary_data, _, _ = self.deseasonalize(series)
                    print(f"     Stationary data length: {len(stationary_data)} days")
                    
                    scaled_data = self.scalers[col].transform(stationary_data.values.reshape(-1, 1))
                    print(f"     Data scaled using stored scaler")
                    
                    # Get last sequence for prediction
                    last_sequence = scaled_data[-self.sequence_length:].flatten()
                    print(f"     Using last {self.sequence_length} values as seed sequence")
                    
                    print(f"  🔄 Step 3/4: Generating stationary predictions for {col}...")
                    print(f"     Target: {n_days} days ({n_years:.1f} years) of predictions")
                    # Predict n days on stationary data
                    stationary_predictions = self.predict_recursive_stationary(
                        self.models[col], self.scalers[col], last_sequence, n_days
                    )
                    print(f"     ✅ Stationary predictions generated: {len(stationary_predictions)} values")
                    
                    print(f"  🔄 Step 4/4: Reseasonalizing predictions for {col}...")
                    # Reseasonalize predictions
                    final_predictions = self.reseasonalize(stationary_predictions, col, n_days)
                    print(f"     ✅ Seasonal components and noise added")
                    predictions[col] = final_predictions
                    
                    print(f"  ✅ {n_years:.1f}-year prediction completed for {col}")
                    
                except Exception as e:
                    print(f"  ❌ Error predicting {n_years:.1f} years for {col}: {e}")
                    predictions[col] = np.zeros(n_days)
                
        return predictions
    
    def predict_test_period(self, data, test_length, target_columns):
        """Predict for test period"""
        predictions = {}
        
        for col in target_columns:
            if col in self.models and col in self.scalers:
                series = data[col].dropna()
                stationary_data, _, _ = self.deseasonalize(series)
                scaled_data = self.scalers[col].transform(stationary_data.values.reshape(-1, 1))
                
                last_sequence = scaled_data[-self.sequence_length:].flatten()
                
                stationary_predictions = self.predict_recursive_stationary(
                    self.models[col], self.scalers[col], last_sequence, test_length
                )
                
                final_predictions = self.reseasonalize(stationary_predictions, col, test_length)
                predictions[col] = final_predictions
                
        return predictions
    
    def evaluate(self, true_values, predictions):
        """Evaluate predictions using MSE and MAE"""
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