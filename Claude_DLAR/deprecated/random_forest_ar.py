import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import time
import os
import pickle
import json

class RandomForestAutoRegressive:
    def __init__(self, sequence_length=60, n_estimators=100, max_depth=20, random_state=42, batch_size_pred=1000):
        self.sequence_length = sequence_length
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.batch_size_pred = batch_size_pred
        self.models = {}
        self.trend_models = {}  # For trend modeling
        self.scalers = {}
        self.seasonal_components = {}
        self.trend_components = {}  # Store trend information
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
    
    def extract_trend_and_seasonal(self, series, period=365):
        """Extract trend and seasonal components using decomposition"""
        print(f"       🔄 Extracting trend and seasonal components...")
        
        # Use rolling mean for trend
        trend = series.rolling(window=period//4, center=True).mean()
        
        # Fill NaN values at the beginning and end
        trend = trend.fillna(method='bfill').fillna(method='ffill')
        
        # Calculate seasonal component
        detrended = series - trend
        seasonal_means = detrended.groupby(detrended.index % period).mean()
        seasonal = np.tile(seasonal_means.values, len(series) // period + 1)[:len(series)]
        
        # Calculate residuals (what's left after removing trend and seasonality)
        residuals = series - trend - seasonal
        
        return trend, seasonal, residuals
    
    def create_time_features(self, dates, values):
        """Create comprehensive time-based features"""
        print(f"       🛠️ Creating time features...")
        
        features = []
        feature_names = []
        
        # Convert to datetime if needed
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.to_datetime(dates)
        
        n_samples = len(values)
        
        # Basic time features
        day_of_year = dates.dayofyear.values
        day_of_week = dates.dayofweek.values
        month = dates.month.values
        
        # Cyclical encoding for time features
        features.extend([
            np.sin(2 * np.pi * day_of_year / 365.25),  # Annual cycle
            np.cos(2 * np.pi * day_of_year / 365.25),
            np.sin(2 * np.pi * day_of_week / 7),       # Weekly cycle
            np.cos(2 * np.pi * day_of_week / 7),
            np.sin(2 * np.pi * month / 12),            # Monthly cycle
            np.cos(2 * np.pi * month / 12)
        ])
        
        feature_names.extend([
            'day_of_year_sin', 'day_of_year_cos',
            'day_of_week_sin', 'day_of_week_cos', 
            'month_sin', 'month_cos'
        ])
        
        return np.column_stack(features), feature_names
    
    def create_lag_features(self, values, max_lags=30):
        """Create lag features and rolling statistics"""
        print(f"       📊 Creating lag features (max_lags={max_lags})...")
        
        features = []
        feature_names = []
        
        # Basic lag features
        lag_features = [1, 2, 3, 7, 14, 30]  # 1 day, 2 days, 3 days, 1 week, 2 weeks, 1 month
        for lag in lag_features:
            if lag < len(values):
                lagged = np.concatenate([np.full(lag, np.nan), values[:-lag]])
                features.append(lagged)
                feature_names.append(f'lag_{lag}')
        
        # Rolling statistics
        windows = [3, 7, 14, 30]
        for window in windows:
            if window < len(values):
                # Rolling mean
                rolling_mean = pd.Series(values).rolling(window=window, min_periods=1).mean().values
                features.append(rolling_mean)
                feature_names.append(f'rolling_mean_{window}')
                
                # Rolling std
                rolling_std = pd.Series(values).rolling(window=window, min_periods=1).std().fillna(0).values
                features.append(rolling_std)
                feature_names.append(f'rolling_std_{window}')
                
                # Rolling min/max
                rolling_min = pd.Series(values).rolling(window=window, min_periods=1).min().values
                rolling_max = pd.Series(values).rolling(window=window, min_periods=1).max().values
                features.append(rolling_min)
                features.append(rolling_max)
                feature_names.append(f'rolling_min_{window}')
                feature_names.append(f'rolling_max_{window}')
        
        # Differences
        diff_1 = np.concatenate([np.array([0]), np.diff(values)])
        diff_7 = np.concatenate([np.zeros(7), values[7:] - values[:-7]])
        features.extend([diff_1, diff_7])
        feature_names.extend(['diff_1', 'diff_7'])
        
        return np.column_stack(features), feature_names
    
    def create_feature_matrix(self, series, residuals):
        """Create comprehensive feature matrix for RandomForest"""
        print(f"       🏗️ Building feature matrix...")
        
        dates = series.index
        values = residuals.values
        
        # Time-based features
        time_features, time_names = self.create_time_features(dates, values)
        
        # Lag features
        lag_features, lag_names = self.create_lag_features(values)
        
        # Combine all features
        all_features = np.column_stack([time_features, lag_features])
        all_names = time_names + lag_names
        
        # Remove rows with NaN values
        valid_rows = ~np.isnan(all_features).any(axis=1)
        clean_features = all_features[valid_rows]
        clean_targets = values[valid_rows]
        clean_dates = dates[valid_rows]
        
        print(f"          ✅ Created {clean_features.shape[1]} features")
        print(f"          ✅ {clean_features.shape[0]} valid samples (removed {np.sum(~valid_rows)} NaN rows)")
        
        return clean_features, clean_targets, clean_dates, all_names
    
    def create_sequences(self, data, sequence_length):
        """Create sequences for RandomForest training"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
            
        return np.array(X), np.array(y)
    
    def build_rf_model(self):
        """Build RandomForest model"""
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        return model
    
    def save_model_components(self, col):
        """Save model, scaler, and seasonal components for a variable"""
        if col in self.models:
            # Save RandomForest model
            model_path = os.path.join(self.model_dir, f"{col}_rf_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[col], f)
            
            # Save trend model
            if col in self.trend_models:
                trend_model_path = os.path.join(self.model_dir, f"{col}_trend_model.pkl")
                with open(trend_model_path, 'wb') as f:
                    pickle.dump(self.trend_models[col], f)
            
            # Save scaler
            scaler_path = os.path.join(self.model_dir, f"{col}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[col], f)
            
            # Save seasonal components
            seasonal_path = os.path.join(self.model_dir, f"{col}_seasonal.pkl")
            with open(seasonal_path, 'wb') as f:
                pickle.dump(self.seasonal_components[col], f)
            
            # Save trend components
            if col in self.trend_components:
                trend_path = os.path.join(self.model_dir, f"{col}_trend.pkl")
                with open(trend_path, 'wb') as f:
                    pickle.dump(self.trend_components[col], f)
                
            print(f"       💾 Model components saved for {col}")
            
    def load_model_components(self, col):
        """Load model, scaler, and seasonal components for a variable"""
        model_path = os.path.join(self.model_dir, f"{col}_rf_model.pkl")
        trend_model_path = os.path.join(self.model_dir, f"{col}_trend_model.pkl")
        scaler_path = os.path.join(self.model_dir, f"{col}_scaler.pkl")
        seasonal_path = os.path.join(self.model_dir, f"{col}_seasonal.pkl")
        trend_path = os.path.join(self.model_dir, f"{col}_trend.pkl")
        
        required_paths = [model_path, scaler_path, seasonal_path]
        if all(os.path.exists(p) for p in required_paths):
            try:
                # Load RandomForest model
                with open(model_path, 'rb') as f:
                    self.models[col] = pickle.load(f)
                
                # Load trend model if exists
                if os.path.exists(trend_model_path):
                    with open(trend_model_path, 'rb') as f:
                        self.trend_models[col] = pickle.load(f)
                
                # Load scaler
                with open(scaler_path, 'rb') as f:
                    self.scalers[col] = pickle.load(f)
                
                # Load seasonal components
                with open(seasonal_path, 'rb') as f:
                    self.seasonal_components[col] = pickle.load(f)
                
                # Load trend components if exists
                if os.path.exists(trend_path):
                    with open(trend_path, 'rb') as f:
                        self.trend_components[col] = pickle.load(f)
                    
                print(f"       📁 Model components loaded for {col}")
                return True
            except Exception as e:
                print(f"       ❌ Error loading model for {col}: {e}")
                return False
        return False
    
    def model_exists(self, col):
        """Check if saved model exists for a variable"""
        model_path = os.path.join(self.model_dir, f"{col}_rf_model.pkl")
        scaler_path = os.path.join(self.model_dir, f"{col}_scaler.pkl")
        seasonal_path = os.path.join(self.model_dir, f"{col}_seasonal.pkl")
        return all(os.path.exists(p) for p in [model_path, scaler_path, seasonal_path])
    
    def fit_trend_model(self, dates, trend_values):
        """Fit a linear trend model"""
        # Convert dates to numeric (days since start)
        start_date = dates[0]
        days_since_start = np.array([(d - start_date).days for d in dates]).reshape(-1, 1)
        
        # Fit linear regression for trend
        trend_model = LinearRegression()
        trend_model.fit(days_since_start, trend_values)
        
        return trend_model
    
    def fit_single_model(self, features, targets, col):
        """Fit RandomForest model on engineered features"""
        try:
            # Build and train model
            model = self.build_rf_model()
            
            print(f"       🌲 Training RandomForest model on {features.shape[1]} features...")
            model.fit(features, targets)
            
            return model
            
        except Exception as e:
            print(f"       ❌ Training failed for {col}: {e}")
            return None

    def fit(self, data, target_columns, force_retrain=False):
        """Fit RandomForest AR models using residual modeling architecture"""
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
                
                print(f"  🌲 Training new residual model for {col}...")
                series = data[col].dropna()
                
                # Limit data size for faster training (last 2 years)
                if len(series) > 730:
                    series = series.tail(730)
                    print(f"       Using last {len(series)} data points for faster training")
                
                try:
                    # Step 1: Decompose into trend, seasonal, and residuals
                    trend, seasonal, residuals = self.extract_trend_and_seasonal(series)
                    
                    # Step 2: Fit trend model
                    print(f"       📈 Fitting trend model...")
                    trend_model = self.fit_trend_model(series.index, trend)
                    self.trend_models[col] = trend_model
                    
                    # Store components for reconstruction
                    self.trend_components[col] = {
                        'trend': trend,
                        'start_date': series.index[0]
                    }
                    
                    self.seasonal_components[col] = {
                        'seasonal': seasonal,
                        'original_series': series,
                        'period': 365
                    }
                    
                    # Step 3: Create feature matrix for residuals
                    residual_series = pd.Series(residuals, index=series.index)
                    features, targets, clean_dates, feature_names = self.create_feature_matrix(series, residual_series)
                    
                    if len(features) < 50:  # Need minimum samples
                        print(f"       ❌ Not enough clean samples for {col}: {len(features)}")
                        continue
                    
                    print(f"       📊 Scaling residual features...")
                    # Scale the features
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    scaled_features = scaler.fit_transform(features)
                    self.scalers[col] = scaler
                    
                    # Step 4: Train RandomForest on residuals
                    model = self.fit_single_model(scaled_features, targets, col)
                    
                    if model is not None:
                        self.models[col] = model
                        print(f"       ✅ Residual model training completed")
                        print(f"       📊 Feature importance (top 5):")
                        
                        # Show feature importance
                        importances = model.feature_importances_
                        top_indices = np.argsort(importances)[-5:][::-1]
                        for idx in top_indices:
                            print(f"          {feature_names[idx]}: {importances[idx]:.4f}")
                        
                        # Save the trained model
                        self.save_model_components(col)
                        print(f"  ✅ Residual model trained and saved for {col}")
                    else:
                        print(f"  ❌ Failed to train residual model for {col}")
                        
                except Exception as e:
                    print(f"  ❌ Error processing {col}: {e}")
                    import traceback
                    traceback.print_exc()
                    
        self.fitted = True
    
    def predict_residuals_recursive(self, model, scaler, last_known_residuals, last_known_values, start_date, steps):
        """Recursively predict residuals using feature engineering"""
        print(f"    🔮 Predicting {steps} residual steps...")
        
        predictions = []
        
        # Initialize with last known values for feature creation
        current_residuals = list(last_known_residuals)
        current_values = list(last_known_values)
        current_date = start_date
        
        # For very long predictions, use batches
        if steps > 1000:
            batch_size = self.batch_size_pred
            print(f"    Predicting {steps} steps in batches of {batch_size}...")
            
            for batch_start in range(0, steps, batch_size):
                batch_end = min(batch_start + batch_size, steps)
                batch_steps = batch_end - batch_start
                
                print(f"    📦 Processing batch {batch_start//batch_size + 1}/{(steps-1)//batch_size + 1}")
                
                batch_predictions = []
                for i in range(batch_steps):
                    # Create features for current time step
                    feature_vector = self.create_prediction_features(
                        current_residuals, current_values, current_date
                    )
                    
                    # Scale features
                    scaled_features = scaler.transform(feature_vector.reshape(1, -1))
                    
                    # Predict next residual
                    next_residual = model.predict(scaled_features)[0]
                    
                    batch_predictions.append(next_residual)
                    
                    # Update for next iteration
                    current_residuals.append(next_residual)
                    current_values.append(next_residual)  # Simplified - would normally add trend+seasonal
                    current_date += pd.Timedelta(days=1)
                    
                    # Keep only recent history for efficiency
                    if len(current_residuals) > 100:
                        current_residuals.pop(0)
                        current_values.pop(0)
                
                predictions.extend(batch_predictions)
                
                # Progress update
                progress = (batch_end / steps) * 100
                print(f"    ✅ Batch completed! Progress: {progress:.1f}% ({batch_end}/{steps})")
                
        else:
            for i in range(steps):
                # Create features for current time step
                feature_vector = self.create_prediction_features(
                    current_residuals, current_values, current_date
                )
                
                # Scale features
                scaled_features = scaler.transform(feature_vector.reshape(1, -1))
                
                # Predict next residual
                next_residual = model.predict(scaled_features)[0]
                
                predictions.append(next_residual)
                
                # Update for next iteration
                current_residuals.append(next_residual)
                current_values.append(next_residual)  # Simplified
                current_date += pd.Timedelta(days=1)
                
                # Keep only recent history
                if len(current_residuals) > 100:
                    current_residuals.pop(0)
                    current_values.pop(0)
            
        return np.array(predictions)
    
    def create_prediction_features(self, residual_history, value_history, current_date):
        """Create features for a single prediction step"""
        # Use recent history for features
        recent_residuals = np.array(residual_history[-50:])  # Last 50 days
        recent_values = np.array(value_history[-50:])
        
        # Time features
        day_of_year = current_date.dayofyear
        day_of_week = current_date.dayofweek
        month = current_date.month
        
        # Cyclical encoding
        time_features = [
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12)
        ]
        
        # Lag features (if enough history)
        lag_features = []
        for lag in [1, 2, 3, 7, 14, 30]:
            if len(recent_residuals) > lag:
                lag_features.append(recent_residuals[-lag])
            else:
                lag_features.append(0.0)
        
        # Rolling statistics
        if len(recent_residuals) >= 3:
            lag_features.extend([
                np.mean(recent_residuals[-3:]),   # 3-day mean
                np.std(recent_residuals[-3:]),    # 3-day std
                np.mean(recent_residuals[-7:]) if len(recent_residuals) >= 7 else np.mean(recent_residuals),
                np.std(recent_residuals[-7:]) if len(recent_residuals) >= 7 else np.std(recent_residuals),
                np.mean(recent_residuals[-14:]) if len(recent_residuals) >= 14 else np.mean(recent_residuals),
                np.std(recent_residuals[-14:]) if len(recent_residuals) >= 14 else np.std(recent_residuals),
                np.mean(recent_residuals[-30:]) if len(recent_residuals) >= 30 else np.mean(recent_residuals),
                np.std(recent_residuals[-30:]) if len(recent_residuals) >= 30 else np.std(recent_residuals),
                np.min(recent_residuals[-3:]),
                np.max(recent_residuals[-3:]),
                np.min(recent_residuals[-7:]) if len(recent_residuals) >= 7 else np.min(recent_residuals),
                np.max(recent_residuals[-7:]) if len(recent_residuals) >= 7 else np.max(recent_residuals),
                np.min(recent_residuals[-14:]) if len(recent_residuals) >= 14 else np.min(recent_residuals),
                np.max(recent_residuals[-14:]) if len(recent_residuals) >= 14 else np.max(recent_residuals),
                np.min(recent_residuals[-30:]) if len(recent_residuals) >= 30 else np.min(recent_residuals),
                np.max(recent_residuals[-30:]) if len(recent_residuals) >= 30 else np.max(recent_residuals)
            ])
        else:
            # Fill with zeros if not enough data
            lag_features.extend([0.0] * 16)
        
        # Differences
        if len(recent_residuals) >= 2:
            lag_features.extend([
                recent_residuals[-1] - recent_residuals[-2],  # 1-day diff
                recent_residuals[-1] - recent_residuals[-7] if len(recent_residuals) >= 7 else 0  # 7-day diff
            ])
        else:
            lag_features.extend([0.0, 0.0])
        
        # Combine all features
        all_features = time_features + lag_features
        
        return np.array(all_features)
    
    def reconstruct_predictions(self, residual_predictions, col, start_date, steps):
        """Reconstruct final predictions from residual predictions"""
        print(f"       🔄 Reconstructing final predictions...")
        
        # Get stored components
        trend_info = self.trend_components[col]
        seasonal_info = self.seasonal_components[col]
        
        # Generate trend for prediction period
        print(f"       📈 Generating trend component...")
        trend_model = self.trend_models[col]
        start_date_ref = trend_info['start_date']
        
        # Create date range for predictions
        prediction_dates = [start_date + pd.Timedelta(days=i) for i in range(steps)]
        days_since_start = np.array([(d - start_date_ref).days for d in prediction_dates]).reshape(-1, 1)
        trend_predictions = trend_model.predict(days_since_start)
        print(f"          ✅ Trend predicted for {steps} days")
        
        # Generate seasonal component
        print(f"       🔄 Generating seasonal component...")
        period = seasonal_info['period']
        seasonal_base = seasonal_info['seasonal'][:period]  # Get base seasonal pattern
        seasonal_predictions = np.tile(seasonal_base, steps // period + 1)[:steps]
        print(f"          ✅ Seasonal pattern (period={period}) tiled for {steps} days")
        
        # Combine all components
        print(f"       🔗 Combining trend + seasonal + residuals...")
        final_predictions = trend_predictions + seasonal_predictions + residual_predictions
        print(f"          ✅ All components combined")
        
        # Add small noise for realism
        print(f"       🎲 Adding white noise for realism...")
        original_series = seasonal_info['original_series']
        noise_std = np.std(original_series.diff().dropna()) * 0.05  # Small noise
        white_noise = np.random.normal(0, noise_std, steps)
        final_predictions += white_noise
        print(f"          ✅ White noise added (std={noise_std:.4f})")
        
        # Ensure predictions are reasonable (basic sanity check)
        original_min, original_max = original_series.min(), original_series.max()
        original_range = original_max - original_min
        
        # Clip extreme values
        lower_bound = original_min - original_range * 0.5
        upper_bound = original_max + original_range * 0.5
        final_predictions = np.clip(final_predictions, lower_bound, upper_bound)
        
        return final_predictions
    
    def predict_50_years(self, data, target_columns):
        """Predict 50 years (18250 days) ahead for all target variables"""
        return self.predict_n_years(data, target_columns, 18250)
    
    def predict_n_years(self, data, target_columns, n_days):
        """Predict n days ahead for all target variables using residual modeling"""
        predictions = {}
        n_years = n_days / 365
        
        for col in target_columns:
            if col in self.models and col in self.scalers and col in self.trend_models:
                print(f"Predicting {n_years:.1f} years ({n_days} days) for {col}...")
                
                try:
                    print(f"  🔄 Step 1/4: Preparing data for {col}...")
                    # Get the original series
                    series = data[col].dropna()
                    print(f"     Original series length: {len(series)} days")
                    
                    # Decompose the series to get residuals
                    trend, seasonal, residuals = self.extract_trend_and_seasonal(series)
                    
                    # Get last known residuals and values for prediction
                    last_residuals = residuals[-50:].tolist()  # Last 50 residuals
                    last_values = series[-50:].values.tolist()  # Last 50 original values
                    start_date = series.index[-1] + pd.Timedelta(days=1)  # Start from next day
                    
                    print(f"     Using last {len(last_residuals)} residuals as seed")
                    print(f"     Prediction start date: {start_date}")
                    
                    print(f"  🔄 Step 2/4: Predicting residuals for {col}...")
                    print(f"     Target: {n_days} days ({n_years:.1f} years) of residual predictions")
                    
                    # Predict residuals using RandomForest
                    residual_predictions = self.predict_residuals_recursive(
                        self.models[col], self.scalers[col], 
                        last_residuals, last_values, start_date, n_days
                    )
                    print(f"     ✅ Residual predictions generated: {len(residual_predictions)} values")
                    
                    print(f"  🔄 Step 3/4: Reconstructing final predictions for {col}...")
                    # Reconstruct final predictions by adding trend and seasonal
                    final_predictions = self.reconstruct_predictions(
                        residual_predictions, col, start_date, n_days
                    )
                    print(f"     ✅ Final predictions reconstructed")
                    predictions[col] = final_predictions
                    
                    print(f"  ✅ {n_years:.1f}-year prediction completed for {col}")
                    print(f"     Prediction range: [{final_predictions.min():.3f}, {final_predictions.max():.3f}]")
                    
                except Exception as e:
                    print(f"  ❌ Error predicting {n_years:.1f} years for {col}: {e}")
                    import traceback
                    traceback.print_exc()
                    predictions[col] = np.zeros(n_days)
            else:
                print(f"  ❌ Missing components for {col} - skipping")
                predictions[col] = np.zeros(n_days)
                
        return predictions
    
    def predict_test_period(self, data, test_length, target_columns):
        """Predict for test period using residual modeling"""
        predictions = {}
        
        for col in target_columns:
            if col in self.models and col in self.scalers and col in self.trend_models:
                try:
                    print(f"  🔮 Predicting test period for {col} ({test_length} days)...")
                    
                    # Get the original series
                    series = data[col].dropna()
                    
                    # Decompose the series to get residuals
                    trend, seasonal, residuals = self.extract_trend_and_seasonal(series)
                    
                    # Get last known residuals and values for prediction
                    last_residuals = residuals[-50:].tolist()  # Last 50 residuals
                    last_values = series[-50:].values.tolist()  # Last 50 original values
                    start_date = series.index[-1] + pd.Timedelta(days=1)  # Start from next day
                    
                    # Predict residuals using RandomForest
                    residual_predictions = self.predict_residuals_recursive(
                        self.models[col], self.scalers[col], 
                        last_residuals, last_values, start_date, test_length
                    )
                    
                    # Reconstruct final predictions
                    final_predictions = self.reconstruct_predictions(
                        residual_predictions, col, start_date, test_length
                    )
                    predictions[col] = final_predictions
                    
                    print(f"     ✅ Test predictions completed for {col}")
                    
                except Exception as e:
                    print(f"     ❌ Error predicting test period for {col}: {e}")
                    predictions[col] = np.zeros(test_length)
            else:
                print(f"     ❌ Missing components for {col} - using zeros")
                predictions[col] = np.zeros(test_length)
                
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