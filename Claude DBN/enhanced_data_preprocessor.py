import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import glob
import os
from datetime import datetime


class EnhancedWeatherDataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k='all')
        self.feature_columns = []
        
    def load_data(self):
        """Load all CSV files from the data directory"""
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        # Filter out specific files we don't want
        csv_files = [f for f in csv_files if not any(exclude in f for exclude in 
                    ["Concatenated", "PREDICTIONS", "Model Output", "Hourly"])]
        
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dataframes.append(df)
                print(f"Loaded {file}: {df.shape}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if dataframes:
            combined_data = pd.concat(dataframes, ignore_index=True)
            print(f"Combined dataset shape: {combined_data.shape}")
            return combined_data
        else:
            raise ValueError("No valid CSV files found in the specified directory")
    
    def clean_data(self, df):
        """Clean and preprocess the weather data"""
        # Drop unnecessary columns
        columns_to_drop = ['name', 'datetime', 'description', 'icon', 'stations', 
                          'sunrise', 'sunset']
        
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Handle missing values
        df = df.dropna(subset=['conditions'])  # Target variable must not be null
        
        # Fill numeric missing values with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Handle categorical missing values
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'conditions' and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown')
        
        return df
    
    def advanced_feature_engineering(self, df):
        """Create advanced features from existing data"""
        # Basic features from original implementation
        if 'tempmax' in df.columns and 'tempmin' in df.columns:
            df['temp_range'] = df['tempmax'] - df['tempmin']
            df['temp_variance'] = (df['tempmax'] - df['tempmin']) ** 2
        
        if 'feelslike' in df.columns and 'temp' in df.columns:
            df['feels_like_diff'] = df['feelslike'] - df['temp']
            df['feels_like_ratio'] = df['feelslike'] / (df['temp'] + 1e-8)
        
        if 'temp' in df.columns and 'dew' in df.columns:
            df['dew_point_depression'] = df['temp'] - df['dew']
            df['relative_humidity_calculated'] = 100 * np.exp((17.625 * df['dew']) / (243.04 + df['dew'])) / np.exp((17.625 * df['temp']) / (243.04 + df['temp']))
        
        # Advanced meteorological features
        if 'temp' in df.columns and 'windspeed' in df.columns:
            df['wind_chill_factor'] = df['temp'] - (df['windspeed'] * 0.1)
            df['apparent_temp'] = df['temp'] - 0.4 * (df['temp'] - 10) * (1 - df.get('humidity', 50)/100)
        
        if 'sealevelpressure' in df.columns:
            df['pressure_normalized'] = (df['sealevelpressure'] - 1013.25) / 1013.25
            df['pressure_anomaly'] = df['sealevelpressure'] - df['sealevelpressure'].mean()
        
        # Solar and visibility features
        if 'solarradiation' in df.columns and 'cloudcover' in df.columns:
            df['solar_efficiency'] = df['solarradiation'] / (100 - df['cloudcover'] + 1e-8)
            df['cloud_solar_interaction'] = df['cloudcover'] * df['solarradiation']
        
        if 'visibility' in df.columns and 'humidity' in df.columns:
            df['visibility_humidity_ratio'] = df['visibility'] / (df['humidity'] + 1e-8)
        
        # Weather intensity features
        if 'precipprob' in df.columns and 'precip' in df.columns:
            df['precip_intensity'] = df['precip'] * df['precipprob'] / 100
            df['precip_efficiency'] = df['precip'] / (df['precipprob'] + 1e-8)
        
        # Wind features
        if 'windspeed' in df.columns and 'windgust' in df.columns:
            df['wind_gust_ratio'] = df['windgust'] / (df['windspeed'] + 1e-8)
            df['wind_variability'] = df['windgust'] - df['windspeed']
        
        # Time-based features (if moonphase is available)
        if 'moonphase' in df.columns:
            df['moon_sin'] = np.sin(2 * np.pi * df['moonphase'])
            df['moon_cos'] = np.cos(2 * np.pi * df['moonphase'])
        
        # Temperature stability indicators
        if all(col in df.columns for col in ['tempmax', 'tempmin', 'temp']):
            df['temp_stability'] = 1 - (df['temp_range'] / (df['temp'] + 1e-8))
            df['temp_position'] = (df['temp'] - df['tempmin']) / (df['temp_range'] + 1e-8)
        
        # Comfort indices
        if all(col in df.columns for col in ['temp', 'humidity', 'windspeed']):
            # Heat Index approximation
            T = df['temp']
            H = df['humidity']
            df['heat_index'] = T + 0.5 * (T + 61.0 + ((T-68.0)*1.2) + (H*0.094))
            
            # Wind Chill approximation
            df['wind_chill'] = 35.74 + (0.6215 * T) - (35.75 * (df['windspeed'] ** 0.16)) + (0.4275 * T * (df['windspeed'] ** 0.16))
        
        # Atmospheric stability
        if all(col in df.columns for col in ['temp', 'sealevelpressure', 'windspeed']):
            df['atmospheric_stability'] = (df['sealevelpressure'] * df['temp']) / (df['windspeed'] + 1e-8)
        
        return df
    
    def create_interaction_features(self, df):
        """Create polynomial and interaction features for key variables"""
        # Select key numeric features for interaction
        key_features = []
        for col in ['temp', 'humidity', 'precip', 'windspeed', 'sealevelpressure', 'cloudcover']:
            if col in df.columns:
                key_features.append(col)
        
        if len(key_features) >= 2:
            # Create interaction features
            key_data = df[key_features].fillna(df[key_features].median())
            
            # Generate polynomial features (degree 2, interactions only)
            poly_data = self.poly_features.fit_transform(key_data)
            poly_feature_names = self.poly_features.get_feature_names_out(key_features)
            
            # Add polynomial features to dataframe
            poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=df.index)
            
            # Only keep interaction terms (not original features or squares)
            interaction_cols = [col for col in poly_feature_names if ' ' in col and '^2' not in col]
            
            for col in interaction_cols:
                df[f'poly_{col}'] = poly_df[col]
        
        return df
    
    def prepare_features_and_target(self, df):
        """Separate features and target variable with feature selection"""
        target_column = 'conditions'
        
        # Encode target variable
        y = self.label_encoder.fit_transform(df[target_column])
        
        # Select feature columns (all numeric columns except target)
        feature_columns = []
        for col in df.columns:
            if col != target_column and df[col].dtype in ['int64', 'float64']:
                feature_columns.append(col)
        
        X = df[feature_columns]
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Remove features with zero variance
        X = X.loc[:, X.var() != 0]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection using mutual information
        self.feature_selector.set_params(k=min(len(X.columns), 50))  # Select top 50 features
        X_selected = self.feature_selector.fit_transform(X_scaled, y)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.feature_columns = [col for col, selected in zip(X.columns, selected_mask) if selected]
        
        print(f"Selected {len(self.feature_columns)} features out of {len(feature_columns)} original features")
        
        return X_selected, y
    
    def get_class_names(self):
        """Get the original class names"""
        return self.label_encoder.classes_
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets with stratification"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def process_all(self):
        """Complete preprocessing pipeline"""
        print("Loading data...")
        df = self.load_data()
        
        print("Cleaning data...")
        df = self.clean_data(df)
        
        print("Engineering advanced features...")
        df = self.advanced_feature_engineering(df)
        
        print("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        print("Preparing features and target...")
        X, y = self.prepare_features_and_target(df)
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Number of classes: {len(self.get_class_names())}")
        print(f"Classes: {self.get_class_names()}")
        print(f"Selected features: {len(self.feature_columns)}")
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("Class distribution:")
        for i, (class_idx, count) in enumerate(zip(unique, counts)):
            class_name = self.get_class_names()[class_idx]
            print(f"  {class_name}: {count} ({count/len(y)*100:.1f}%)")
        
        return X, y