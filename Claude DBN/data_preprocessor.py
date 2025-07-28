import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import glob
import os


class WeatherDataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
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
                          'sunrise', 'sunset', 'moonphase']
        
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
    
    def engineer_features(self, df):
        """Create additional features from existing data"""
        # Temperature range
        if 'tempmax' in df.columns and 'tempmin' in df.columns:
            df['temp_range'] = df['tempmax'] - df['tempmin']
        
        # Feels like difference
        if 'feelslike' in df.columns and 'temp' in df.columns:
            df['feels_like_diff'] = df['feelslike'] - df['temp']
        
        # Dew point depression
        if 'temp' in df.columns and 'dew' in df.columns:
            df['dew_point_depression'] = df['temp'] - df['dew']
        
        # Wind chill factor (simplified)
        if 'temp' in df.columns and 'windspeed' in df.columns:
            df['wind_chill_factor'] = df['temp'] - (df['windspeed'] * 0.1)
        
        # Pressure tendency (if we have multiple readings - simplified)
        if 'sealevelpressure' in df.columns:
            df['pressure_normalized'] = (df['sealevelpressure'] - 1013.25) / 1013.25
        
        return df
    
    def prepare_features_and_target(self, df):
        """Separate features and target variable"""
        target_column = 'conditions'
        
        # Encode target variable
        y = self.label_encoder.fit_transform(df[target_column])
        
        # Select feature columns (all numeric columns except target)
        feature_columns = []
        for col in df.columns:
            if col != target_column and df[col].dtype in ['int64', 'float64']:
                feature_columns.append(col)
        
        X = df[feature_columns]
        self.feature_columns = feature_columns
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def get_class_names(self):
        """Get the original class names"""
        return self.label_encoder.classes_
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def process_all(self):
        """Complete preprocessing pipeline"""
        print("Loading data...")
        df = self.load_data()
        
        print("Cleaning data...")
        df = self.clean_data(df)
        
        print("Engineering features...")
        df = self.engineer_features(df)
        
        print("Preparing features and target...")
        X, y = self.prepare_features_and_target(df)
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Number of classes: {len(self.get_class_names())}")
        print(f"Classes: {self.get_class_names()}")
        print(f"Feature columns: {self.feature_columns}")
        
        return X, y