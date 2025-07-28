import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import glob
import os


class WeatherDataPreprocessor:
    def __init__(self, data_path, use_2011_onwards=False):
        self.data_path = data_path
        self.use_2011_onwards = use_2011_onwards
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self, exclude_2025=False):
        """Load CSV files from the data directory with year filtering"""
        csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        # Filter out specific files we don't want
        csv_files = [f for f in csv_files if not any(exclude in f for exclude in 
                    ["Concatenated", "PREDICTIONS", "Model Output", "Hourly"])]
        
        # Filter by year range if specified
        if self.use_2011_onwards or exclude_2025:
            filtered_files = []
            for file in csv_files:
                filename = os.path.basename(file)
                if filename.replace('.csv', '').isdigit():
                    year = int(filename.replace('.csv', ''))
                    if exclude_2025 and year == 2025:
                        continue
                    if self.use_2011_onwards and year < 2011:
                        continue
                    if not self.use_2011_onwards and year == 2025:
                        continue
                filtered_files.append(file)
            csv_files = filtered_files
        
        dataframes = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dataframes.append(df)
                filename = os.path.basename(file)
                print(f"Loaded {filename}: {df.shape}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if dataframes:
            combined_data = pd.concat(dataframes, ignore_index=True)
            year_range = "2011-2024" if self.use_2011_onwards else "2001-2024"
            print(f"Combined training dataset ({year_range}) shape: {combined_data.shape}")
            return combined_data
        else:
            raise ValueError("No valid CSV files found in the specified directory")
    
    def load_test_data_2025(self):
        """Load 2025.csv as test data"""
        test_file = os.path.join(self.data_path, "2025.csv")
        if os.path.exists(test_file):
            df = pd.read_csv(test_file)
            print(f"Loaded 2025 test data: {df.shape}")
            return df
        else:
            raise ValueError("2025.csv not found in the specified directory")
    
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
        print("Loading training data...")
        df = self.load_data(exclude_2025=True)
        
        print("Cleaning training data...")
        df = self.clean_data(df)
        
        print("Engineering features...")
        df = self.engineer_features(df)
        
        print("Preparing features and target...")
        X_train, y_train = self.prepare_features_and_target(df)
        
        print(f"Final training dataset shape: {X_train.shape}")
        print(f"Number of classes: {len(self.get_class_names())}")
        print(f"Classes: {self.get_class_names()}")
        print(f"Feature columns: {self.feature_columns}")
        
        return X_train, y_train
    
    def process_test_2025(self):
        """Process 2025 test data using fitted preprocessors"""
        print("Loading 2025 test data...")
        df_test = self.load_test_data_2025()
        
        print("Cleaning test data...")
        df_test = self.clean_data(df_test)
        
        print("Engineering features for test data...")
        df_test = self.engineer_features(df_test)
        
        # Encode target variable using fitted encoder
        if 'conditions' in df_test.columns:
            y_test = self.label_encoder.transform(df_test['conditions'])
        else:
            raise ValueError("conditions column not found in test data")
        
        # Select same feature columns as training data
        X_test = df_test[self.feature_columns]
        
        # Handle missing values more robustly
        X_test = X_test.fillna(X_test.median())
        
        # If there are still NaNs (columns with all NaN values), fill with 0
        X_test = X_test.fillna(0)
        
        # Replace any infinite values with finite values
        X_test = X_test.replace([np.inf, -np.inf], 0)
        
        # Final check for NaN values before scaling
        if X_test.isnull().any().any():
            print("Warning: Found remaining NaN values, filling with 0")
            X_test = X_test.fillna(0)
        
        # Scale features using fitted scaler
        X_test_scaled = self.scaler.transform(X_test)
        
        # Final check for NaN values after scaling
        if np.isnan(X_test_scaled).any():
            print("Warning: Found NaN values after scaling, replacing with 0")
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Final test dataset shape: {X_test_scaled.shape}")
        
        return X_test_scaled, y_test