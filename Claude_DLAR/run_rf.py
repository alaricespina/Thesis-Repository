#!/usr/bin/env python3
"""
Deep Learning Auto Regressive (DLAR) Models - RandomForest Variant
Comprehensive implementation of AR models with RandomForest replacing LSTM

Models implemented:
1. Vanilla Auto Regressive (Linear Regression)
2. ARIMA 
3. SARIMA
4. RandomForest AR (RandomForest with deseasonalization replacing LSTM)

Predicts temperature, humidity, and precipitation for 50 years after 2024
Uses historical data from 2001-2024, tests on 2025 data
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our models
from data_loader import DataLoader
from vanilla_ar import VanillaAutoRegressive
from arima_model import ARIMAModel
from sarima_model import SARIMAModel
from simple_sarima import SimpleSARIMA
from random_forest_ar import RandomForestAutoRegressive
from simple_deep_ar import SimpleDeepAR
from fast_deep_ar import FastDeepAR

class DLARRunnerRF:
    def __init__(self, prediction_years=50):
        self.data_loader = DataLoader()
        self.models = {}
        self.results = {}
        self.prediction_years = prediction_years
        self.prediction_days = prediction_years * 365  # Convert years to days
        
    def load_data(self):
        """Load historical and test data"""
        print("Loading historical data (2001-2024)...")
        self.historical_data = self.data_loader.load_historical_data(2001, 2024)
        print(f"Loaded {len(self.historical_data)} historical records")
        
        print("Loading test data (2025)...")
        self.test_data = self.data_loader.load_test_data(2025)
        if self.test_data is not None:
            print(f"Loaded {len(self.test_data)} test records")
        else:
            print("No test data available")
            
        # Prepare target variables
        self.historical_targets = self.data_loader.prepare_target_variables(self.historical_data)
        if self.test_data is not None:
            self.test_targets = self.data_loader.prepare_target_variables(self.test_data)
        
        self.target_columns = ['temp', 'humidity', 'precip']
        available_targets = [col for col in self.target_columns if col in self.historical_targets.columns]
        print(f"Available target variables: {available_targets}")
        self.target_columns = available_targets
        
    def train_vanilla_ar(self):
        """Train Vanilla Auto Regressive model"""
        print("\n" + "="*50)
        print("Training Vanilla Auto Regressive Model")
        print("="*50)
        
        model = VanillaAutoRegressive(order=7)  # 1 week lookback
        model.fit(self.historical_targets, self.target_columns)
        self.models['Vanilla_AR'] = model
        print("Vanilla AR model trained successfully")
        
    def train_arima(self):
        """Train ARIMA model"""
        print("\n" + "="*50)
        print("Training ARIMA Model")
        print("="*50)
        
        model = ARIMAModel(order=(2, 1, 2))
        model.fit(self.historical_targets, self.target_columns)
        self.models['ARIMA'] = model
        print("ARIMA model trained successfully")
        
    def train_sarima(self):
        """Train SARIMA model with fallback"""
        print("\n" + "="*50)
        print("Training SARIMA Model")  
        print("="*50)
        
        try:
            print("Attempting statsmodels SARIMA...")
            model = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            model.fit(self.historical_targets, self.target_columns)
            
            # Check if at least one model was fitted
            if any(col in model.models for col in self.target_columns):
                self.models['SARIMA'] = model
                print("✓ Statsmodels SARIMA trained successfully")
                return
            else:
                print("✗ Statsmodels SARIMA failed - trying Simple SARIMA fallback...")
                
        except Exception as e:
            print(f"✗ Statsmodels SARIMA failed: {e}")
            print("Trying Simple SARIMA fallback...")
        
        # Fallback to Simple SARIMA
        try:
            simple_model = SimpleSARIMA(ar_order=5, seasonal_period=12)
            simple_model.fit(self.historical_targets, self.target_columns)
            
            if any(col in simple_model.models for col in self.target_columns):
                self.models['SARIMA'] = simple_model
                print("✓ Simple SARIMA fallback trained successfully")
            else:
                print("✗ All SARIMA methods failed - skipping SARIMA")
                
        except Exception as e:
            print(f"✗ Simple SARIMA also failed: {e}")
            print("Skipping SARIMA model entirely")
        
    def train_random_forest_ar(self):
        """Train RandomForest AR model with fallback"""
        print("\n" + "="*50)
        print("Training RandomForest Auto Regressive Model")
        print("="*50)
        
        try:
            print("Training RandomForest model...")
            model = RandomForestAutoRegressive(
                sequence_length=30, 
                n_estimators=100,
                max_depth=15,
                batch_size_pred=2000  # Larger batch size for faster predictions
            )
            model.fit(self.historical_targets, self.target_columns)
            
            # Check if at least one model was trained
            if any(col in model.models for col in self.target_columns):
                self.models['RandomForest_AR'] = model
                print("✓ RandomForest Auto Regressive trained successfully")
                return
            else:
                print("✗ RandomForest AR failed - trying Simple Deep AR fallback...")
                
        except Exception as e:
            print(f"✗ RandomForest AR failed: {e}")
            print("Trying Simple Deep AR fallback...")
        
        # Fallback to Simple Deep AR
        try:
            simple_model = SimpleDeepAR(sequence_length=20, hidden_layers=(50, 25))
            simple_model.fit(self.historical_targets, self.target_columns)
            
            if any(col in simple_model.models for col in self.target_columns):
                self.models['RandomForest_AR'] = simple_model
                print("✓ Simple Deep AR fallback trained successfully")
                return
            else:
                print("✗ Simple Deep AR failed - trying Fast Deep AR fallback...")
                
        except Exception as e:
            print(f"✗ Simple Deep AR failed: {e}")
            print("Trying Fast Deep AR fallback...")
        
        # Final fallback to Fast Deep AR
        try:
            fast_model = FastDeepAR(pattern_length=365)
            fast_model.fit(self.historical_targets, self.target_columns)
            
            if any(col in fast_model.patterns for col in self.target_columns):
                self.models['RandomForest_AR'] = fast_model
                print("✓ Fast Deep AR fallback trained successfully")
            else:
                print("✗ All RandomForest/Deep Learning methods failed - skipping RandomForest AR")
                
        except Exception as e:
            print(f"✗ All RandomForest/Deep Learning methods failed: {e}")
            print("Skipping RandomForest AR model entirely")
        
    def generate_long_term_predictions(self):
        """Generate long-term predictions for all models"""
        print("\n" + "="*50)
        print(f"Generating {self.prediction_years}-Year Predictions ({self.prediction_days} days)")
        print("="*50)
        
        self.predictions_long_term = {}
        
        for model_name, model in self.models.items():
            print(f"Generating {self.prediction_years}-year predictions with {model_name}...")
            
            if model_name == 'Vanilla_AR':
                predictions = model.predict_n_years(self.historical_targets, self.target_columns, self.prediction_days)
            elif model_name == 'RandomForest_AR':
                predictions = model.predict_n_years(self.historical_targets, self.target_columns, self.prediction_days)
            else:  # ARIMA, SARIMA
                predictions = model.predict_n_years(self.target_columns, self.prediction_days)
                
            self.predictions_long_term[model_name] = predictions
            print(f"✓ {self.prediction_years}-year predictions generated for {model_name}")
            
    def test_on_2025_data(self):
        """Test models on 2025 data if available"""
        if self.test_data is None:
            print("No test data available for validation")
            return
            
        print("\n" + "="*50)
        print("Testing Models on 2025 Data")
        print("="*50)
        
        test_length = len(self.test_targets)
        self.test_predictions = {}
        self.test_results = {}
        
        for model_name, model in self.models.items():
            print(f"Testing {model_name} on 2025 data...")
            
            if model_name == 'Vanilla_AR':
                # Use last values from historical data for prediction
                predictions = {}
                for col in self.target_columns:
                    if col in self.historical_targets.columns:
                        series = self.historical_targets[col].values
                        initial_values = series[-model.order:]
                        predictions[col] = model.predict_recursive(initial_values, test_length, col)
            elif model_name == 'RandomForest_AR':
                predictions = model.predict_test_period(self.historical_targets, test_length, self.target_columns)
            else:  # ARIMA, SARIMA
                predictions = model.predict_test_period(test_length, self.target_columns)
                
            self.test_predictions[model_name] = predictions
            
            # Evaluate predictions
            true_values = {}
            for col in self.target_columns:
                if col in self.test_targets.columns:
                    true_values[col] = self.test_targets[col].values
                    
            results = model.evaluate(true_values, predictions)
            self.test_results[model_name] = results
            
            # Print results
            print(f"Results for {model_name}:")
            for col, metrics in results.items():
                print(f"  {col}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
                
    def save_predictions(self):
        """Save predictions to CSV files"""
        print("\n" + "="*50)
        print("Saving Predictions")
        print("="*50)
        
        # Create predictions directory
        pred_dir = "predictions"
        os.makedirs(pred_dir, exist_ok=True)
        
        # Generate date range for prediction years
        start_date = datetime(2025, 1, 1)
        date_range = [start_date + timedelta(days=i) for i in range(self.prediction_days)]
        
        # Save long-term predictions
        for model_name, predictions in self.predictions_long_term.items():
            data_dict = {'Date': date_range}
            
            for col in self.target_columns:
                if col in predictions:
                    data_dict[f'{col}_predicted'] = predictions[col]
                    
            df = pd.DataFrame(data_dict)
            filename = f"{pred_dir}/{model_name}_{self.prediction_years}_year_predictions_rf.csv"
            df.to_csv(filename, index=False)
            print(f"✓ Saved {filename}")
            
        # Save test predictions if available
        if hasattr(self, 'test_predictions'):
            test_date_range = self.test_targets['datetime'].values
            
            for model_name, predictions in self.test_predictions.items():
                data_dict = {'Date': test_date_range}
                
                # Add true values
                for col in self.target_columns:
                    if col in self.test_targets.columns:
                        data_dict[f'{col}_true'] = self.test_targets[col].values
                        
                # Add predictions
                for col in self.target_columns:
                    if col in predictions:
                        pred_values = predictions[col]
                        if len(pred_values) >= len(test_date_range):
                            data_dict[f'{col}_predicted'] = pred_values[:len(test_date_range)]
                        else:
                            # Pad with NaN if not enough predictions
                            padded = np.full(len(test_date_range), np.nan)
                            padded[:len(pred_values)] = pred_values
                            data_dict[f'{col}_predicted'] = padded
                            
                df = pd.DataFrame(data_dict)
                filename = f"{pred_dir}/{model_name}_2025_test_predictions_rf.csv"
                df.to_csv(filename, index=False)
                print(f"✓ Saved {filename}")
                
    def plot_results(self):
        """Create visualization plots"""
        print("\n" + "="*50)
        print("Creating Visualization Plots")
        print("="*50)
        
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot test results if available
        if hasattr(self, 'test_predictions') and self.test_data is not None:
            for col in self.target_columns:
                if col in self.test_targets.columns:
                    plt.figure(figsize=(15, 8))
                    
                    # Plot true values
                    dates = pd.to_datetime(self.test_targets['datetime'])
                    true_values = self.test_targets[col].values
                    plt.plot(dates, true_values, label='True Values', linewidth=2, color='black')
                    
                    # Plot predictions from each model
                    colors = ['red', 'blue', 'green', 'orange']
                    for i, (model_name, predictions) in enumerate(self.test_predictions.items()):
                        if col in predictions:
                            pred_values = predictions[col][:len(dates)]
                            plt.plot(dates, pred_values, label=f'{model_name}', 
                                   linewidth=1.5, color=colors[i % len(colors)], alpha=0.8)
                    
                    plt.title(f'{col.title()} Predictions - 2025 Test Data (RandomForest)', fontsize=14)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel(f'{col.title()}', fontsize=12)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'{plots_dir}/{col}_2025_predictions_rf.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Saved {col}_2025_predictions_rf.png")
                    
        # Plot sample of long-term predictions (first year)
        sample_days = min(365, self.prediction_days)  # First year or full prediction if less than a year
        sample_dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(sample_days)]
        
        for col in self.target_columns:
            plt.figure(figsize=(15, 8))
            
            colors = ['red', 'blue', 'green', 'orange']
            for i, (model_name, predictions) in enumerate(self.predictions_long_term.items()):
                if col in predictions:
                    sample_pred = predictions[col][:sample_days]
                    plt.plot(sample_dates, sample_pred, label=f'{model_name}', 
                           linewidth=1.5, color=colors[i % len(colors)], alpha=0.8)
            
            title_period = "First Year" if self.prediction_years >= 1 else f"{self.prediction_years}-Year"
            plt.title(f'{col.title()} - {title_period} of {self.prediction_years}-Year Predictions (RandomForest)', fontsize=14)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel(f'{col.title()}', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/{col}_{self.prediction_years}_year_sample_rf.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved {col}_{self.prediction_years}_year_sample_rf.png")
            
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*60)
        print("DEEP LEARNING AUTO REGRESSIVE (DLAR) - RANDOMFOREST VARIANT - SUMMARY")
        print("="*60)
        
        print(f"Historical Data: {len(self.historical_data)} records (2001-2024)")
        if self.test_data is not None:
            print(f"Test Data: {len(self.test_data)} records (2025)")
        print(f"Target Variables: {', '.join(self.target_columns)}")
        print(f"Models Trained: {len(self.models)}")
        
        print(f"\nModels:")
        for model_name in self.models.keys():
            print(f"  ✓ {model_name}")
            
        print(f"\n{self.prediction_years}-Year Predictions: Generated for all models ({self.prediction_days:,} days)")
        print(f"Files saved in: predictions/ and plots/ directories")
        
        if hasattr(self, 'test_results'):
            print(f"\n2025 Test Results (RMSE):")
            for model_name, results in self.test_results.items():
                print(f"  {model_name}:")
                for col, metrics in results.items():
                    print(f"    {col}: {metrics['RMSE']:.4f}")
                    
        print("\n" + "="*60)
        
def main(prediction_years=50):
    """Main execution function"""
    print("DEEP LEARNING AUTO REGRESSIVE (DLAR) MODELS - RANDOMFOREST VARIANT")
    print(f"Weather Prediction for {prediction_years} Years")
    print("=" * 60)
    
    runner = DLARRunnerRF(prediction_years=prediction_years)
    
    try:
        # Load data
        runner.load_data()
        
        # Train all models
        runner.train_vanilla_ar()
        runner.train_arima()
        runner.train_sarima()
        runner.train_random_forest_ar()
        
        # Generate predictions
        runner.generate_long_term_predictions()
        
        # Test on 2025 data
        runner.test_on_2025_data()
        
        # Save results
        runner.save_predictions()
        runner.plot_results()
        
        # Print summary
        runner.print_summary()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    prediction_years = 50  # Default
    
    if len(sys.argv) > 1:
        try:
            prediction_years = int(sys.argv[1])
            if prediction_years <= 0:
                print("Error: Prediction years must be positive")
                sys.exit(1)
        except ValueError:
            print("Error: Invalid prediction years. Must be an integer.")
            print("Usage: python run_rf.py [prediction_years]")
            print("Example: python run_rf.py 5")
            sys.exit(1)
    
    main(prediction_years)