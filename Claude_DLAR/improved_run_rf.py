"""
DEEP LEARNING AUTO REGRESSIVE (DLAR) MODELS - COMPREHENSIVE VARIANT

Integrates all model variants with the improved framework.

Models included:
1. Vanilla AR (with seasonal decomposition)
2. ARIMA (with seasonal handling)
3. SARIMA (cross-platform compatible)
4. Legacy Random Forest AR (custom deseasonalization)
5. Legacy LSTM AR (custom deseasonalization with LSTM)

Predicts temperature, tempmax, tempmin, humidity, and precipitation for 50 years after 2024
Uses historical data from 2001-2024, tests on 2025 data
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import existing models
from data_loader import DataLoader
from vanilla_ar import VanillaAutoRegressive
from arima_model import ARIMAModel
from sarima_model import SARIMAModel
from legacy_rf_model import LegacyRandomForestAR
from legacy_lstm_model import LegacyLSTMAR


class ImprovedModelRunner:
    def __init__(self, years_to_predict):
        self.years_to_predict = years_to_predict
        self.data_loader = DataLoader()
        self.models = {}
        
        # Create models directory
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        print(f"DEEP LEARNING AUTO REGRESSIVE (DLAR) MODELS - COMPREHENSIVE VARIANT")
        print(f"Weather Prediction for {years_to_predict} Years")
        print("=" * 60)
        
        # Load data
        print("Loading historical data (2001-2024)...")
        self.historical_data = self.data_loader.load_historical_data()
        print(f"Loaded {len(self.historical_data)} historical records")
        
        print("Loading test data (2025)...")
        self.test_data = self.data_loader.load_test_data()
        if self.test_data is not None:
            print(f"Loaded {len(self.test_data)} test records")
        
        # Prepare target variables
        self.historical_targets = self.data_loader.prepare_target_variables(self.historical_data)
        if self.test_data is not None:
            self.test_targets = self.data_loader.prepare_target_variables(self.test_data)
        
        self.target_columns = ['temp', 'tempmax', 'tempmin', 'humidity', 'precip']
        available_targets = [col for col in self.target_columns if col in self.historical_targets.columns]
        print(f"Available target variables: {available_targets}")
        self.target_columns = available_targets
        
    def train_vanilla_ar(self):
        """Train Vanilla Auto Regressive model"""
        print("\n" + "="*50)
        print("Training Vanilla Auto Regressive Model")
        print("="*50)
        
        try:
            self.vanilla_ar = VanillaAutoRegressive(order=5)
            self.vanilla_ar.fit(self.historical_targets, self.target_columns)
            self.models['Vanilla_AR'] = self.vanilla_ar
            
            # Save model
            model_path = os.path.join(self.models_dir, 'vanilla_ar_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.vanilla_ar, f)
            print(f"✓ Vanilla AR model saved to {model_path}")
            print("Vanilla AR model trained successfully")
            
        except Exception as e:
            print(f"✗ Error training Vanilla AR: {e}")
    
    def train_arima(self):
        """Train ARIMA model"""
        print("\n" + "="*50)
        print("Training ARIMA Model")
        print("="*50)
        
        try:
            self.arima = ARIMAModel(order=(2, 1, 2))
            self.arima.fit(self.historical_targets, self.target_columns)
            self.models['ARIMA'] = self.arima
            
            # Save model
            model_path = os.path.join(self.models_dir, 'arima_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.arima, f)
            print(f"✓ ARIMA model saved to {model_path}")
            print("ARIMA model trained successfully")
            
        except Exception as e:
            print(f"✗ Error training ARIMA: {e}")
    
    def train_sarima(self):
        """Train SARIMA model"""
        print("\n" + "="*50)
        print("Training SARIMA Model")
        print("="*50)
        
        try:
            self.sarima = SARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 365))
            self.sarima.fit(self.historical_targets, self.target_columns)
            self.models['SARIMA'] = self.sarima
            
            # Save model
            model_path = os.path.join(self.models_dir, 'sarima_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.sarima, f)
            print(f"✓ SARIMA model saved to {model_path}")
            print("SARIMA model trained successfully")
            
        except Exception as e:
            print(f"✗ Error training SARIMA: {e}")
    
    def train_legacy_random_forest(self):
        """Train Legacy Random Forest model"""
        print("\n" + "="*50)
        print("Training Legacy Random Forest Model")
        print("="*50)
        
        try:
            self.legacy_rf = LegacyRandomForestAR(model_order=3, yearly_season_length=7)
            self.legacy_rf.fit(self.historical_data, self.target_columns)
            self.models['Legacy_RF'] = self.legacy_rf
            
            # Save model
            model_path = os.path.join(self.models_dir, 'legacy_rf_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.legacy_rf, f)
            print(f"✓ Legacy RF model saved to {model_path}")
            print("Legacy Random Forest model trained successfully")
            
        except Exception as e:
            print(f"✗ Error training Legacy RF: {e}")
    
    def train_legacy_lstm(self):
        """Train Legacy LSTM model"""
        print("\n" + "="*50)
        print("Training Legacy LSTM Model")
        print("="*50)
        
        try:
            self.legacy_lstm = LegacyLSTMAR(model_order=3, yearly_season_length=7)
            self.legacy_lstm.fit(self.historical_data, self.target_columns)
            self.models['Legacy_LSTM'] = self.legacy_lstm
            
            # Save model
            model_path = os.path.join(self.models_dir, 'legacy_lstm_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.legacy_lstm, f)
            print(f"✓ Legacy LSTM model saved to {model_path}")
            print("Legacy LSTM model trained successfully")
            
        except Exception as e:
            print(f"✗ Error training Legacy LSTM: {e}")
    
    def generate_predictions(self):
        """Generate long-term predictions for all models"""
        print("\n" + "="*50)
        print(f"Generating {self.years_to_predict}-Year Predictions ({self.years_to_predict * 365} days)")
        print("="*50)
        
        n_days = self.years_to_predict * 365
        self.long_term_predictions = {}
        
        for model_name, model in self.models.items():
            try:
                print(f"Generating {self.years_to_predict}-year predictions with {model_name}...")
                
                if model_name == 'Legacy_RF':
                    predictions = model.predict_n_years(self.target_columns, n_days)
                else:
                    predictions = model.predict_n_years(self.target_columns, n_days)
                
                self.long_term_predictions[model_name] = predictions
                print(f"✓ {self.years_to_predict}-year predictions generated for {model_name}")
                
            except Exception as e:
                print(f"✗ Error generating predictions for {model_name}: {e}")
    
    def test_models(self):
        """Test models on 2025 data"""
        if self.test_data is None:
            print("No test data available for evaluation")
            return
            
        print("\n" + "="*50)
        print("Testing Models on 2025 Data")
        print("="*50)
        
        self.test_predictions = {}
        self.test_results = {}
        test_length = len(self.test_data)
        
        for model_name, model in self.models.items():
            try:
                print(f"Testing {model_name} on 2025 data...")
                
                if model_name == 'Legacy_RF':
                    predictions = model.predict_test_period(test_length, self.target_columns)
                else:
                    predictions = model.predict_test_period(test_length, self.target_columns)
                
                self.test_predictions[model_name] = predictions
                
                # Evaluate
                results = model.evaluate(self.test_targets, predictions)
                self.test_results[model_name] = results
                
                print(f"Results for {model_name}:")
                for var, metrics in results.items():
                    print(f"  {var}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
                    
            except Exception as e:
                print(f"✗ Error testing {model_name}: {e}")
    
    def save_predictions(self):
        """Save predictions to CSV files"""
        print("\n" + "="*50)
        print("Saving Predictions")
        print("="*50)
        
        predictions_dir = "predictions"
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save long-term predictions
        for model_name, predictions in self.long_term_predictions.items():
            try:
                # Create date range starting from 2025
                start_date = datetime(2025, 1, 1)
                dates = pd.date_range(start=start_date, periods=len(list(predictions.values())[0]), freq='D')
                
                df = pd.DataFrame({'date': dates})
                for var, pred_values in predictions.items():
                    df[var] = pred_values
                
                filename = f"{model_name}_{self.years_to_predict}_year_predictions_improved.csv"
                filepath = os.path.join(predictions_dir, filename)
                df.to_csv(filepath, index=False)
                print(f"✓ Saved {filepath}")
                
            except Exception as e:
                print(f"✗ Error saving predictions for {model_name}: {e}")
        
        # Save test predictions
        if hasattr(self, 'test_predictions'):
            for model_name, predictions in self.test_predictions.items():
                try:
                    dates = pd.to_datetime(self.test_data['datetime'])
                    df = pd.DataFrame({'date': dates})
                    for var, pred_values in predictions.items():
                        df[var] = pred_values[:len(dates)]  # Ensure same length
                    
                    filename = f"{model_name}_2025_test_predictions_improved.csv"
                    filepath = os.path.join(predictions_dir, filename)
                    df.to_csv(filepath, index=False)
                    print(f"✓ Saved {filepath}")
                    
                except Exception as e:
                    print(f"✗ Error saving test predictions for {model_name}: {e}")
    
    def create_plots(self):
        """Create visualization plots"""
        print("\n" + "="*50)
        print("Creating Visualization Plots")
        print("="*50)
        
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        if not hasattr(self, 'test_predictions'):
            print("No test predictions available for plotting")
            return
        
        # Plot test period predictions
        for var in self.target_columns:
            try:
                plt.figure(figsize=(15, 8))
                
                # Plot actual values
                if self.test_data is not None:
                    dates = pd.to_datetime(self.test_data['datetime'])
                    actual_values = self.test_targets[var].values
                    plt.plot(dates, actual_values, 'k-', label='True Values', linewidth=2)
                
                # Plot model predictions
                colors = ['red', 'blue', 'green', 'orange']
                for i, (model_name, predictions) in enumerate(self.test_predictions.items()):
                    if var in predictions:
                        color = colors[i % len(colors)]
                        pred_values = predictions[var][:len(dates)]
                        plt.plot(dates, pred_values, color=color, label=model_name, alpha=0.8)
                
                plt.title(f'{var.title()} Predictions - 2025 Test Data (Improved)')
                plt.xlabel('Date')
                plt.ylabel(var.title())
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                filename = f"{var}_2025_predictions_improved.png"
                filepath = os.path.join(plots_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved {filepath}")
                
            except Exception as e:
                print(f"✗ Error creating plot for {var}: {e}")
        
        # Plot sample of long-term predictions
        for var in self.target_columns:
            try:
                plt.figure(figsize=(15, 8))
                
                # Plot first year of predictions
                sample_days = 365
                start_date = datetime(2025, 1, 1)
                dates = pd.date_range(start=start_date, periods=sample_days, freq='D')
                
                colors = ['red', 'blue', 'green', 'orange']
                for i, (model_name, predictions) in enumerate(self.long_term_predictions.items()):
                    if var in predictions:
                        color = colors[i % len(colors)]
                        pred_values = predictions[var][:sample_days]
                        plt.plot(dates, pred_values, color=color, label=model_name, alpha=0.8)
                
                plt.title(f'{var.title()} {self.years_to_predict}-Year Sample (Improved)')
                plt.xlabel('Date')
                plt.ylabel(var.title())
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                filename = f"{var}_{self.years_to_predict}_year_sample_improved.png"
                filepath = os.path.join(plots_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved {filepath}")
                
            except Exception as e:
                print(f"✗ Error creating long-term plot for {var}: {e}")
    
    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*60)
        print("DEEP LEARNING AUTO REGRESSIVE (DLAR) - COMPREHENSIVE VARIANT - SUMMARY")
        print("="*60)
        
        print(f"Historical Data: {len(self.historical_data)} records (2001-2024)")
        if self.test_data is not None:
            print(f"Test Data: {len(self.test_data)} records (2025)")
        print(f"Target Variables: {', '.join(self.target_columns)}")
        print(f"Models Trained: {len(self.models)}")
        
        print(f"\nModels:")
        for model_name in self.models.keys():
            print(f"  ✓ {model_name}")
        
        print(f"\n{self.years_to_predict}-Year Predictions: Generated for all models ({self.years_to_predict * 365:,} days)")
        print(f"Files saved in: predictions/ and plots/ directories")
        print(f"Models saved in: {self.models_dir}/ directory")
        
        if hasattr(self, 'test_results'):
            print(f"\n2025 Test Results (RMSE):")
            for model_name, results in self.test_results.items():
                print(f"  {model_name}:")
                for var, metrics in results.items():
                    print(f"    {var}: {metrics['RMSE']:.4f}")
        
        print("="*60)


def main():
    if len(sys.argv) != 2:
        print("Usage: python improved_run_rf.py <years_to_predict>")
        print("Example: python improved_run_rf.py 5")
        sys.exit(1)
    
    try:
        years_to_predict = int(sys.argv[1])
        if years_to_predict <= 0:
            raise ValueError("Years must be positive")
    except ValueError as e:
        print(f"Error: Invalid number of years - {e}")
        sys.exit(1)
    
    # Initialize runner
    runner = ImprovedModelRunner(years_to_predict)
    
    try:
        # Train all models
        # runner.train_vanilla_ar()
        # runner.train_arima()
        # runner.train_sarima()
        runner.train_legacy_random_forest()
        # runner.train_legacy_lstm()
        
        # Generate predictions
        runner.generate_predictions()
        
        # Test models
        runner.test_models()
        
        # Save results
        runner.save_predictions()
        runner.create_plots()
        
        # Print summary
        runner.print_summary()
        
    except KeyboardInterrupt:
        print("\n✗ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()