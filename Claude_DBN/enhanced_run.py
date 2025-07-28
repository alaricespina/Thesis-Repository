#!/usr/bin/env python3
"""
Enhanced Deep Belief Network with Random Forest Classifier for Weather Condition Prediction

This script implements an optimized pipeline for training and evaluating both a Deep Belief Network
with Random Forest classifier and a vanilla Random Forest classifier for comparison.
Features: Advanced feature engineering, optimized hyperparameters, ensemble methods, and graph saving.
"""

import os
import sys
import numpy as np
import pandas as pd
import time
# from sklearn.model_selection import train_test_split  # No longer needed
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_data_preprocessor import EnhancedWeatherDataPreprocessor
from optimized_dbn_rf_classifier import OptimizedDBNRandomForestClassifier, OptimizedVanillaRandomForestClassifier
from enhanced_model_comparison import EnhancedModelComparison


def main():
    """Main execution function"""
    print("=== ENHANCED Weather Classifier: Optimized DBN+RF vs Vanilla RF ===\n")
    
    # Ask user for training data range preference
    print("Training Data Options:")
    print("1. Use all historical data (2001-2024)")
    print("2. Use recent data only (2011-2024)")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    use_2011_onwards = choice == '2'
    year_suffix = "2011_2024" if use_2011_onwards else "2001_2024"
    
    # Configuration
    DATA_PATH = "../Data Source Files/"
    DBN_MODEL_PATH = f"models/optimized_dbn_rf_model_{year_suffix}.pkl"
    VANILLA_MODEL_PATH = f"models/optimized_vanilla_rf_model_{year_suffix}.pkl"
    PLOTS_DIR = f"plots_{year_suffix}"
    
    # Optimized DBN Configuration for higher accuracy
    DBN_CONFIG = {
        'dbn_hidden_layers': [256, 128, 64],      # Deeper network
        'dbn_learning_rate': 0.001,               # Lower learning rate
        'dbn_epochs': 10,                        # More epochs
        'dbn_batch_size': 128,                    # Larger batches
        'rf_n_estimators': 500,                   # More trees
        'rf_max_depth': 20,                       # Deeper trees
        'rf_min_samples_split': 5,
        'rf_min_samples_leaf': 2,
        'rf_random_state': 42,
        'use_ensemble': True                      # Use ensemble methods
    }
    
    # Optimized Vanilla RF Configuration
    VANILLA_RF_CONFIG = {
        'n_estimators': 500,                      # More trees
        'max_depth': 20,                          # Deeper trees
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'use_ensemble': True                      # Use ensemble methods
    }
    
    try:
        # Step 1: Load and preprocess training data with enhanced features
        print("Step 1: Loading and preprocessing training data with advanced feature engineering...")
        preprocessor = EnhancedWeatherDataPreprocessor(DATA_PATH, use_2011_onwards=use_2011_onwards)
        X_train, y_train = preprocessor.process_all()
        
        class_names = preprocessor.get_class_names()
        print(f"\nTraining dataset loaded successfully!")
        print(f"Features shape: {X_train.shape}")
        print(f"Target shape: {y_train.shape}")
        print(f"Number of weather conditions: {len(class_names)}")
        print(f"Weather conditions: {list(class_names)}")
        
        # Initialize enhanced model comparison with plot saving
        comparison = EnhancedModelComparison(save_plots=True, plot_dir=PLOTS_DIR)
        
        # Plot and save class distribution
        comparison.plot_class_distribution(y_train, class_names, f"Enhanced Training Data Distribution ({year_suffix})")
        
        # Step 2: Load and preprocess 2025 test data
        print("\nStep 2: Loading 2025 test data...")
        X_test, y_test = preprocessor.process_test_2025()
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set (2025): {X_test.shape[0]} samples")
        
        # Step 3: Train or load optimized models
        print("\nStep 3: Training/Loading optimized models...")
        
        # Train Optimized DBN + Random Forest Model
        print("\n--- Training Optimized DBN + Random Forest Model ---")
        if os.path.exists(DBN_MODEL_PATH):
            print(f"Loading existing optimized DBN+RF model from {DBN_MODEL_PATH}")
            dbn_model = OptimizedDBNRandomForestClassifier()
            dbn_model.load_model(DBN_MODEL_PATH)
            dbn_training_time = None
        else:
            print("Training new optimized DBN+RF model...")
            dbn_model = OptimizedDBNRandomForestClassifier(**DBN_CONFIG)
            
            start_time = time.time()
            dbn_model.fit(X_train, y_train)
            dbn_training_time = time.time() - start_time
            
            print(f"Optimized DBN+RF training completed in {dbn_training_time:.2f} seconds")
            dbn_model.save_model(DBN_MODEL_PATH)
        
        # Train Optimized Vanilla Random Forest Model
        print("\n--- Training Optimized Vanilla Random Forest Model ---")
        if os.path.exists(VANILLA_MODEL_PATH):
            print(f"Loading existing optimized Vanilla RF model from {VANILLA_MODEL_PATH}")
            vanilla_model = OptimizedVanillaRandomForestClassifier()
            vanilla_model.load_model(VANILLA_MODEL_PATH)
            vanilla_training_time = None
        else:
            print("Training new optimized Vanilla RF model...")
            vanilla_model = OptimizedVanillaRandomForestClassifier(**VANILLA_RF_CONFIG)
            
            start_time = time.time()
            vanilla_model.fit(X_train, y_train)
            vanilla_training_time = time.time() - start_time
            
            print(f"Optimized Vanilla RF training completed in {vanilla_training_time:.2f} seconds")
            vanilla_model.save_model(VANILLA_MODEL_PATH)
        
        # Step 4: Comprehensive model evaluation
        print("\nStep 4: Comprehensive model evaluation...")
        
        # Evaluate Optimized DBN + RF Model
        print("\n--- Optimized DBN + RF Model Evaluation ---")
        dbn_train_score = dbn_model.score(X_train, y_train)
        print(f"Training Accuracy: {dbn_train_score:.4f}")
        dbn_results = dbn_model.evaluate(X_test, y_test, class_names)
        comparison.add_model_results(
            "Optimized DBN+RF", y_test, dbn_results['predictions'], 
            dbn_results['probabilities'], dbn_training_time
        )
        
        # Evaluate Optimized Vanilla RF Model
        print("\n--- Optimized Vanilla RF Model Evaluation ---")
        vanilla_train_score = vanilla_model.score(X_train, y_train)
        print(f"Training Accuracy: {vanilla_train_score:.4f}")
        vanilla_results = vanilla_model.evaluate(X_test, y_test, class_names)
        comparison.add_model_results(
            "Optimized Vanilla RF", y_test, vanilla_results['predictions'],
            vanilla_results['probabilities'], vanilla_training_time
        )
        
        # Step 5: Enhanced Model Comparison with saved plots
        print("\nStep 5: Enhanced Model Comparison with saved visualizations...")
        
        # Generate comparison report
        comparison.generate_comparison_report()
        
        # Plot and save all comparisons
        comparison.plot_accuracy_comparison()
        comparison.plot_metrics_comparison()
        comparison.plot_confusion_matrices(class_names)
        comparison.plot_training_time_comparison()
        
        # Save comparison results
        comparison.save_results_to_csv(f"optimized_model_comparison_results_{year_suffix}.csv")
        
        # Step 6: Advanced feature analysis for DBN model
        print("\nStep 6: Advanced DBN feature extraction analysis...")
        dbn_features_train = dbn_model.get_dbn_features(X_train)
        dbn_features_test = dbn_model.get_dbn_features(X_test)
        
        print(f"Original features: {X_train.shape[1]}")
        print(f"DBN extracted features: {dbn_features_train.shape[1]}")
        print(f"Feature reduction ratio: {dbn_features_train.shape[1] / X_train.shape[1]:.2f}")
        print(f"Dimensionality reduction: {((X_train.shape[1] - dbn_features_train.shape[1]) / X_train.shape[1] * 100):.1f}%")
        
        # Step 7: Model configuration details
        print("\nStep 7: Model configuration details...")
        print("\nOptimized DBN+RF Configuration:")
        dbn_info = dbn_model.get_model_info()
        for key, value in dbn_info.items():
            print(f"  {key}: {value}")
        
        print("\nOptimized Vanilla RF Configuration:")
        vanilla_info = vanilla_model.get_model_info()
        for key, value in vanilla_info.items():
            print(f"  {key}: {value}")
        
        # Step 8: Sample predictions analysis
        print("\nStep 8: Sample predictions analysis...")
        sample_indices = np.random.choice(len(X_test), 10, replace=False)
        
        correct_predictions = 0
        disagreements = 0
        
        for i, idx in enumerate(sample_indices):
            sample = X_test[idx:idx+1]
            
            # DBN+RF predictions
            dbn_prediction = dbn_model.predict(sample)[0]
            dbn_probabilities = dbn_model.predict_proba(sample)[0]
            
            # Vanilla RF predictions
            vanilla_prediction = vanilla_model.predict(sample)[0]
            vanilla_probabilities = vanilla_model.predict_proba(sample)[0]
            
            actual = y_test[idx]
            
            print(f"\nSample {i+1}:")
            print(f"  Actual: {class_names[actual]}")
            print(f"  DBN+RF: {class_names[dbn_prediction]} (conf: {dbn_probabilities[dbn_prediction]:.3f})")
            print(f"  Vanilla RF: {class_names[vanilla_prediction]} (conf: {vanilla_probabilities[vanilla_prediction]:.3f})")
            
            # Track accuracy
            if dbn_prediction == actual:
                correct_predictions += 1
            
            # Check if predictions match
            if dbn_prediction == vanilla_prediction:
                print(f"  ✓ Both models agree")
            else:
                print(f"  ✗ Models disagree")
                disagreements += 1
        
        print(f"\nSample Analysis Summary:")
        print(f"  DBN+RF correct predictions: {correct_predictions}/{len(sample_indices)} ({correct_predictions/len(sample_indices)*100:.1f}%)")
        print(f"  Model disagreements: {disagreements}/{len(sample_indices)} ({disagreements/len(sample_indices)*100:.1f}%)")
        
        # Final results summary
        print(f"\n{'='*80}")
        print("FINAL ENHANCED MODEL COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"Training data range: {year_suffix.replace('_', '-')}")
        print(f"Test data: 2025")
        print(f"Optimized DBN+RF model accuracy on 2025: {dbn_results['accuracy']:.4f} ({dbn_results['accuracy']*100:.2f}%)")
        print(f"Optimized Vanilla RF model accuracy on 2025: {vanilla_results['accuracy']:.4f} ({vanilla_results['accuracy']*100:.2f}%)")
        
        improvement = ((dbn_results['accuracy'] - vanilla_results['accuracy']) / vanilla_results['accuracy']) * 100
        if improvement > 0:
            print(f"DBN+RF shows {improvement:.2f}% improvement over Vanilla RF")
        else:
            print(f"Vanilla RF performs {abs(improvement):.2f}% better than DBN+RF")
        
        print(f"\nModel files saved:")
        print(f"  - {DBN_MODEL_PATH}")
        print(f"  - {VANILLA_MODEL_PATH}")
        print(f"  - optimized_model_comparison_results_{year_suffix}.csv")
        print(f"  - All plots saved in '{PLOTS_DIR}/' directory")
        
        target_accuracy = 0.86
        if dbn_results['accuracy'] >= target_accuracy:
            print(f"\n🎉 SUCCESS: DBN+RF achieved target accuracy of {target_accuracy*100:.0f}%!")
        elif vanilla_results['accuracy'] >= target_accuracy:
            print(f"\n🎉 SUCCESS: Vanilla RF achieved target accuracy of {target_accuracy*100:.0f}%!")
        else:
            best_acc = max(dbn_results['accuracy'], vanilla_results['accuracy'])
            print(f"\n📈 Best accuracy achieved: {best_acc:.4f} ({best_acc*100:.2f}%)")
            print(f"   Still need {(target_accuracy - best_acc)*100:.2f}% improvement to reach {target_accuracy*100:.0f}% target")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)