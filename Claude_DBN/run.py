#!/usr/bin/env python3
"""
Deep Belief Network with Random Forest Classifier for Weather Condition Prediction

This script implements a complete pipeline for training and evaluating both a Deep Belief Network
with Random Forest classifier and a vanilla Random Forest classifier for comparison.
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

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessor import WeatherDataPreprocessor
from dbn_rf_classifier import DBNRandomForestClassifier
from vanilla_rf_classifier import VanillaRandomForestClassifier
from model_comparison import ModelComparison


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plot confusion matrix using seaborn"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


def plot_class_distribution(y, class_names, title="Class Distribution"):
    """Plot the distribution of classes in the dataset"""
    unique, counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar([class_names[i] for i in unique], counts)
    plt.title(title)
    plt.xlabel('Weather Conditions')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function"""
    print("=== Weather Classifier Comparison: DBN+RF vs Vanilla RF ===\n")
    
    # Ask user for training data range preference
    print("Training Data Options:")
    print("1. Use all historical data (2001-2024)")
    print("2. Use recent data only (2011-2024)")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    use_2011_onwards = choice == '2'
    year_suffix = "2011_2024" if use_2011_onwards else "2001_2024"
    
    # Configuration
    DATA_PATH = "../Data Source Files/"
    DBN_MODEL_PATH = f"models/trained_dbn_rf_model_{year_suffix}.pkl"
    VANILLA_MODEL_PATH = f"models/trained_vanilla_rf_model_{year_suffix}.pkl"
    
    # DBN Configuration
    DBN_CONFIG = {
        'dbn_hidden_layers': [128, 64, 32],
        'dbn_learning_rate': 0.01,
        'dbn_epochs': 10,
        'dbn_batch_size': 64
    }
    
    # Random Forest Configuration
    RF_CONFIG = {
        'rf_n_estimators': 200,
        'rf_max_depth': 15,
        'rf_random_state': 42
    }
    
    # Vanilla RF Configuration
    VANILLA_RF_CONFIG = {
        'n_estimators': 200,
        'max_depth': 15,
        'random_state': 42
    }
    
    try:
        # Step 1: Load and preprocess training data
        print("Step 1: Loading and preprocessing training data...")
        preprocessor = WeatherDataPreprocessor(DATA_PATH, use_2011_onwards=use_2011_onwards)
        X_train, y_train = preprocessor.process_all()
        
        class_names = preprocessor.get_class_names()
        print(f"\nTraining dataset loaded successfully!")
        print(f"Features shape: {X_train.shape}")
        print(f"Target shape: {y_train.shape}")
        print(f"Number of weather conditions: {len(class_names)}")
        
        # Plot class distribution
        plot_class_distribution(y_train, class_names, f"Training Data Distribution ({year_suffix})")
        
        # Step 2: Load and preprocess 2025 test data
        print("\nStep 2: Loading 2025 test data...")
        X_test, y_test = preprocessor.process_test_2025()
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set (2025): {X_test.shape[0]} samples")
        
        # Step 3: Check for existing models or train new ones
        print("\nStep 3: Training/Loading models...")
        
        # Initialize model comparison
        comparison = ModelComparison()
        
        # Train DBN + Random Forest Model
        print("\n--- Training DBN + Random Forest Model ---")
        if os.path.exists(DBN_MODEL_PATH):
            print(f"Loading existing DBN+RF model from {DBN_MODEL_PATH}")
            dbn_model = DBNRandomForestClassifier()
            dbn_model.load_model(DBN_MODEL_PATH)
            dbn_training_time = None
        else:
            print("Training new DBN+RF model...")
            dbn_model = DBNRandomForestClassifier(
                **DBN_CONFIG,
                **RF_CONFIG
            )
            
            start_time = time.time()
            dbn_model.fit(X_train, y_train)
            dbn_training_time = time.time() - start_time
            
            print(f"DBN+RF training completed in {dbn_training_time:.2f} seconds")
            dbn_model.save_model(DBN_MODEL_PATH)
        
        # Train Vanilla Random Forest Model
        print("\n--- Training Vanilla Random Forest Model ---")
        if os.path.exists(VANILLA_MODEL_PATH):
            print(f"Loading existing Vanilla RF model from {VANILLA_MODEL_PATH}")
            vanilla_model = VanillaRandomForestClassifier()
            vanilla_model.load_model(VANILLA_MODEL_PATH)
            vanilla_training_time = None
        else:
            print("Training new Vanilla RF model...")
            vanilla_model = VanillaRandomForestClassifier(**VANILLA_RF_CONFIG)
            
            start_time = time.time()
            vanilla_model.fit(X_train, y_train)
            vanilla_training_time = time.time() - start_time
            
            print(f"Vanilla RF training completed in {vanilla_training_time:.2f} seconds")
            vanilla_model.save_model(VANILLA_MODEL_PATH)
        
        # Step 4: Evaluate both models
        print("\nStep 4: Evaluating model performance...")
        
        # Evaluate DBN + RF Model
        print("\n--- DBN + RF Model Evaluation ---")
        dbn_results = dbn_model.evaluate(X_test, y_test, class_names)
        comparison.add_model_results(
            "DBN+RF", y_test, dbn_results['predictions'], 
            dbn_results['probabilities'], dbn_training_time
        )
        
        # Evaluate Vanilla RF Model
        print("\n--- Vanilla RF Model Evaluation ---")
        vanilla_results = vanilla_model.evaluate(X_test, y_test, class_names)
        comparison.add_model_results(
            "Vanilla RF", y_test, vanilla_results['predictions'],
            vanilla_results['probabilities'], vanilla_training_time
        )
        
        # Step 5: Model Comparison
        print("\nStep 5: Comprehensive Model Comparison...")
        
        # Generate comparison report
        comparison.generate_comparison_report()
        
        # Plot comparisons
        comparison.plot_accuracy_comparison()
        comparison.plot_metrics_comparison()
        comparison.plot_confusion_matrices(class_names)
        
        # Save comparison results
        comparison.save_results_to_csv("model_comparison_results.csv")
        
        # Step 6: Feature analysis for DBN model
        print("\nStep 6: Analyzing DBN feature extraction...")
        dbn_features_train = dbn_model.get_dbn_features(X_train)
        dbn_features_test = dbn_model.get_dbn_features(X_test)
        
        print(f"Original features: {X_train.shape[1]}")
        print(f"DBN extracted features: {dbn_features_train.shape[1]}")
        print(f"Feature reduction ratio: {dbn_features_train.shape[1] / X_train.shape[1]:.2f}")
        
        # Step 7: Sample predictions comparison
        print("\nStep 7: Sample predictions comparison...")
        sample_indices = np.random.choice(len(X_test), 5, replace=False)
        
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
            print(f"  DBN+RF Predicted: {class_names[dbn_prediction]} (confidence: {dbn_probabilities[dbn_prediction]:.3f})")
            print(f"  Vanilla RF Predicted: {class_names[vanilla_prediction]} (confidence: {vanilla_probabilities[vanilla_prediction]:.3f})")
            
            # Check if predictions match
            if dbn_prediction == vanilla_prediction:
                print(f"  ✓ Both models agree")
            else:
                print(f"  ✗ Models disagree")
        
        print(f"\n=== Model Comparison Completed! ===")
        print(f"Training data range: {year_suffix.replace('_', '-')}")
        print(f"Test data: 2025")
        print(f"DBN+RF model saved as: {DBN_MODEL_PATH}")
        print(f"Vanilla RF model saved as: {VANILLA_MODEL_PATH}")
        print(f"DBN+RF accuracy on 2025 data: {dbn_results['accuracy']:.4f} ({dbn_results['accuracy']*100:.2f}%)")
        print(f"Vanilla RF accuracy on 2025 data: {vanilla_results['accuracy']:.4f} ({vanilla_results['accuracy']*100:.2f}%)")
        
        # Calculate improvement
        improvement = ((dbn_results['accuracy'] - vanilla_results['accuracy']) / vanilla_results['accuracy']) * 100
        if improvement > 0:
            print(f"DBN+RF shows {improvement:.2f}% improvement over Vanilla RF")
        else:
            print(f"Vanilla RF performs {abs(improvement):.2f}% better than DBN+RF")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)