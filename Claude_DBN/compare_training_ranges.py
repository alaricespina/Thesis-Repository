#!/usr/bin/env python3
"""
Comparison Script: 2001-2024 vs 2011-2024 Training Data Performance

This script automatically runs both training data options and compares their performance
on 2025 test data to show the impact of training data range on model accuracy.
"""

import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def run_model_training(use_2011_onwards=False):
    """Run model training with specified data range"""
    year_range = "2011-2024" if use_2011_onwards else "2001-2024"
    print(f"\n{'='*60}")
    print(f"TRAINING MODELS WITH {year_range} DATA")
    print(f"{'='*60}")
    
    # Import and run the enhanced training
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from enhanced_data_preprocessor import EnhancedWeatherDataPreprocessor
    from optimized_dbn_rf_classifier import OptimizedDBNRandomForestClassifier, OptimizedVanillaRandomForestClassifier
    
    DATA_PATH = "../Data Source Files/"
    year_suffix = "2011_2024" if use_2011_onwards else "2001_2024"
    
    # Configuration
    DBN_MODEL_PATH = f"models/comparison_dbn_rf_model_{year_suffix}.pkl"
    VANILLA_MODEL_PATH = f"models/comparison_vanilla_rf_model_{year_suffix}.pkl"
    
    # Optimized configurations
    DBN_CONFIG = {
        'dbn_hidden_layers': [256, 128, 64],
        'dbn_learning_rate': 0.001,
        'dbn_epochs': 10,
        'dbn_batch_size': 128,
        'rf_n_estimators': 500,
        'rf_max_depth': 20,
        'rf_min_samples_split': 5,
        'rf_min_samples_leaf': 2,
        'rf_random_state': 42,
        'use_ensemble': True
    }
    
    VANILLA_RF_CONFIG = {
        'n_estimators': 500,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'use_ensemble': True
    }
    
    # Load and preprocess data
    print("Loading and preprocessing training data...")
    preprocessor = EnhancedWeatherDataPreprocessor(DATA_PATH, use_2011_onwards=use_2011_onwards)
    X_train, y_train = preprocessor.process_all()
    
    # Load test data
    print("Loading 2025 test data...")
    X_test, y_test = preprocessor.process_test_2025()
    
    class_names = preprocessor.get_class_names()
    
    # Train DBN model
    print("\nTraining DBN+RF model...")
    dbn_model = OptimizedDBNRandomForestClassifier(**DBN_CONFIG)
    dbn_model.fit(X_train, y_train)
    dbn_model.save_model(DBN_MODEL_PATH)
    
    # Train Vanilla RF model
    print("Training Vanilla RF model...")
    vanilla_model = OptimizedVanillaRandomForestClassifier(**VANILLA_RF_CONFIG)
    vanilla_model.fit(X_train, y_train)
    vanilla_model.save_model(VANILLA_MODEL_PATH)
    
    # Evaluate models
    print("\nEvaluating models on 2025 test data...")
    dbn_results = dbn_model.evaluate(X_test, y_test, class_names)
    vanilla_results = vanilla_model.evaluate(X_test, y_test, class_names)
    
    return {
        'year_range': year_range,
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'dbn_accuracy': dbn_results['accuracy'],
        'vanilla_accuracy': vanilla_results['accuracy'],
        'dbn_model_path': DBN_MODEL_PATH,
        'vanilla_model_path': VANILLA_MODEL_PATH
    }

def create_comparison_report(results_2001, results_2011):
    """Create a comprehensive comparison report"""
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TRAINING DATA RANGE COMPARISON")
    print(f"{'='*80}")
    
    # Create comparison DataFrame
    comparison_data = {
        'Training Range': [results_2001['year_range'], results_2011['year_range']],
        'Training Samples': [results_2001['training_samples'], results_2011['training_samples']],
        'DBN+RF Accuracy': [results_2001['dbn_accuracy'], results_2011['dbn_accuracy']],
        'Vanilla RF Accuracy': [results_2001['vanilla_accuracy'], results_2011['vanilla_accuracy']]
    }
    
    df = pd.DataFrame(comparison_data)
    print("\nResults Summary:")
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Calculate improvements
    print(f"\nDetailed Analysis:")
    print(f"Training data difference: {results_2001['training_samples'] - results_2011['training_samples']:,} samples")
    
    dbn_improvement = ((results_2001['dbn_accuracy'] - results_2011['dbn_accuracy']) / results_2011['dbn_accuracy']) * 100
    vanilla_improvement = ((results_2001['vanilla_accuracy'] - results_2011['vanilla_accuracy']) / results_2011['vanilla_accuracy']) * 100
    
    print(f"\nDBN+RF Performance:")
    print(f"  2001-2024: {results_2001['dbn_accuracy']:.4f} ({results_2001['dbn_accuracy']*100:.2f}%)")
    print(f"  2011-2024: {results_2011['dbn_accuracy']:.4f} ({results_2011['dbn_accuracy']*100:.2f}%)")
    if dbn_improvement > 0:
        print(f"  → 2001-2024 performs {dbn_improvement:.2f}% better")
    else:
        print(f"  → 2011-2024 performs {abs(dbn_improvement):.2f}% better")
    
    print(f"\nVanilla RF Performance:")
    print(f"  2001-2024: {results_2001['vanilla_accuracy']:.4f} ({results_2001['vanilla_accuracy']*100:.2f}%)")
    print(f"  2011-2024: {results_2011['vanilla_accuracy']:.4f} ({results_2011['vanilla_accuracy']*100:.2f}%)")
    if vanilla_improvement > 0:
        print(f"  → 2001-2024 performs {vanilla_improvement:.2f}% better")
    else:
        print(f"  → 2011-2024 performs {abs(vanilla_improvement):.2f}% better")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"training_range_comparison_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")
    
    # Create visualization
    create_comparison_plots(df, timestamp)
    
    # Recommendations
    print(f"\n{'='*50}")
    print("RECOMMENDATIONS")
    print(f"{'='*50}")
    
    best_dbn = "2001-2024" if results_2001['dbn_accuracy'] > results_2011['dbn_accuracy'] else "2011-2024"
    best_vanilla = "2001-2024" if results_2001['vanilla_accuracy'] > results_2011['vanilla_accuracy'] else "2011-2024"
    
    print(f"Best DBN+RF performance: {best_dbn}")
    print(f"Best Vanilla RF performance: {best_vanilla}")
    
    if best_dbn == best_vanilla:
        print(f"\n✓ Consistent result: {best_dbn} data range performs better for both models")
    else:
        print(f"\n⚠ Mixed results: Different optimal ranges for different models")
    
    return df

def create_comparison_plots(df, timestamp):
    """Create comparison visualizations"""
    # Set up the plotting style
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Training samples comparison
    ax1.bar(df['Training Range'], df['Training Samples'], color=['skyblue', 'lightcoral'])
    ax1.set_title('Training Data Size Comparison')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(df['Training Samples']):
        ax1.text(i, v + 500, f'{v:,}', ha='center', va='bottom')
    
    # Plot 2: DBN+RF Accuracy comparison
    ax2.bar(df['Training Range'], df['DBN+RF Accuracy'], color=['green', 'orange'])
    ax2.set_title('DBN+RF Model Accuracy Comparison')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(df['DBN+RF Accuracy']):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot 3: Vanilla RF Accuracy comparison
    ax3.bar(df['Training Range'], df['Vanilla RF Accuracy'], color=['purple', 'brown'])
    ax3.set_title('Vanilla RF Model Accuracy Comparison')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim(0, 1)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(df['Vanilla RF Accuracy']):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Plot 4: Side-by-side accuracy comparison
    x = range(len(df))
    width = 0.35
    
    ax4.bar([i - width/2 for i in x], df['DBN+RF Accuracy'], width, label='DBN+RF', color='green', alpha=0.7)
    ax4.bar([i + width/2 for i in x], df['Vanilla RF Accuracy'], width, label='Vanilla RF', color='blue', alpha=0.7)
    
    ax4.set_title('Model Accuracy Comparison by Training Range')
    ax4.set_ylabel('Accuracy')
    ax4.set_xlabel('Training Range')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['Training Range'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"training_range_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plots saved to: {plot_filename}")
    plt.show()

def main():
    """Main execution function"""
    print("=== AUTOMATED TRAINING DATA RANGE COMPARISON ===")
    print("This will train models with both 2001-2024 and 2011-2024 data")
    print("and compare their performance on 2025 test data.\n")
    
    response = input("Do you want to proceed? This will take some time. (y/n): ").strip().lower()
    if response != 'y':
        print("Comparison cancelled.")
        return
    
    try:
        # Run training with 2001-2024 data
        results_2001 = run_model_training(use_2011_onwards=False)
        
        # Run training with 2011-2024 data
        results_2011 = run_model_training(use_2011_onwards=True)
        
        # Create comprehensive comparison report
        comparison_df = create_comparison_report(results_2001, results_2011)
        
        print(f"\n{'='*60}")
        print("COMPARISON COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)