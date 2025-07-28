#!/usr/bin/env python3
"""
Example usage of DLAR with different prediction periods
"""

from run import DLARRunner

def main():
    """Examples of different prediction periods"""
    
    print("="*60)
    print("DLAR EXAMPLE USAGE - Different Prediction Periods")
    print("="*60)
    
    # Example 1: 5-year prediction
    print("\n🔮 Example 1: 5-Year Prediction")
    print("-" * 40)
    runner_5yr = DLARRunner(prediction_years=5)
    # You would call runner_5yr.load_data(), train models, etc.
    print(f"Configured for {runner_5yr.prediction_years} years ({runner_5yr.prediction_days} days)")
    
    # Example 2: 10-year prediction  
    print("\n🔮 Example 2: 10-Year Prediction")
    print("-" * 40)
    runner_10yr = DLARRunner(prediction_years=10)
    print(f"Configured for {runner_10yr.prediction_years} years ({runner_10yr.prediction_days} days)")
    
    # Example 3: 1-year prediction (faster for testing)
    print("\n🔮 Example 3: 1-Year Prediction (Testing)")
    print("-" * 40)
    runner_1yr = DLARRunner(prediction_years=1)
    print(f"Configured for {runner_1yr.prediction_years} years ({runner_1yr.prediction_days} days)")
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS:")
    print("="*60)
    print("1. Command line: python run.py [years]")
    print("   Examples:")
    print("   - python run.py      # Default 50 years")
    print("   - python run.py 5    # 5 years (1,825 days)")
    print("   - python run.py 1    # 1 year (365 days)")
    print("")
    print("2. Programmatic:")
    print("   runner = DLARRunner(prediction_years=5)")
    print("   runner.load_data()")
    print("   runner.train_vanilla_ar()")
    print("   # ... etc")
    print("")
    print("3. Output files will be named accordingly:")
    print("   - Vanilla_AR_5_year_predictions.csv")
    print("   - temp_5_year_sample.png")
    print("   - etc.")

if __name__ == "__main__":
    main()