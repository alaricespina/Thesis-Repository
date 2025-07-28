#!/usr/bin/env python3
"""
Test script to verify DBN model loading and prediction functionality
"""

import sys
import os
import numpy as np
from datetime import datetime

# Test model loading
print("Testing DBN Model Loading...")
print("=" * 50)

try:
    import joblib
    print("✓ joblib imported successfully")
except ImportError:
    print("✗ joblib not available")
    sys.exit(1)

try:
    import sklearn
    print("✓ sklearn imported successfully")
except ImportError:
    print("✗ sklearn not available")
    sys.exit(1)

# Test model loading
try:
    dbn_model = joblib.load("FinalDBNRFCModel.pkl")
    print("✓ DBN Model loaded successfully")
    print(f"Model type: {type(dbn_model)}")
    
    # Check if model has predict method
    if hasattr(dbn_model, 'predict'):
        print("✓ Model has predict method")
    else:
        print("✗ Model does not have predict method")
        
except Exception as e:
    print(f"✗ Failed to load DBN model: {e}")
    sys.exit(1)

# Test prediction with dummy data
print("\nTesting Model Prediction...")
print("=" * 50)

try:
    # Create dummy input data similar to what prepareData() would generate
    # The exact shape depends on the model training, but let's try common shapes
    dummy_shapes = [
        (1, 42),   # 6 features * 7 window - 1 (common DBN input)
        (1, 35),   # 5 features * 7 window
        (1, 28),   # 4 features * 7 
        (1, 21),   # 3 features * 7
        (1, 14),   # 2 features * 7
        (1, 7),    # 1 feature * 7
    ]
    
    successful_shape = None
    for shape in dummy_shapes:
        try:
            dummy_data = np.random.random(shape)
            prediction = dbn_model.predict(dummy_data)
            print(f"✓ Prediction successful with input shape {shape}")
            print(f"  Output shape: {prediction.shape}")
            print(f"  Prediction values: {prediction}")
            
            # Try to get class prediction
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                weather_conditions = ["Cloudy", "Rainy", "Sunny", "Windy"]
                pred_class = weather_conditions[np.argmax(prediction[0])]
                print(f"  Predicted weather class: {pred_class}")
            
            successful_shape = shape
            break
            
        except Exception as e:
            print(f"✗ Failed with shape {shape}: {e}")
            continue
    
    if successful_shape:
        print(f"\n✓ Model is working correctly with input shape: {successful_shape}")
    else:
        print("\n✗ Could not find compatible input shape for the model")
        
except Exception as e:
    print(f"✗ Error during prediction testing: {e}")

print("\nModel Test Complete!")
print("=" * 50)