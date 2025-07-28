# AutoRegressive Long-term Climate Prediction System

## Overview
This system implements an AutoRegressive model for long-term climate change projections, predicting temperature, humidity, and precipitation using historical weather data from 1989-2023.

## Files Structure
- `AutoRegressiveGUI.py` - Main GUI application with sensor integration
- `DHT11.py` - Temperature and humidity sensor interface
- `RainSensor.py` - Precipitation sensor interface  
- `1989 to 2023 CombinedData.csv` - Historical weather dataset
- `legacy_rf_model.pkl` - Trained Random Forest model

## Features
- Real-time sensor data collection
- Historical weather pattern analysis
- 14-day forecast generation with caching
- Interactive graphs and predictions
- Smart refresh system preventing unnecessary recalculation

## Requirements
- Python 3.8+
- customtkinter
- pandas
- numpy
- matplotlib
- scikit-learn
- PIL

## Hardware Requirements
- Raspberry Pi
- DHT11 temperature/humidity sensor
- Rain detection sensor
- GPIO connections

## Usage
```bash
python3 AutoRegressiveGUI.py
```

## Model Performance
The system uses a Random Forest model trained on 24 years of historical weather data, providing accurate short-term and long-term climate predictions with seasonal pattern recognition.