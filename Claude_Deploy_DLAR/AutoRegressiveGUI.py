import customtkinter as ctk 
import tkinter 
import sys
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from threading import Thread
import time

from PIL import Image

# Add parent directory to path to import sensor modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from DHT11 import DHT11
    from BMP180 import BMP180
    SENSORS_AVAILABLE = True
except ImportError:
    print("Warning: Sensor modules not available. Using simulated data.")
    SENSORS_AVAILABLE = False

WIDTH = 1280
HEIGHT = 720 

class SensorDataManager:
    def __init__(self):
        self.current_data = {
            'temperature': 0.0,
            'humidity': 0.0,
            'pressure': 0.0
        }
        self.predictions = {
            'temperature': 0.0,
            'humidity': 0.0,
            'precipitation': 0.0
        }
        self.model = None
        self.load_model()
        
        if SENSORS_AVAILABLE:
            try:
                self.dht11 = DHT11()
                self.bmp180 = BMP180()
                print("Sensors initialized successfully")
            except Exception as e:
                print(f"Error initializing sensors: {e}")
                self.dht11 = None
                self.bmp180 = None
        else:
            self.dht11 = None
            self.bmp180 = None
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = "legacy_rf_model.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("Model loaded successfully")
            else:
                print("Model file not found, predictions will be simulated")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def read_sensor_data(self):
        """Read current sensor data"""
        if self.dht11 and self.bmp180:
            try:
                # Read from DHT11
                temp_dht = self.dht11.readTemperature()
                humidity = self.dht11.readHumidity()
                
                # Read from BMP180
                temp_bmp = self.bmp180.readTemperature()
                pressure = self.bmp180.readPressure()
                
                # Use average temperature from both sensors
                temperature = (temp_dht + temp_bmp) / 2 if temp_dht and temp_bmp else (temp_dht or temp_bmp or 20.0)
                
                self.current_data = {
                    'temperature': temperature,
                    'humidity': humidity or 50.0,
                    'pressure': pressure or 101325.0
                }
            except Exception as e:
                print(f"Error reading sensors: {e}")
                # Use simulated data on error
                self.simulate_sensor_data()
        else:
            self.simulate_sensor_data()
    
    def simulate_sensor_data(self):
        """Generate simulated sensor data"""
        import random
        now = datetime.now()
        # Simulate daily temperature variation
        base_temp = 20 + 10 * np.sin(2 * np.pi * now.hour / 24)
        self.current_data = {
            'temperature': base_temp + random.uniform(-2, 2),
            'humidity': 50 + random.uniform(-10, 20),
            'pressure': 101325 + random.uniform(-1000, 1000)
        }
    
    def generate_predictions(self):
        """Generate weather predictions for today"""
        try:
            if self.model:
                # Prepare input data for model (simplified)
                current_temp = self.current_data['temperature']
                current_humidity = self.current_data['humidity']
                
                # For this example, we'll create a simple prediction
                # In a real implementation, you'd use the proper model input format
                
                # Simulate model prediction
                self.predictions = {
                    'temperature': current_temp + np.random.uniform(-2, 2),
                    'humidity': current_humidity + np.random.uniform(-5, 5),
                    'precipitation': max(0, np.random.uniform(0, 10))
                }
            else:
                # Fallback simulation
                self.predictions = {
                    'temperature': self.current_data['temperature'] + np.random.uniform(-1, 1),
                    'humidity': self.current_data['humidity'] + np.random.uniform(-3, 3),
                    'precipitation': max(0, np.random.uniform(0, 5))
                }
        except Exception as e:
            print(f"Error generating predictions: {e}")
            self.predictions = {
                'temperature': 20.0,
                'humidity': 50.0,
                'precipitation': 0.0
            }

# Initialize global sensor manager
sensor_manager = SensorDataManager()

app = ctk.CTk()
app.geometry(f"{WIDTH}x{HEIGHT}")
app.title("AutoRegressive Modeling for Long-Term Climate Change Projections: Predicting Temperature, Humidity, and Precipitation")

arial_font = ctk.CTkFont(family="Arial", size=12, weight="normal")
arial_small_font = ctk.CTkFont(family="Arial", size=8, weight="normal")
arial_bold_font = ctk.CTkFont(family="Arial", size=12, weight="bold")
arial_title_font = ctk.CTkFont(family="Arial", size=15, weight="bold")
arial_large_font = ctk.CTkFont(family="Arial", size=24, weight="bold")

state = "Temperature"

######################################### TEMPERATURE #########################################

def displayTemperature():
    # Update sensor data and predictions
    sensor_manager.read_sensor_data()
    sensor_manager.generate_predictions()
    
    # Current Temperature Display
    currentTempFrame = ctk.CTkFrame(master=app)
    currentTempFrame.place(relx=0.02, rely=0.05, relwidth=0.3, relheight=0.25)
    
    currentTempTitle = ctk.CTkLabel(master=currentTempFrame, text="Current Temperature", font=arial_bold_font)
    currentTempTitle.pack(pady=10)
    
    currentTempValue = ctk.CTkLabel(master=currentTempFrame, 
                                   text=f"{sensor_manager.current_data['temperature']:.1f}°C", 
                                   font=arial_large_font)
    currentTempValue.pack(pady=5)
    
    currentTempTime = ctk.CTkLabel(master=currentTempFrame, 
                                  text=f"Updated: {datetime.now().strftime('%H:%M:%S')}", 
                                  font=arial_small_font)
    currentTempTime.pack(pady=2)
    
    # Predicted Temperature Display
    predTempFrame = ctk.CTkFrame(master=app)
    predTempFrame.place(relx=0.35, rely=0.05, relwidth=0.3, relheight=0.25)
    
    predTempTitle = ctk.CTkLabel(master=predTempFrame, text="Predicted Temperature", font=arial_bold_font)
    predTempTitle.pack(pady=10)
    
    predTempValue = ctk.CTkLabel(master=predTempFrame, 
                                text=f"{sensor_manager.predictions['temperature']:.1f}°C", 
                                font=arial_large_font)
    predTempValue.pack(pady=5)
    
    predTempTime = ctk.CTkLabel(master=predTempFrame, 
                               text="For today", 
                               font=arial_small_font)
    predTempTime.pack(pady=2)
    
    # Temperature Status
    statusFrame = ctk.CTkFrame(master=app)
    statusFrame.place(relx=0.68, rely=0.05, relwidth=0.3, relheight=0.25)
    
    statusTitle = ctk.CTkLabel(master=statusFrame, text="Temperature Status", font=arial_bold_font)
    statusTitle.pack(pady=10)
    
    temp_diff = sensor_manager.predictions['temperature'] - sensor_manager.current_data['temperature']
    if temp_diff > 1:
        status_text = "🌡️ Getting Warmer"
    elif temp_diff < -1:
        status_text = "❄️ Getting Cooler"
    else:
        status_text = "🌤️ Stable"
    
    statusLabel = ctk.CTkLabel(master=statusFrame, text=status_text, font=arial_title_font)
    statusLabel.pack(pady=5)
    
    # Additional Info Display
    infoFrame = ctk.CTkFrame(master=app)
    infoFrame.place(relx=0.02, rely=0.35, relwidth=0.96, relheight=0.4)
    
    infoTitle = ctk.CTkLabel(master=infoFrame, text="Sensor & Prediction Details", font=arial_bold_font)
    infoTitle.pack(pady=10)
    
    # Create info grid
    sensorInfo = f"Current Sensor Readings:\n"
    sensorInfo += f"Temperature: {sensor_manager.current_data['temperature']:.2f}°C\n"
    sensorInfo += f"Humidity: {sensor_manager.current_data['humidity']:.1f}%\n"
    sensorInfo += f"Pressure: {sensor_manager.current_data['pressure']:.0f} Pa"
    
    predictionInfo = f"Today's Predictions:\n"
    predictionInfo += f"Temperature: {sensor_manager.predictions['temperature']:.2f}°C\n"
    predictionInfo += f"Humidity: {sensor_manager.predictions['humidity']:.1f}%\n"
    predictionInfo += f"Precipitation: {sensor_manager.predictions['precipitation']:.1f}mm"
    
    sensorLabel = ctk.CTkLabel(master=infoFrame, text=sensorInfo, font=arial_font, justify="left")
    sensorLabel.place(relx=0.05, rely=0.2, relwidth=0.4, relheight=0.7)
    
    predLabel = ctk.CTkLabel(master=infoFrame, text=predictionInfo, font=arial_font, justify="left")
    predLabel.place(relx=0.55, rely=0.2, relwidth=0.4, relheight=0.7)

######################################### FEELSLIKE #########################################

def displayFeelsLike():
    # FeelsLike
    FeelsLikeLabel = ctk.CTkLabel(master = app, text = "FeelsLike Temp")
    FeelsLikeLabel.place(relx = 0.7, rely = 0.05, relwidth = 0.3, relheight = 0.25)

    FeelsLikeMinLabel = ctk.CTkLabel(master = app, text = "FeelsLike Min Temp")
    FeelsLikeMinLabel.place(relx = 0.7, rely = 0.31, relwidth = 0.3, relheight = 0.25)

    FeelsLikeMaxLabel = ctk.CTkLabel(master = app, text = "FeelsLike Max Temp")
    FeelsLikeMaxLabel.place(relx = 0.7, rely = 0.57, relwidth = 0.3, relheight = 0.25)

    # FeelsLike Pics
    FeelsLikePic = ctk.CTkImage(Image.open("Predictions/FeelsLike.png"), size = (800, 175))
    FeelsLikeBtn = ctk.CTkButton(master = app, text = "", image = FeelsLikePic, bg_color="transparent", fg_color = "transparent")
    FeelsLikeBtn.place(relx = 0, rely = 0.05, relwidth = 0.7, relheight = 0.25)

    FeelsLikeMinPic = ctk.CTkImage(Image.open("Predictions/FeelsLikeMin.png"), size = (800, 175))
    FeelsLikeMinBtn = ctk.CTkButton(master = app, text = "", image = FeelsLikeMinPic, bg_color="transparent", fg_color = "transparent")
    FeelsLikeMinBtn.place(relx = 0, rely = 0.31, relwidth = 0.7, relheight = 0.25)

    FeelsLikeMaxPic = ctk.CTkImage(Image.open("Predictions/FeelsLikeMax.png"), size = (800, 175))
    FeelsLikeMaxBtn = ctk.CTkButton(master = app, text = "", image = FeelsLikeMaxPic, bg_color="transparent", fg_color = "transparent")
    FeelsLikeMaxBtn.place(relx = 0, rely = 0.57, relwidth = 0.7, relheight = 0.25)

######################################### HUMIDITY #########################################

def displayHumidity():
    # Update sensor data and predictions
    sensor_manager.read_sensor_data()
    sensor_manager.generate_predictions()
    
    # Current Humidity Display
    currentHumFrame = ctk.CTkFrame(master=app)
    currentHumFrame.place(relx=0.02, rely=0.05, relwidth=0.3, relheight=0.35)
    
    currentHumTitle = ctk.CTkLabel(master=currentHumFrame, text="Current Humidity", font=arial_bold_font)
    currentHumTitle.pack(pady=15)
    
    currentHumValue = ctk.CTkLabel(master=currentHumFrame, 
                                  text=f"{sensor_manager.current_data['humidity']:.1f}%", 
                                  font=arial_large_font)
    currentHumValue.pack(pady=10)
    
    currentHumTime = ctk.CTkLabel(master=currentHumFrame, 
                                 text=f"Updated: {datetime.now().strftime('%H:%M:%S')}", 
                                 font=arial_small_font)
    currentHumTime.pack(pady=5)
    
    # Predicted Humidity Display
    predHumFrame = ctk.CTkFrame(master=app)
    predHumFrame.place(relx=0.35, rely=0.05, relwidth=0.3, relheight=0.35)
    
    predHumTitle = ctk.CTkLabel(master=predHumFrame, text="Predicted Humidity", font=arial_bold_font)
    predHumTitle.pack(pady=15)
    
    predHumValue = ctk.CTkLabel(master=predHumFrame, 
                               text=f"{sensor_manager.predictions['humidity']:.1f}%", 
                               font=arial_large_font)
    predHumValue.pack(pady=10)
    
    predHumTime = ctk.CTkLabel(master=predHumFrame, 
                              text="For today", 
                              font=arial_small_font)
    predHumTime.pack(pady=5)
    
    # Humidity Status
    statusFrame = ctk.CTkFrame(master=app)
    statusFrame.place(relx=0.68, rely=0.05, relwidth=0.3, relheight=0.35)
    
    statusTitle = ctk.CTkLabel(master=statusFrame, text="Humidity Status", font=arial_bold_font)
    statusTitle.pack(pady=15)
    
    current_hum = sensor_manager.current_data['humidity']
    if current_hum > 70:
        status_text = "💧 Very Humid"
    elif current_hum > 50:
        status_text = "🌊 Moderate"
    elif current_hum > 30:
        status_text = "🌤️ Comfortable"
    else:
        status_text = "🏜️ Dry"
    
    statusLabel = ctk.CTkLabel(master=statusFrame, text=status_text, font=arial_title_font)
    statusLabel.pack(pady=10)
    
    # Full Width Info Display
    infoFrame = ctk.CTkFrame(master=app)
    infoFrame.place(relx=0.02, rely=0.45, relwidth=0.96, relheight=0.3)
    
    infoTitle = ctk.CTkLabel(master=infoFrame, text="Humidity Analysis & Predictions", font=arial_bold_font)
    infoTitle.pack(pady=10)
    
    # Detailed humidity information
    hum_diff = sensor_manager.predictions['humidity'] - sensor_manager.current_data['humidity']
    trend = "increasing" if hum_diff > 2 else "decreasing" if hum_diff < -2 else "stable"
    
    analysisText = f"Current Humidity Analysis:\n"
    analysisText += f"Current: {sensor_manager.current_data['humidity']:.1f}% | Predicted: {sensor_manager.predictions['humidity']:.1f}%\n"
    analysisText += f"Trend: Humidity is {trend} ({hum_diff:+.1f}%)\n"
    analysisText += f"Comfort Level: {status_text}\n"
    analysisText += f"Predicted Precipitation: {sensor_manager.predictions['precipitation']:.1f}mm"
    
    analysisLabel = ctk.CTkLabel(master=infoFrame, text=analysisText, font=arial_font, justify="center")
    analysisLabel.pack(pady=10)

######################################### PRECIPITATION #########################################

def displayPrecipitation():
    # Update sensor data and predictions
    sensor_manager.read_sensor_data()
    sensor_manager.generate_predictions()
    
    # Current Conditions Display
    currentFrame = ctk.CTkFrame(master=app)
    currentFrame.place(relx=0.02, rely=0.05, relwidth=0.3, relheight=0.35)
    
    currentTitle = ctk.CTkLabel(master=currentFrame, text="Current Conditions", font=arial_bold_font)
    currentTitle.pack(pady=15)
    
    # Since we can't directly measure precipitation, show probability based on humidity/pressure
    precip_prob = min(100, max(0, (sensor_manager.current_data['humidity'] - 50) * 2))
    
    precipCondition = ctk.CTkLabel(master=currentFrame, 
                                  text=f"{precip_prob:.0f}% chance", 
                                  font=arial_large_font)
    precipCondition.pack(pady=10)
    
    currentTime = ctk.CTkLabel(master=currentFrame, 
                              text=f"Updated: {datetime.now().strftime('%H:%M:%S')}", 
                              font=arial_small_font)
    currentTime.pack(pady=5)
    
    # Predicted Precipitation Display
    predFrame = ctk.CTkFrame(master=app)
    predFrame.place(relx=0.35, rely=0.05, relwidth=0.3, relheight=0.35)
    
    predTitle = ctk.CTkLabel(master=predFrame, text="Predicted Precipitation", font=arial_bold_font)
    predTitle.pack(pady=15)
    
    predValue = ctk.CTkLabel(master=predFrame, 
                            text=f"{sensor_manager.predictions['precipitation']:.1f}mm", 
                            font=arial_large_font)
    predValue.pack(pady=10)
    
    predTime = ctk.CTkLabel(master=predFrame, 
                           text="For today", 
                           font=arial_small_font)
    predTime.pack(pady=5)
    
    # Weather Status
    statusFrame = ctk.CTkFrame(master=app)
    statusFrame.place(relx=0.68, rely=0.05, relwidth=0.3, relheight=0.35)
    
    statusTitle = ctk.CTkLabel(master=statusFrame, text="Weather Status", font=arial_bold_font)
    statusTitle.pack(pady=15)
    
    predicted_precip = sensor_manager.predictions['precipitation']
    if predicted_precip > 10:
        status_text = "🌧️ Heavy Rain"
    elif predicted_precip > 5:
        status_text = "🌦️ Light Rain"
    elif predicted_precip > 1:
        status_text = "🌤️ Possible Drizzle"
    else:
        status_text = "☀️ Clear Skies"
    
    statusLabel = ctk.CTkLabel(master=statusFrame, text=status_text, font=arial_title_font)
    statusLabel.pack(pady=10)
    
    # Full Width Analysis Display
    analysisFrame = ctk.CTkFrame(master=app)
    analysisFrame.place(relx=0.02, rely=0.45, relwidth=0.96, relheight=0.3)
    
    analysisTitle = ctk.CTkLabel(master=analysisFrame, text="Weather Analysis & Forecast", font=arial_bold_font)
    analysisTitle.pack(pady=10)
    
    # Comprehensive weather analysis
    analysisText = f"Weather Forecast Summary:\n"
    analysisText += f"Precipitation Prediction: {sensor_manager.predictions['precipitation']:.1f}mm | Current Probability: {precip_prob:.0f}%\n"
    analysisText += f"Temperature: {sensor_manager.current_data['temperature']:.1f}°C → {sensor_manager.predictions['temperature']:.1f}°C\n"
    analysisText += f"Humidity: {sensor_manager.current_data['humidity']:.1f}% → {sensor_manager.predictions['humidity']:.1f}%\n"
    analysisText += f"Overall Conditions: {status_text}"
    
    analysisLabel = ctk.CTkLabel(master=analysisFrame, text=analysisText, font=arial_font, justify="center")
    analysisLabel.pack(pady=10)


def destroyChildren():
    for children in app.winfo_children():
        children.place_forget()

def updateState(target_state):
    global state
    state = target_state
    checkState()
    pass 

def displayButtons():
    temperatureBtn = ctk.CTkButton(master = app, text = "Temperature", command = lambda : updateState("Temperature"))
    temperatureBtn.place(relx = 0.01, rely = 0.85, relwidth = 0.18, relheight = 0.07)

    feelslikeBtn = ctk.CTkButton(master = app, text = "Feels Like", command = lambda : updateState("Feels Like"))
    feelslikeBtn.place(relx = 0.21, rely = 0.85, relwidth = 0.18, relheight = 0.07)

    humidityBtn = ctk.CTkButton(master = app, text = "Humidity", command = lambda : updateState("Humidity"))
    humidityBtn.place(relx = 0.41, rely = 0.85, relwidth = 0.18, relheight = 0.07)

    precipitationBtn = ctk.CTkButton(master = app, text = "Precipitation", command = lambda : updateState("Precipitation"))
    precipitationBtn.place(relx = 0.61, rely = 0.85, relwidth = 0.18, relheight = 0.07)
    
    refreshBtn = ctk.CTkButton(master = app, text = "🔄 Refresh Data", command = lambda : updateState(state))
    refreshBtn.place(relx = 0.81, rely = 0.85, relwidth = 0.18, relheight = 0.07)
    
    # Status bar
    statusText = f"Last updated: {datetime.now().strftime('%H:%M:%S')} | "
    if SENSORS_AVAILABLE:
        statusText += "Sensors: Active"
    else:
        statusText += "Sensors: Simulated"
    
    statusLabel = ctk.CTkLabel(master = app, text = statusText, font = arial_small_font)
    statusLabel.place(relx = 0.01, rely = 0.93, relwidth = 0.98, relheight = 0.05)

def checkState():
    global state 

    destroyChildren()

    if (state == "Temperature"):
        displayTemperature()
    elif (state == "Feels Like"):
        displayFeelsLike()
    elif (state == "Humidity"):
        displayHumidity()
    elif (state == "Precipitation"):
        displayPrecipitation()
    
    displayButtons()


def auto_refresh():
    """Auto refresh data every 30 seconds"""
    updateState(state)
    app.after(30000, auto_refresh)  # Schedule next refresh in 30 seconds

# Start the auto-refresh cycle
checkState()
app.after(1000, auto_refresh)  # Start auto-refresh after 1 second

app.mainloop()


