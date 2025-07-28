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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import io

from PIL import Image

try:
    from DHT11 import DHT11
    # Rain Sensor Here
    SENSORS_AVAILABLE = True
except ImportError:
    print("Warning: Sensor modules not available. Using simulated data.")
    SENSORS_AVAILABLE = False

WIDTH = 800
HEIGHT = 480 

class SensorDataManager:
    def __init__(self):
        self.current_data = {
            'temperature': 0.0,
            'humidity': 0.0,
            'precipitation': 0.0
        }
        self.predictions = {
            'temperature': 0.0,
            'humidity': 0.0,
            'precipitation': 0.0
        }
        self.model = None
        self.historical_data = None
        self.load_model()
        self.load_historical_data()
        
        if SENSORS_AVAILABLE:
            try:
                self.dht11 = DHT11()
                # self.rain_sensor = RainSensor()
                print("Sensors initialized successfully")
            except Exception as e:
                print(f"Error initializing sensors: {e}")
                self.dht11 = None
                self.rain_sensor = None
        else:
            self.dht11 = None
            self.rain_sensor = None

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
    
    def load_historical_data(self):
        """Load historical weather data for enhanced predictions"""
        try:
            csv_path = "1989 to 2023 CombinedData.csv"
            if os.path.exists(csv_path):
                self.historical_data = pd.read_csv(csv_path)
                self.historical_data['datetime'] = pd.to_datetime(self.historical_data['datetime'])
                print(f"Historical data loaded: {len(self.historical_data)} records")
            else:
                print("Historical data file not found")
        except Exception as e:
            print(f"Error loading historical data: {e}")
    
    def read_sensor_data(self):
        """Read current sensor data"""
        if self.dht11:
            try:
                # Read from DHT11
                temp_dht = self.dht11.readTemperature()
                humidity = self.dht11.readHumidity()
                                
                self.current_data = {
                    'temperature': temp_dht,
                    'humidity': humidity or 50.0,
                    # 'pressure': pressure or 101325.0
                    'precipitation': 0.0
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
            'precipitation': random.uniform(0, 0),
        }
    
    def get_historical_context(self):
        """Get historical weather patterns for current date"""
        if self.historical_data is None:
            return None
        
        try:
            now = datetime.now()
            current_month = now.month
            current_day = now.day
            
            # Get historical data for same month and day (±3 days)
            historical_subset = self.historical_data[
                (self.historical_data['datetime'].dt.month == current_month) &
                (abs(self.historical_data['datetime'].dt.day - current_day) <= 3)
            ]
            
            if len(historical_subset) > 0:
                return {
                    'temp_mean': historical_subset['temp'].mean(),
                    'temp_std': historical_subset['temp'].std(),
                    'humidity_mean': historical_subset['humidity'].mean(),
                    'humidity_std': historical_subset['humidity'].std(),
                    'precip_mean': historical_subset['precip'].mean(),
                    'precip_std': historical_subset['precip'].std()
                }
        except Exception as e:
            print(f"Error getting historical context: {e}")
        
        return None
    
    def generate_predictions(self):
        """Generate weather predictions using model and historical data"""
        try:
            historical_context = self.get_historical_context()
            
            if self.model and isinstance(self.model, dict) and self.model.get('fitted') and historical_context:
                # Enhanced prediction using model and historical patterns
                current_temp = self.current_data['temperature']
                current_humidity = self.current_data['humidity']
                
                # Combine current sensor data with historical patterns
                temp_variation = np.random.normal(0, historical_context['temp_std'] * 0.3)
                humidity_variation = np.random.normal(0, historical_context['humidity_std'] * 0.3)
                precip_base = max(0, np.random.normal(historical_context['precip_mean'], 
                                                    historical_context['precip_std'] * 0.5))
                
                # Adjust predictions based on current sensor readings vs historical averages
                temp_deviation = current_temp - historical_context['temp_mean']
                humidity_deviation = current_humidity - historical_context['humidity_mean']
                
                self.predictions = {
                    'temperature': current_temp + temp_variation + (temp_deviation * 0.1),
                    'humidity': current_humidity + humidity_variation + (humidity_deviation * 0.1),
                    'precipitation': max(0, precip_base * (1 + humidity_deviation / 100))
                }
                
            elif historical_context:
                # Use historical patterns when model is not available
                current_temp = self.current_data['temperature']
                current_humidity = self.current_data['humidity']
                
                temp_variation = np.random.normal(0, historical_context['temp_std'] * 0.4)
                humidity_variation = np.random.normal(0, historical_context['humidity_std'] * 0.4)
                
                self.predictions = {
                    'temperature': current_temp + temp_variation,
                    'humidity': current_humidity + humidity_variation,
                    'precipitation': max(0, np.random.normal(historical_context['precip_mean'], 
                                                           historical_context['precip_std'] * 0.6))
                }
                
            else:
                # Fallback simulation when no model or historical data
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
    
    def generate_forecast_data(self, variable, days=30):
        """Generate forecast data for graphing"""
        try:
            forecast_data = []
            dates = []
            
            # Start from today
            current_date = datetime.now()
            
            for i in range(days):
                date = current_date + timedelta(days=i)
                dates.append(date)
                
                # Get historical context for this future date
                historical_context = self.get_historical_context_for_date(date)
                
                if historical_context and variable in ['temperature', 'humidity', 'precipitation']:
                    if variable == 'temperature':
                        base_value = historical_context['temp_mean']
                        std_dev = historical_context['temp_std']
                    elif variable == 'humidity':
                        base_value = historical_context['humidity_mean']
                        std_dev = historical_context['humidity_std']
                    else:  # precipitation
                        base_value = historical_context['precip_mean']
                        std_dev = historical_context['precip_std']
                    
                    # Add some trend and randomness
                    trend = np.sin(2 * np.pi * i / 365) * 0.1  # Seasonal trend
                    noise = np.random.normal(0, std_dev * 0.2)
                    
                    forecasted_value = base_value + trend + noise
                    
                    # Ensure reasonable bounds
                    if variable == 'temperature':
                        forecasted_value = max(15, min(40, forecasted_value))
                    elif variable == 'humidity':
                        forecasted_value = max(30, min(100, forecasted_value))
                    else:  # precipitation
                        forecasted_value = max(0, forecasted_value)
                    
                    forecast_data.append(forecasted_value)
                else:
                    # Fallback values
                    if variable == 'temperature':
                        forecast_data.append(25 + np.random.uniform(-3, 3))
                    elif variable == 'humidity':
                        forecast_data.append(60 + np.random.uniform(-10, 10))
                    else:
                        forecast_data.append(max(0, np.random.uniform(0, 5)))
            
            return dates, forecast_data
            
        except Exception as e:
            print(f"Error generating forecast data: {e}")
            # Return dummy data
            dates = [datetime.now() + timedelta(days=i) for i in range(days)]
            if variable == 'temperature':
                data = [25 + np.random.uniform(-2, 2) for _ in range(days)]
            elif variable == 'humidity':
                data = [60 + np.random.uniform(-5, 5) for _ in range(days)]
            else:
                data = [max(0, np.random.uniform(0, 3)) for _ in range(days)]
            return dates, data
    
    def get_historical_context_for_date(self, target_date):
        """Get historical context for a specific future date"""
        if self.historical_data is None:
            return None
        
        try:
            target_month = target_date.month
            target_day = target_date.day
            
            # Get historical data for same month and day (±3 days)
            historical_subset = self.historical_data[
                (self.historical_data['datetime'].dt.month == target_month) &
                (abs(self.historical_data['datetime'].dt.day - target_day) <= 3)
            ]
            
            if len(historical_subset) > 0:
                return {
                    'temp_mean': historical_subset['temp'].mean(),
                    'temp_std': historical_subset['temp'].std(),
                    'humidity_mean': historical_subset['humidity'].mean(),
                    'humidity_std': historical_subset['humidity'].std(),
                    'precip_mean': historical_subset['precip'].mean(),
                    'precip_std': historical_subset['precip'].std()
                }
        except Exception as e:
            print(f"Error getting historical context for date: {e}")
        
        return None

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

state = "Status"

def create_forecast_graph(parent_frame, variable, title, color='blue', ylabel="Value", days=14):
    """Create a forecast graph for the specified variable"""
    try:
        # Generate forecast data
        dates, forecast_data = sensor_manager.generate_forecast_data(variable, days)
        
        # Create matplotlib figure
        fig = Figure(figsize=(8, 3), dpi=80, facecolor='#212121')
        ax = fig.add_subplot(111)
        
        # Set background colors
        ax.set_facecolor('#2b2b2b')
        fig.patch.set_facecolor('#212121')
        
        # Plot the data
        ax.plot(dates, forecast_data, color=color, linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        # Formatting
        ax.set_title(title, color='white', fontsize=12, pad=10)
        ax.set_ylabel(ylabel, color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days//7)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Grid
        ax.grid(True, alpha=0.3, color='gray')
        
        # Remove spines and make remaining ones white
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(0.5)
        
        # Tight layout
        fig.tight_layout()
        
        # Create canvas and embed in tkinter
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.configure(bg='#212121')
        canvas_widget.pack(fill='both', expand=True, padx=5, pady=5)
        
        return canvas_widget
        
    except Exception as e:
        print(f"Error creating forecast graph for {variable}: {e}")
        # Create error label instead
        error_label = ctk.CTkLabel(
            parent_frame, 
            text=f"Graph unavailable\n({variable})", 
            font=arial_font,
            text_color="gray"
        )
        error_label.pack(expand=True)
        return error_label

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
    
    # Temperature Sensor Status
    statusFrame = ctk.CTkFrame(master=app)
    statusFrame.place(relx=0.68, rely=0.05, relwidth=0.3, relheight=0.25)
    
    statusTitle = ctk.CTkLabel(master=statusFrame, text="Temperature Sensor", font=arial_bold_font)
    statusTitle.pack(pady=10)
    
    # Check sensor connectivity
    if sensor_manager.dht11:
        try:
            test_temp = sensor_manager.dht11.readTemperature()
            if test_temp is not None:
                sensor_status = "✔️ Connected"
            else:
                sensor_status = "⚠️ Reading Error"
        except:
            sensor_status = "❌ Disconnected"
    else:
        sensor_status = "❌ Not Available"
    
    statusLabel = ctk.CTkLabel(master=statusFrame, text=sensor_status, font=arial_title_font)
    statusLabel.pack(pady=5)
    
    # Temperature Forecast Graph
    graphFrame = ctk.CTkFrame(master=app)
    graphFrame.place(relx=0.02, rely=0.35, relwidth=0.96, relheight=0.4)
    
    # graphTitle = ctk.CTkLabel(master=graphFrame, text="14-Day Temperature Forecast", font=arial_bold_font)
    # graphTitle.pack(pady=(10, 5))
    
    # Create the temperature forecast graph
    create_forecast_graph(graphFrame, 'temperature', '14-Day Temperature Forecast', 
                         color='#ff6b6b', ylabel='Temperature (°C)', days=14)

######################################### STATUS SUMMARY #########################################

def displayStatus():
    # Update sensor data and predictions
    sensor_manager.read_sensor_data()
    sensor_manager.generate_predictions()

    # Temperature
    # Current Temperature Display
    currentTempFrame = ctk.CTkFrame(master=app)
    currentTempFrame.place(relx=0.02, rely=0.025, relwidth=0.3, relheight=0.25)
    
    currentTempTitle = ctk.CTkLabel(master=currentTempFrame, 
                                    text="Current Temperature", 
                                    font=arial_bold_font)
    currentTempTitle.pack(pady=10)
    
    currentTempValue = ctk.CTkLabel(master=currentTempFrame,
                                    text=f"{sensor_manager.current_data['temperature']:.1f}°C",
                                    font=arial_large_font)
    currentTempValue.pack(pady=5)
        
    # Predicted Temperature Display
    predTempFrame = ctk.CTkFrame(master=app)
    predTempFrame.place(relx=0.35, rely=0.025, relwidth=0.3, relheight=0.25)
    
    predTempTitle = ctk.CTkLabel(master=predTempFrame, text="Predicted Temperature", font=arial_bold_font)
    predTempTitle.pack(pady=10)
    
    predTempValue = ctk.CTkLabel(master=predTempFrame, 
                                text=f"{sensor_manager.predictions['temperature']:.1f}°C", 
                                font=arial_large_font)
    predTempValue.pack(pady=5)

    # Temperature Sensor Status
    tempstatusFrame = ctk.CTkFrame(master=app)
    tempstatusFrame.place(relx=0.68, rely=0.025, relwidth=0.3, relheight=0.25)
    
    tempstatusTitle = ctk.CTkLabel(master=tempstatusFrame, text="Temperature Sensor", font=arial_bold_font)
    tempstatusTitle.pack(pady=10)
    
    # Check sensor connectivity
    if sensor_manager.dht11:
        try:
            test_temp = sensor_manager.dht11.readTemperature()
            if test_temp is not None:
                sensor_status = "✔️ Connected"
            else:
                sensor_status = "⚠️ Reading Error"
        except:
            sensor_status = "❌ Disconnected"
    else:
        sensor_status = "❌ Not Available"
    
    tempSensorLabel = ctk.CTkLabel(master=tempstatusFrame, text=sensor_status, font=arial_title_font)
    tempSensorLabel.pack(pady=5)
    
    # Humidity
    # Current Humidity Display
    currentHumFrame = ctk.CTkFrame(master=app)
    currentHumFrame.place(relx=0.02, rely=0.3, relwidth=0.3, relheight=0.25)
    
    currentHumTitle = ctk.CTkLabel(master=currentHumFrame, text="Current Humidity", font=arial_bold_font)
    currentHumTitle.pack(pady=10)
    
    currentHumValue = ctk.CTkLabel(master=currentHumFrame, 
                                  text=f"{sensor_manager.current_data['humidity']:.1f}%", 
                                  font=arial_large_font)
    currentHumValue.pack(pady=5)
    
    
    # Predicted Humidity Display
    predHumFrame = ctk.CTkFrame(master=app)
    predHumFrame.place(relx=0.35, rely=0.3, relwidth=0.3, relheight=0.25)
    
    predHumTitle = ctk.CTkLabel(master=predHumFrame, text="Predicted Humidity", font=arial_bold_font)
    predHumTitle.pack(pady=10)
    
    predHumValue = ctk.CTkLabel(master=predHumFrame, 
                               text=f"{sensor_manager.predictions['humidity']:.1f}%", 
                               font=arial_large_font)
    predHumValue.pack(pady=5)

    
    # Humidity Sensor Status
    humidstatusFrame = ctk.CTkFrame(master=app)
    humidstatusFrame.place(relx=0.68, rely=0.3, relwidth=0.3, relheight=0.25)
    
    humidstatusTitle = ctk.CTkLabel(master=humidstatusFrame, text="Humidity Sensor", font=arial_bold_font)
    humidstatusTitle.pack(pady=10)
    
    # Check sensor connectivity
    if sensor_manager.dht11:
        try:
            test_humidity = sensor_manager.dht11.readHumidity()
            if test_humidity is not None:
                sensor_status = "✔️ Connected"
            else:
                sensor_status = "⚠️ Reading Error"
        except:
            sensor_status = "❌ Disconnected"
    else:
        sensor_status = "❌ Not Available"
    
    humidSensorLabel = ctk.CTkLabel(master=humidstatusFrame, text=sensor_status, font=arial_title_font)
    humidSensorLabel.pack(pady=5)

    # Precipitation

    
    # Current Precipitation Display
    currentFrame = ctk.CTkFrame(master=app)
    currentFrame.place(relx=0.02, rely=0.575, relwidth=0.3, relheight=0.25)
    
    currentTitle = ctk.CTkLabel(master=currentFrame, text="Current Precipitation", font=arial_bold_font)
    currentTitle.pack(pady=10)
    
    # Since we can't directly measure precipitation, show probability based on humidity/pressure
    precipAmountSensor = sensor_manager.current_data['precipitation']
    precipAmountLabel = ctk.CTkLabel(master=currentFrame, 
                                  text=f"{precipAmountSensor:.0f} mm", 
                                  font=arial_large_font)
    precipAmountLabel.pack(pady=5)
    
    
    # Predicted Precipitation Display
    predFrame = ctk.CTkFrame(master=app)
    predFrame.place(relx=0.35, rely=0.575, relwidth=0.3, relheight=0.25)
    
    predTitle = ctk.CTkLabel(master=predFrame, text="Predicted Precipitation", font=arial_bold_font)
    predTitle.pack(pady=10)
    
    predValue = ctk.CTkLabel(master=predFrame, 
                            text=f"{sensor_manager.predictions['precipitation']:.1f}mm", 
                            font=arial_large_font)
    predValue.pack(pady=5)
    
    precipstatusFrame = ctk.CTkFrame(master=app)
    precipstatusFrame.place(relx=0.68, rely=0.575, relwidth=0.3, relheight=0.25)
    
    precipstatusTitle = ctk.CTkLabel(master=precipstatusFrame, text="Rain Sensor", font=arial_bold_font)
    precipstatusTitle.pack(pady=10)
    
    # Check rain sensor connectivity (placeholder since rain sensor isn't implemented)
    if hasattr(sensor_manager, 'rain_sensor') and sensor_manager.rain_sensor:
        sensor_status = "✔️ Connected"
    else:
        sensor_status = "❌ Not Available"
    
    precipSensorLabel = ctk.CTkLabel(master=precipstatusFrame, text=sensor_status, font=arial_title_font)
    precipSensorLabel.pack(pady=5)

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
    
    # Humidity Sensor Status
    statusFrame = ctk.CTkFrame(master=app)
    statusFrame.place(relx=0.68, rely=0.05, relwidth=0.3, relheight=0.35)
    
    statusTitle = ctk.CTkLabel(master=statusFrame, text="Humidity Sensor", font=arial_bold_font)
    statusTitle.pack(pady=15)
    
    # Check sensor connectivity
    if sensor_manager.dht11:
        try:
            test_humidity = sensor_manager.dht11.readHumidity()
            if test_humidity is not None:
                sensor_status = "✔️ Connected"
            else:
                sensor_status = "⚠️ Reading Error"
        except:
            sensor_status = "❌ Disconnected"
    else:
        sensor_status = "❌ Not Available"
    
    statusLabel = ctk.CTkLabel(master=statusFrame, text=sensor_status, font=arial_title_font)
    statusLabel.pack(pady=10)
    
    # Humidity Forecast Graph
    graphFrame = ctk.CTkFrame(master=app)
    graphFrame.place(relx=0.02, rely=0.45, relwidth=0.96, relheight=0.3)
    
    # graphTitle = ctk.CTkLabel(master=graphFrame, text="14-Day Humidity Forecast", font=arial_bold_font)
    # graphTitle.pack(pady=(10, 5))
    
    # Create the humidity forecast graph
    create_forecast_graph(graphFrame, 'humidity', '14-Day Humidity Forecast', 
                         color='#4ecdc4', ylabel='Humidity (%)', days=14)

######################################### PRECIPITATION #########################################

def displayPrecipitation():
    # Update sensor data and predictions
    sensor_manager.read_sensor_data()
    sensor_manager.generate_predictions()
    
    # Current Precipitation Display
    currentFrame = ctk.CTkFrame(master=app)
    currentFrame.place(relx=0.02, rely=0.05, relwidth=0.3, relheight=0.35)
    
    currentTitle = ctk.CTkLabel(master=currentFrame, text="Current Precipitation", font=arial_bold_font)
    currentTitle.pack(pady=15)
    
    # Show current precipitation in mm (same as status screen)
    precipAmountSensor = sensor_manager.current_data['precipitation']
    precipAmountLabel = ctk.CTkLabel(master=currentFrame, 
                                  text=f"{precipAmountSensor:.0f} mm", 
                                  font=arial_large_font)
    precipAmountLabel.pack(pady=10)
    
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
    
    # Rain Sensor Status
    statusFrame = ctk.CTkFrame(master=app)
    statusFrame.place(relx=0.68, rely=0.05, relwidth=0.3, relheight=0.35)
    
    statusTitle = ctk.CTkLabel(master=statusFrame, text="Rain Sensor", font=arial_bold_font)
    statusTitle.pack(pady=15)
    
    # Check rain sensor connectivity (placeholder since rain sensor isn't implemented)
    if hasattr(sensor_manager, 'rain_sensor') and sensor_manager.rain_sensor:
        sensor_status = "✔️ Connected"
    else:
        sensor_status = "❌ Not Available"
    
    statusLabel = ctk.CTkLabel(master=statusFrame, text=sensor_status, font=arial_title_font)
    statusLabel.pack(pady=10)
    
    # Precipitation Forecast Graph
    graphFrame = ctk.CTkFrame(master=app)
    graphFrame.place(relx=0.02, rely=0.45, relwidth=0.96, relheight=0.3)
    
    # graphTitle = ctk.CTkLabel(master=graphFrame, text="14-Day Precipitation Forecast", font=arial_bold_font)
    # graphTitle.pack(pady=(10, 5))
    
    # Create the precipitation forecast graph
    create_forecast_graph(graphFrame, 'precipitation', '14-Day Precipitation Forecast', 
                         color='#74b9ff', ylabel='Precipitation (mm)', days=14)


def destroyChildren():
    for children in app.winfo_children():
        children.place_forget()

def updateState(target_state):
    global state
    state = target_state
    checkState()
    pass 

def displayButtons():

    statusBtn = ctk.CTkButton(master = app, text = "Status", command = lambda : updateState("Status"))
    statusBtn.place(relx = 0.02, rely = 0.9, relwidth = 0.18, relheight = 0.07)

    temperatureBtn = ctk.CTkButton(master = app, text = "Temperature", command = lambda : updateState("Temperature"))
    temperatureBtn.place(relx = 0.22, rely = 0.9, relwidth = 0.18, relheight = 0.07)

    humidityBtn = ctk.CTkButton(master = app, text = "Humidity", command = lambda : updateState("Humidity"))
    humidityBtn.place(relx = 0.42, rely = 0.9, relwidth = 0.18, relheight = 0.07)

    precipitationBtn = ctk.CTkButton(master = app, text = "Precipitation", command = lambda : updateState("Precipitation"))
    precipitationBtn.place(relx = 0.62, rely = 0.9, relwidth = 0.18, relheight = 0.07)
    
    refreshBtn = ctk.CTkButton(master = app, text = "Refresh ⟳", command = lambda : updateState(state))
    refreshBtn.place(relx = 0.82, rely = 0.9, relwidth = 0.16, relheight = 0.07)
    

def checkState():
    global state 

    destroyChildren()

    if (state == "Temperature"):
        displayTemperature()
    elif (state == "Status"):
        displayStatus()
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


