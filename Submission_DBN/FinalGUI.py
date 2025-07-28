import sys
import os 
import pandas as pd 
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QSizePolicy, QTextEdit, QSpacerItem, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from random import randint 
from datetime import datetime, date, time
from CalendarWidgetClass import CalendarWidget
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, minmax_scale
import joblib 


app = QApplication(sys.argv)

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

in_board = False

try:
    from HALL import HALL_EFFECT
    from BMP180 import BMP180
    from DHT11 import DHT11
    HALL = HALL_EFFECT()
    BMP = BMP180()
    DHT = DHT11()
    in_board = True
    print("Succesfully imported Necessary Packages in RPI")

except Exception as E:
    print("Error: ", E)

############################################
# HELPER FUNCTIONS
############################################

class HelperFunctions():
    def __init__():
        pass 
    
    @staticmethod
    def setPlotLimits(plot, ymin, ymax):
        if ymin == 0 and ymax == 0:
            return 
        
        try:
            plot.setYRange(ymin, ymax)
        except Exception as E:
            print("Cannot set to", ymin, ymax)
        
        
        plot.getAxis('left').setPen('k')
        plot.showAxis('left', True)
        plot.getAxis('bottom').setTicks([[]])
        plot.getViewBox().setBackgroundColor('w')

    @staticmethod
    def rollArrayAndAppend(arr, val):
        if val == np.nan:
            return arr
        arr = np.roll(arr, -1)
        arr[-1] = val
        return arr
    
    @staticmethod
    def rollArrayAndIncrease(arr):
        arr = np.roll(arr, -1)
        arr[-1] = arr[-2] + 1
        return arr

############################################
# MAIN GUI
############################################

class MainGUI():
    def __init__(self):
        self.last_update_time = datetime.now()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.mainDriver)
        self.timer.start(250)
        self.initializeMainWindow()
        self.readHistoricalData()
        self.initializeCurrentFrame()
        self.initializeSiteFrame()
        self.initializeLocalFrame()
        # self.initializeHourlyFrame()
        self.initializeTrueFrame()
        self.generateInitialData()
        try:
            self.dbn_model = joblib.load("FinalDBNRFCModel.pkl")
            print("[Start Up] DBN Model loaded successfully")
        except Exception as e:
            print(f"[Error] Failed to load DBN model: {e}")
            self.dbn_model = None
        self.predicted_Flag = True
        
        
    
    def readHistoricalData(self):
        all_data = pd.DataFrame()
        for file in os.listdir("Data/Yearly"):
            df = pd.read_csv(os.path.join("Data/Yearly", file))
            all_data = pd.concat([all_data, df], axis=0)

        print("[Start Up] Getting Historical Data")
        self.historical_temperature_data = all_data['temp'].copy()
        self.historical_humidity_data = all_data['humidity'].copy()
        self.historical_pressure_data = all_data['sealevelpressure'].copy()
        self.historical_wind_data = all_data['windspeed'].copy()
        self.raw_df = all_data[["datetime","conditions", "tempmax", "tempmin", "temp", "humidity", "windspeed", "sealevelpressure"]].copy()
        self.pred_df = pd.read_csv("Data/April PREDICTIONS.csv")


    def adjustGridWidths(self):
        num_columns = self.mainLayout.columnCount()
        for column in range(num_columns):
            self.mainLayout.setColumnStretch(column, 1)

    # Main Window - Main Layout - [Dynamic Frame] - [Navigation Frame]
    def initializeMainWindow(self) -> None:
        self.window = QMainWindow()
        self.window.setWindowTitle("DeepBelief Networks - Weather Prediction")
        self.window.setMinimumSize(600, 400)

        self.tabWidget = QTabWidget()
        self.window.setCentralWidget(self.tabWidget)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.South)

        self.mainWidget = QWidget()
        self.localWidget = QWidget()
        self.PAGASAWidget = QWidget()
        self.siteWidget = QWidget()

        self.mainWidget.setStyleSheet("background-color: white;")
        self.mainWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        

        self.mainLayout = QGridLayout(self.mainWidget)

        self.tabWidget.addTab(self.mainWidget, "Current - Sensor Data")
        self.tabWidget.addTab(self.localWidget, "[Model] Local Predictions")
        self.tabWidget.addTab(self.PAGASAWidget, "[PAGASA] Weather Conditions")
        self.tabWidget.addTab(self.siteWidget, "Historical Predictions")


    # Current - Weather Prediction Frame
    def initializeCurrentWeatherPredictionFrame(self):
        ############################################
        # Weather Icons
        ############################################
        weatherGroupWidget = QWidget(self.mainWidget)
        self.mainLayout.addWidget(weatherGroupWidget, 1, 2, 1, 1)

        weatherGroupLayout = QGridLayout(weatherGroupWidget)

        cloudyPic = QPixmap("WeatherIcons/CLOUDY INACTIVE.png")
        rainyPic = QPixmap("WeatherIcons/RAINY INACTIVE.png")
        sunnyPic = QPixmap("WeatherIcons/SUNNY INACTIVE.png")
        rainySunnyPic = QPixmap("WeatherIcons/RAINY AND SUNNY INACTIVE.png")

        self.cloudyLabel = QLabel()
        self.rainyLabel = QLabel()
        self.sunnyLabel = QLabel()
        self.rainySunnyLabel = QLabel()
        self.cloudyLabel.setAlignment(Qt.AlignCenter)
        self.rainyLabel.setAlignment(Qt.AlignCenter)
        self.sunnyLabel.setAlignment(Qt.AlignCenter)
        self.rainySunnyLabel.setAlignment(Qt.AlignCenter)

        s = 50
        cloudyPic = cloudyPic.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        rainyPic = rainyPic.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        sunnyPic = sunnyPic.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        rainySunnyPic = rainySunnyPic.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.cloudyLabel.setPixmap(cloudyPic)
        self.rainyLabel.setPixmap(rainyPic)
        self.sunnyLabel.setPixmap(sunnyPic)
        self.rainySunnyLabel.setPixmap(rainySunnyPic)

        weatherGroupLayout.addWidget(self.cloudyLabel, 0, 0, 1, 1)
        weatherGroupLayout.addWidget(self.rainyLabel, 0, 1, 1, 1)
        weatherGroupLayout.addWidget(self.sunnyLabel, 1, 0, 1, 1)
        weatherGroupLayout.addWidget(self.rainySunnyLabel, 1, 1, 1, 1)

    # Current - Sensor Frame
    def initializeCurrentSensorFrame(self):
        ############################################
        # Sensor Graphs
        ############################################
        graphicsView = pg.GraphicsLayoutWidget()
        graphicsView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mainLayout.addWidget(graphicsView, 1, 0, 2, 2)

        self.temp_plot = graphicsView.addPlot(row=0, col=0, title="Temperature (°C)")
        self.humid_plot = graphicsView.addPlot(row=0, col=1, title="Humidity (%)")
        graphicsView.nextRow()
        self.pressure_plot = graphicsView.addPlot(row=1, col=0, title="Pressure (mBar)")
        self.wind_plot = graphicsView.addPlot(row=1, col=1, title="Wind Speed (kph)")

        for plot in [self.temp_plot, self.humid_plot, self.pressure_plot, self.wind_plot]:
            plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # Current - Console Frame
    def initializeCurrentConsoleFrame(self):
        self.console = QLabel("Info Here")
        self.mainLayout.addWidget(self.console, 1, 2, 2, 1)
        

    # Calls :
    # Weather Prediction Frame
    # Sensor Frame
    # Console Frame
    def initializeCurrentFrame(self):
        ############################################
        # Label
        ############################################
        self.siteActive = False
        self.localActive = False
        self.currentActive = True

        spanning_label = QLabel("Implementation of a Deep Belief Network with Sensor Correction Algorithm to predict Weather on a Raspberry Pi")
        spanning_label.setWordWrap(True)
        spanning_label.setAlignment(Qt.AlignCenter)
        self.mainLayout.addWidget(spanning_label, 0, 0, 1, 3)

        self.initializeCurrentSensorFrame()
        self.initializeCurrentConsoleFrame()

    def initializeHourlyFrame(self):
        return 

    def initializeSiteFrame(self):
        C = CalendarWidget(self.raw_df)
        self.siteLayout = QHBoxLayout(self.siteWidget)
        self.siteWidget.setLayout(self.siteLayout)
        self.siteLayout.addWidget(C)

    
    def initializeLocalFrame(self):
        C = CalendarWidget(self.pred_df, "predictionsClass", False)
        self.localLayout = QHBoxLayout(self.localWidget)
        self.localWidget.setLayout(self.localLayout)
        self.localLayout.addWidget(C)

    def initializeTrueFrame(self):
        C = CalendarWidget(self.pred_df, "conditionsClass", False)
        self.trueLayout = QHBoxLayout(self.PAGASAWidget)
        self.PAGASAWidget.setLayout(self.trueLayout)
        self.trueLayout.addWidget(C)
    

    def generateFakeIndex(self, arr):
        return [i for i in range(len(arr))]
    


    # Called After All [Dyanimc Frames] are done initializing
    def generateInitialData(self):
        ############################################
        # Seed Data
        ############################################
        self.temperatureData = []
        self.humidityData = []
        self.pressureData = []
        self.windData = []


        if not in_board:
            self.temperatureData = np.random.randint(0, 100, 100)
            self.humidityData = np.random.randint(0, 100, 100)
            self.pressureData = np.random.randint(0, 100, 100)
            self.windData = np.random.randint(0, 100, 100)
        else:
            self.temperatureData = np.zeros(100)
            self.humidityData = np.zeros(100)
            self.pressureData = np.zeros(100)
            self.windData = np.zeros(100)

        self.t_x = self.generateFakeIndex(self.temperatureData)
        self.h_x = self.generateFakeIndex(self.humidityData)
        self.p_x = self.generateFakeIndex(self.pressureData)
        self.w_x = self.generateFakeIndex(self.windData)

        self.temperature_curve = self.temp_plot.plot(self.t_x, self.temperatureData, pen='b', name='Temperature')
        self.humidity_curve = self.humid_plot.plot(self.h_x, self.humidityData, pen='b', name='Humidity')
        self.pressure_curve = self.pressure_plot.plot(self.p_x, self.pressureData, pen='b', name='Pressure')
        self.wind_curve = self.wind_plot.plot(self.w_x, self.windData, pen='b', name='Wind Speed')
        
        HelperFunctions.setPlotLimits(self.temp_plot, self.temperatureData.min() * 0.9, self.temperatureData.max() * 1.1)
        HelperFunctions.setPlotLimits(self.humid_plot, self.humidityData.min() * 0.9, self.humidityData.max() * 1.1)
        HelperFunctions.setPlotLimits(self.pressure_plot, self.pressureData.min() * 0.9, self.pressureData.max() * 1.1)
        HelperFunctions.setPlotLimits(self.wind_plot, self.windData.min() * 0.9, self.windData.max() * 1.1)

    def currentUpdate(self):
        
        self.t_x = HelperFunctions.rollArrayAndIncrease(self.t_x)
        self.h_x = HelperFunctions.rollArrayAndIncrease(self.h_x)
        self.p_x = HelperFunctions.rollArrayAndIncrease(self.p_x)
        self.w_x = HelperFunctions.rollArrayAndIncrease(self.w_x)
        
        if in_board:
            current_temp_data = DHT.readTemperature()
            current_humid_data = DHT.readHumidity()
            current_pressure_data = BMP.readPressure() * -1 / 1000 * 10
            current_wind_data = HALL.readSpeed()
        else:
            current_temp_data = randint(1, 100)
            current_humid_data = randint(1, 100)
            current_pressure_data = randint(1, 100)
            current_wind_data = randint(1, 100)

        self.temperatureData = HelperFunctions.rollArrayAndAppend(self.temperatureData, current_temp_data)
        self.humidityData = HelperFunctions.rollArrayAndAppend(self.humidityData, current_humid_data)
        self.pressureData = HelperFunctions.rollArrayAndAppend(self.pressureData, current_pressure_data)
        self.windData = HelperFunctions.rollArrayAndAppend(self.windData, current_wind_data)

        self.temperature_curve.setData(self.t_x, self.temperatureData)
        self.humidity_curve.setData(self.h_x, self.humidityData)
        self.pressure_curve.setData(self.p_x, self.pressureData)
        self.wind_curve.setData(self.w_x, self.windData)

        HelperFunctions.setPlotLimits(self.temp_plot, self.temperatureData.min() * 0.9, self.temperatureData.max() * 1.1)
        HelperFunctions.setPlotLimits(self.humid_plot, self.humidityData.min() * 0.9, self.humidityData.max() * 1.1)
        HelperFunctions.setPlotLimits(self.pressure_plot, self.pressureData.min() * 0.9, self.pressureData.max() * 1.1)
        HelperFunctions.setPlotLimits(self.wind_plot, self.windData.min() * 0.9, self.windData.max() * 1.1)

        try:
            currentMaxTemp = self.temperatureData.max()
            currentMinTemp = self.temperatureData.min()
            currentMaxHumid = self.humidityData.max()
            currentMinHumid = self.humidityData.min()
            currentMaxPressure = self.pressureData.max()
            currentMinPressure = self.pressureData.min()
            currentMaxWind = self.windData.max()
            currentMinWind = self.windData.min()
        except:
            currentMaxTemp = 0
            currentMinTemp = 0
            currentMaxHumid = 0
            currentMinHumid = 0
            currentMaxPressure = 0
            currentMinPressure = 0
            currentMaxWind = 0
            currentMinWind = 0
            self.print("-")

        # Create the message
        message = (f"Temperature: {current_temp_data:.2f}°C\n"
                   + f"Max: {currentMaxTemp:.2f}°C\n"
                   + f"Min: {currentMinTemp:.2f}°C\n\n"
                   + f"Humidity: {current_humid_data:.2f}%\n"
                   + f"Max: {currentMaxHumid:.2f}%\n"
                   + f"Min: {currentMinHumid:.2f}%\n\n"
                   + f"Pressure:{current_pressure_data:.2f}mBar\n"
                   + f"Max: {currentMaxPressure:.2f}mBar\n"
                   + f"Min: {currentMinPressure:.2f}mBar\n\n"
                   + f"Wind:{current_wind_data:.2f}kph\n"
                   + f"Max: {currentMaxWind:.2f}kph\n"
                   + f"Min: {currentMinWind:.2f}kph\n"
                   )

        self.console.setText(message)

        # Update last_update_time
        now = datetime.now()
        self.last_update_time = now
        if time(0, 0, 0) <= now.time() <= time(0, 0, 1):
            self.processPrediction()

        

    def prepareData(self):
        try:
            # ["tempmax", "tempmin", "temp", "humidity", "windspeed", "sealevelpressure"]
            if len(self.temperatureData) < 7 or len(self.humidityData) < 7:
                print("[Warning] Insufficient sensor data for prediction")
                return None
                
            recordLength = min(60 * 60 * 24 - 1, len(self.temperatureData) - 1)
            
            # Handle case where all temperature data might be the same
            temp_max = self.temperatureData.max() if self.temperatureData.max() != self.temperatureData.min() else self.temperatureData.max() + 0.1
            temp_min = self.temperatureData.min()
            
            maxTempData = [temp_max] * recordLength
            minTempData = [temp_min] * recordLength
            
            # Ensure arrays have the same length
            temp_data = self.temperatureData[-recordLength:] if len(self.temperatureData) > recordLength else self.temperatureData
            humid_data = self.humidityData[-recordLength:] if len(self.humidityData) > recordLength else self.humidityData  
            wind_data = self.windData[-recordLength:] if len(self.windData) > recordLength else self.windData
            pressure_data = self.pressureData[-recordLength:] if len(self.pressureData) > recordLength else self.pressureData
            
            min_length = min(len(temp_data), len(humid_data), len(wind_data), len(pressure_data))
            temp_data = temp_data[-min_length:]
            humid_data = humid_data[-min_length:]
            wind_data = wind_data[-min_length:]
            pressure_data = pressure_data[-min_length:]

            rawDataArr = [maxTempData[:min_length], minTempData[:min_length], temp_data, humid_data, wind_data, pressure_data]
            rawData = np.array(rawDataArr)
            
            inputXData = []
            WINDOW_LENGTH = 7
            
            # Ensure we have enough data for windowing
            if rawData.shape[1] < WINDOW_LENGTH:
                print(f"[Warning] Not enough data for window length {WINDOW_LENGTH}")
                return None
            
            for i in range(rawData.shape[0] - WINDOW_LENGTH + 1):
                t_row = []
                for j in range(WINDOW_LENGTH):
                    if i + j < rawData.shape[0]:
                        t_row.extend(rawData[i + j, :-1])
                inputXData.append(t_row)

            inputXData = np.array(inputXData)
            
            # Apply normalization if we have valid data
            if len(inputXData) > 0 and inputXData.shape[1] > 0:
                inputXData = minmax_scale(inputXData, feature_range=(0, 1))
                return inputXData
            else:
                print("[Warning] No valid input data after processing")
                return None
                
        except Exception as e:
            print(f"[Error] Data preparation failed: {e}")
            return None

    def processPrediction(self):
        if self.dbn_model is None:
            print("[Warning] DBN model not loaded, skipping prediction")
            return
            
        try:
            weatherConditions = ["Cloudy", "Rainy", "Sunny", "Windy"]
            
            # Prepare and preprocess data
            input_data = self.prepareData()
            if input_data is None or len(input_data) == 0:
                print("[Warning] No valid input data for prediction")
                return
                
            # Make prediction
            self.predictionOutput = self.dbn_model.predict(input_data)
            self.predictionClass = weatherConditions[np.argmax(self.predictionOutput, axis=1)[0]]
            
            # Create new data entry
            newData = {
                "datetime": date.today().strftime("%m/%d/%Y"),
                "sensor_windspeed": np.mean(self.windData[self.windData != 0]) if np.any(self.windData != 0) else 0,
                "sensor_pressure": np.mean(self.pressureData),
                "sensor_temperature": np.mean(self.temperatureData),
                "sensor_humidity": np.mean(self.humidityData),
                "predictions": self.predictionOutput[0] if len(self.predictionOutput) > 0 else [0, 0, 0, 0],
                "predictionsClass": self.predictionClass
            }
            
            # Add to dataframe
            self.pred_df = pd.concat([self.pred_df, pd.DataFrame([newData])], ignore_index=True)
            print(f"[Prediction] Weather class: {self.predictionClass}")
            
        except Exception as e:
            print(f"[Error] Prediction failed: {e}")
            # Continue without adding prediction data     

    # Drives:
    # Current Frame - Sensor Graph [Real Time, 1 Hour, 1 Day] - Weather Prediction - Console
    def mainDriver(self): 
        self.currentUpdate()
        

    def show(self):  
        self.window.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    A = MainGUI()
    A.show()
