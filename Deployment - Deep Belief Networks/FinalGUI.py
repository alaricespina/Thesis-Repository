import sys
import os 
import pandas as pd 
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QSizePolicy, QTextEdit, QSpacerItem, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui  # Import QtGui
import numpy as np
from random import randint 
from datetime import datetime

# Condition - Not Yet Predicting 
# Local Records
# Site Records
# Dapat Hourly Records 

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
        
        plot.setYRange(ymin, ymax)
        plot.getAxis('left').setPen('k')
        plot.showAxis('left', True)
        plot.getAxis('bottom').setTicks([[]])
        plot.getViewBox().setBackgroundColor('w')

    @staticmethod
    def rollArrayAndAppend(arr, val):
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
        self.initializeMainWindow()
        self.readHistoricalData()
        self.initializeCurrentFrame()
        # self.initializeLocalFrame()
        # self.initializeSiteFrame()
        self.initializeFrameNavigation()
        self.generateInitialData()
        self.adjustGridWidths()
        # self.bindTimer()
        # self.show()
    
    def readHistoricalData(self):
        all_data = pd.DataFrame()
        for file in os.listdir("Data"):
            df = pd.read_csv(os.path.join("Data", file))
            all_data = pd.concat([all_data, df], axis=0)

        print(all_data.describe())
        print(all_data.columns)

        self.historical_temperature_data = all_data['Temperature'].copy()
        self.historical_humidity_data = all_data['Humidity'].copy()
        self.historical_pressure_data = all_data['Pressure'].copy()
        self.historical_wind_data = all_data['Wind Speed'].copy()



    def adjustGridWidths(self):
        num_columns = self.mainLayout.columnCount()
        for column in range(num_columns):
            self.mainLayout.setColumnStretch(column, 1)

    # Main Window - Main Layout - [Dynamic Frame] - [Navigation Frame]
    def initializeMainWindow(self) -> None:
        self.window = QMainWindow()
        self.window.setWindowTitle("DeepBelief Networks - Weather Prediction")
        self.window.setGeometry(0, 0, 800, 400)
        self.mainWidget = QWidget(self.window)
        self.mainWidget.setStyleSheet("background-color: white;")
        self.mainWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.window.setCentralWidget(self.mainWidget)

        self.mainLayout = QGridLayout(self.mainWidget)

    # Navigation Frame
    def initializeFrameNavigation(self):
        ############################################
        # Frame Nav
        ############################################
        fbs = 50
        frameControlsWidget = QWidget(self.mainWidget)
        fCWLayout = QHBoxLayout(frameControlsWidget)
        frameControlsWidget.setLayout(fCWLayout)

        localButton = QPushButton("Local")
        siteButton = QPushButton("Site")
        currentButton = QPushButton("Current")
        # localButton.clicked.connect(local_button_clicked)
        # siteButton.clicked.connect(site_button_clicked)
        # currentButton.clicked.connect(current_button_clicked)   
        localButton.setFixedWidth(fbs)
        siteButton.setFixedWidth(fbs)
        currentButton.setFixedWidth(fbs)
        fCWLayout.addWidget(currentButton)
        fCWLayout.addWidget(localButton)
        fCWLayout.addWidget(siteButton)

        self.mainLayout.addWidget(frameControlsWidget, 3, 0, 1, 3)

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

        cloudyLabel = QLabel()
        rainyLabel = QLabel()
        sunnyLabel = QLabel()
        rainySunnyLabel = QLabel()
        cloudyLabel.setAlignment(Qt.AlignCenter)
        rainyLabel.setAlignment(Qt.AlignCenter)
        sunnyLabel.setAlignment(Qt.AlignCenter)
        rainySunnyLabel.setAlignment(Qt.AlignCenter)

        s = 50
        cloudyPic = cloudyPic.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        rainyPic = rainyPic.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        sunnyPic = sunnyPic.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        rainySunnyPic = rainySunnyPic.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        cloudyLabel.setPixmap(cloudyPic)
        rainyLabel.setPixmap(rainyPic)
        sunnyLabel.setPixmap(sunnyPic)
        rainySunnyLabel.setPixmap(rainySunnyPic)

        weatherGroupLayout.addWidget(cloudyLabel, 0, 0, 1, 1)
        weatherGroupLayout.addWidget(rainyLabel, 0, 1, 1, 1)
        weatherGroupLayout.addWidget(sunnyLabel, 1, 0, 1, 1)
        weatherGroupLayout.addWidget(rainySunnyLabel, 1, 1, 1, 1)

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
        ############################################
        # Console
        ############################################
        cw = QWidget(self.mainWidget)
        self.mainLayout.addWidget(cw, 2, 2, 1, 1)
        cwLayout = QVBoxLayout(cw)

        self.console = QTextEdit(cw)
        self.console.setReadOnly(True)  # Make it read-only
        self.console.setStyleSheet("background-color: black; color: white;") 
        self.console.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff) 
        self.console.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        cwLayout.addWidget(self.console)

        bw = QWidget(cw)
        cwLayout.addWidget(bw)
        bwLayout = QHBoxLayout(bw)

        bs = 50
        realtimeButton = QPushButton("RT")
        hourlyButton = QPushButton("H1")
        dailyButton = QPushButton("D1")
        realtimeButton.setFixedWidth(bs)
        hourlyButton.setFixedWidth(bs)
        dailyButton.setFixedWidth(bs)
        bwLayout.addWidget(realtimeButton)
        bwLayout.addWidget(hourlyButton)
        bwLayout.addWidget(dailyButton)
        bw.setLayout(bwLayout)

    # Calls :
    # Weather Prediction Frame
    # Sensor Frame
    # Console Frame
    def initializeCurrentFrame(self):
        ############################################
        # Label
        ############################################
        spanning_label = QLabel("Implementation of a Deep Belief Network with Sensor Correction Algorithm to predict Weather on a Raspberry Pi")
        spanning_label.setAlignment(Qt.AlignCenter)
        self.mainLayout.addWidget(spanning_label, 0, 0, 1, 3)

        self.initializeCurrentWeatherPredictionFrame()
        self.initializeCurrentSensorFrame()
        self.initializeCurrentConsoleFrame()

    def initializeLocalFrame(self):
        pass 

    def initializeSiteFrame(self):
        pass
    
    # Helper
    def generateFakeData(size, minVal, maxVal):
        return 
    
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
            self.temperatureData = np.zeroes(100)
            self.humidityData = np.zeroes(100)
            self.pressureData = np.zeroes(100)
            self.windData = np.zeroes(100)

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

    # Drives:
    # Current Frame - Sensor Graph [Real Time, 1 Hour, 1 Day] - Weather Prediction - Console
    # Local Frame - 
    def mainDriver(self):    

        # Measure time difference
        now = datetime.now()
        time_diff = now - self.last_update_time
        elapsed_ms = time_diff.total_seconds() * 1000  # Convert to milliseconds

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

        # Update last_update_time
        self.last_update_time = now

        # Create the message
        message = f"Temperature: {current_temp_data:.2f}°C\nHumidity: {current_humid_data:.2f}%\nPressure:{current_pressure_data:.2f}mBar\nWind:{current_wind_data:.2f}kph"
        print(f"[{elapsed_ms:.2f} ms] - {message.replace('\n', ' ')}")
        # Append the message to the console
        self.console.setText(message)



    def show(self):
        self.window.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    current_directory = os.path.basename(os.getcwd())
    if current_directory != "Deployment - Deep Belief Networks":
        print("Current Directory is not Deployment - Deep Belief Networks")
        print("Changing from:", current_directory, "to: Deployment - Deep Belief Networks")
        try:
            os.chdir("Deployment - Deep Belief Networks")
            print("Working Directory is now:" , os.getcwd())
        except Exception as E:
            print("Encountered Error while changing Directory")
            print("Error: ", E)
            exit()

    A = MainGUI()
    timer = QtCore.QTimer()
    timer.timeout.connect(A.mainDriver)
    timer.start(250)
    A.show()
