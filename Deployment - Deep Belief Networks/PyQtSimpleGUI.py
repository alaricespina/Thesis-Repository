import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time
from random import randint

board_connected = False
temp_humid_sensor = None
testing_change = True

try:
    from HALL import HALL_EFFECT
    from BMP180 import BMP180
    from DHT11 import DHT11
    # import RPi.GPIO as GPIO
    # GPIO.setmode(GPIO.BOARD)
    HALL = HALL_EFFECT()
    BMP = BMP180()
    DHT = DHT11()

    print("Successfully imported Necessary Packages in RPI")

except Exception as E:
    print("Error: ", E)

'''
Green : 00ff24
Yellow : ffc600
Red : ff0000
'''

class MainGUI(QtWidgets.QMainWindow):
    def __init__(self, w = 800, h = 480, title = "DBN Implementation on Weather Prediction using RPI"):
        super().__init__()
        self.WIDTH = w
        self.HEIGHT = h
        self.TITLE = title

        self.BMP_CONNECTED = False
        self.HALL_CONNECTED = False
        self.DHT_CONNECTED = False
        self.DEMO_MODE = True

        self.temp_data = [0]
        self.humid_data = [0]
        self.wind_data = [0]
        self.pressure_data = [0]

        self.corrected_temp_data = [0]
        self.corrected_humid_data = [0]
        self.corrected_wind_data = [0]
        self.corrected_pressure_data = [0]

        self.prediction_data = []
        self.date = []
        self.time = []

        self.local_data_viewing = 25
        self.site_data_viewing = 25

        # self.WII = WeatherImageIcons() # Removed ImageHandler dependency
        # self.II = IndicatorIcons() # Removed ImageHandler dependency
        self.loadHistoricalData()
        self.last_check = datetime.now()
        self.total_delay = 0
        self.delay_count = 0

        self.initializeGUI()

    def loadHistoricalData(self):
        list_data = []
        for file_name in os.listdir("Data"):
            print("Opening:",file_name)
            current_data = pd.read_csv("Data/" + file_name)
            list_data.append(current_data)

        self.concatenated_data = pd.concat(list_data, ignore_index=True, sort=False)

    def initializeGUI(self):
        self.setWindowTitle(self.TITLE)
        self.setGeometry(100, 100, self.WIDTH, self.HEIGHT)

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.initializeCurrentFrame()
        self.initializeFrameControls()

    def clearScreen(self):
        try:
            self.deintializeCurrentFrames()
        except:
            pass

        try:
            self.deinitializeLocalFrame()
        except:
            pass

        try:
            self.deinitializeSiteFrame()
        except:
            pass

        print("Cleared")

    def showCurrent(self):
        self.clearScreen()
        self.initializeCurrentFrame()
        self.setupAnimationAndExecute()

    def showLocal(self):
        self.clearScreen()
        self.initializeLocalFrame()

    def showSite(self):
        self.clearScreen()
        self.initializeSiteFrame()

    # ========================================================================================================================
    # FRAME SECTION
    # ========================================================================================================================

    # Current (Graph Readings - with sensor corerction frame) Frame
    def initializeCurrentFrame(self):
        # Title Label
        self.title_label = QtWidgets.QLabel("Implementation of a Deep Belief Network with Sensor \nCorrection Algorithm to predict Weather on a Raspberry Pi")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)

        # Main Frames
        self.sensor_frame = QtWidgets.QFrame()
        self.prediction_frame = QtWidgets.QFrame()
        self.prediction_frame.setStyleSheet("background-color: transparent;")
        self.prediction_layout = QtWidgets.QVBoxLayout(self.prediction_frame)

        # Weather Prediction Frame
        weather_prediction_frame = QtWidgets.QFrame()
        weather_prediction_frame.setStyleSheet("background-color: red;")
        self.prediction_layout.addWidget(weather_prediction_frame)

        weather_prediction_layout = QtWidgets.QGridLayout(weather_prediction_frame)

        self.cloudy_indicator = QtWidgets.QPushButton()
        self.rainy_indicator = QtWidgets.QPushButton()
        self.sunny_indicator = QtWidgets.QPushButton()
        self.rainy_and_sunny_indicator = QtWidgets.QPushButton()

        self.cloudy_indicator.setStyleSheet("background-color: black;")
        self.rainy_indicator.setStyleSheet("background-color: black;")
        self.sunny_indicator.setStyleSheet("background-color: black;")
        self.rainy_and_sunny_indicator.setStyleSheet("background-color: black;")

        self.cloudy_indicator.clicked.connect(self.ToggleCloudy)
        self.rainy_indicator.clicked.connect(self.ToggleRainy)
        self.sunny_indicator.clicked.connect(self.ToggleSunny)
        self.rainy_and_sunny_indicator.clicked.connect(self.ToggleRainySunny)

        weather_prediction_layout.addWidget(self.cloudy_indicator, 0, 0)
        weather_prediction_layout.addWidget(self.rainy_indicator, 0, 1)
        weather_prediction_layout.addWidget(self.sunny_indicator, 1, 0)
        weather_prediction_layout.addWidget(self.rainy_and_sunny_indicator, 1, 1)

        # w = self.cloudy_indicator.width() # Removed ImageHandler dependency
        # weather_scaling = 0.4 # Removed ImageHandler dependency
        # if (self.WII.w == -1 or self.WII.h == -1): # Removed ImageHandler dependency
        #     self.WII.setDimensions(w * weather_scaling, w * weather_scaling) # Removed ImageHandler dependency
        #     self.WII.makeImages() # Removed ImageHandler dependency

        # self.cloudy_indicator.setIcon(self.WII.CLOUDY_INACTIVE) # Removed ImageHandler dependency
        # self.rainy_indicator.setIcon(self.WII.RAINY_INACTIVE) # Removed ImageHandler dependency
        # self.sunny_indicator.setIcon(self.WII.SUNNY_INACTIVE) # Removed ImageHandler dependency
        # self.rainy_and_sunny_indicator.setIcon(self.WII.RAINY_AND_SUNNY_INACTIVE) # Removed ImageHandler dependency

        # Mini Console Frame for Weather
        weather_console_frame = QtWidgets.QFrame()
        self.weather_textbox = QtWidgets.QTextEdit()
        self.weather_textbox.setStyleSheet("background-color: black;")
        weather_console_layout = QtWidgets.QVBoxLayout(weather_console_frame)
        weather_console_layout.addWidget(self.weather_textbox)
        self.prediction_layout.addWidget(weather_console_frame)

        # Console Frame - For Generic Console Logs
        generic_console_frame = QtWidgets.QFrame()
        self.generic_textbox = QtWidgets.QTextEdit()
        self.generic_textbox.setStyleSheet("background-color: black;")
        generic_console_layout = QtWidgets.QVBoxLayout(generic_console_frame)
        generic_console_layout.addWidget(self.generic_textbox)
        self.prediction_layout.addWidget(generic_console_frame)

        # Sensor Frame
        self.sensor_graphics_view = pg.GraphicsView()
        self.sensor_layout = pg.GraphicsLayout()
        self.sensor_graphics_view.addItem(self.sensor_layout)

        self.temp_plot = self.sensor_layout.addPlot(title="Temperature")
        self.sensor_layout.nextRow()
        self.humid_plot = self.sensor_layout.addPlot(title="Humidity")
        self.sensor_layout.nextRow()
        self.pressure_plot = self.sensor_layout.addPlot(title="Pressure")
        self.sensor_layout.nextRow()
        self.wind_plot = self.sensor_layout.addPlot(title="Wind Speed")

        self.main_layout.addWidget(self.sensor_graphics_view)
        self.main_layout.addWidget(self.prediction_frame)

    def insertGenericConsole(self, text, location = "end"):
        print(f"Generic Console: {text}")
        self.generic_textbox.insertPlainText("\n" + text)
        self.generic_textbox.moveCursor(QtGui.QTextCursor.End)

    def insertWeatherConsole(self, text, location = "end"):
        print(f"Weather Console: {text}")
        self.weather_textbox.clear()
        self.weather_textbox.insertPlainText(text)
        self.weather_textbox.moveCursor(QtGui.QTextCursor.End)

    def clearConditions(self):
        # self.cloudy_indicator.setIcon(self.WII.CLOUDY_INACTIVE) # Removed ImageHandler dependency
        # self.rainy_indicator.setIcon(self.WII.RAINY_INACTIVE) # Removed ImageHandler dependency
        # self.sunny_indicator.setIcon(self.WII.SUNNY_INACTIVE) # Removed ImageHandler dependency
        # self.rainy_and_sunny_indicator.setIcon(self.WII.RAINY_AND_SUNNY_INACTIVE) # Removed ImageHandler dependency
        pass

    def ToggleCloudy(self):
        if self.DEMO_MODE:
            self.clearConditions()
            # self.cloudy_indicator.setIcon(self.WII.CLOUDY_ACTIVE) # Removed ImageHandler dependency
            self.insertWeatherConsole("DEMO - CLOUDY WEATHER CONDITION")

    def ToggleSunny(self):
        if self.DEMO_MODE:
            self.clearConditions()
            # self.sunny_indicator.setIcon(self.WII.SUNNY_ACTIVE) # Removed ImageHandler dependency
            self.insertWeatherConsole("DEMO - SUNNY WEATHER CONDITION")

    def ToggleRainy(self):
        if self.DEMO_MODE:
            self.clearConditions()
            # self.rainy_indicator.setIcon(self.WII.RAINY_ACTIVE) # Removed ImageHandler dependency
            self.insertWeatherConsole("DEMO - RAINY WEATHER CONDITION")

    def ToggleRainySunny(self):
        if self.DEMO_MODE:
            self.clearConditions()
            # self.rainy_and_sunny_indicator.setIcon(self.WII.RAINY_AND_SUNNY_ACTIVE) # Removed ImageHandler dependency
            self.insertWeatherConsole("DEMO - RAINY & SUNNY WEATHER CONDITION")

    # Remove Current Frame
    def deintializeCurrentFrames(self):
        # self.sensor_frame.hide()
        self.prediction_frame.hide()
        self.title_label.hide()

    # Frame Controls on bottom of screen
    def initializeFrameControls(self):
        self.button_frame = QtWidgets.QFrame()
        button_layout = QtWidgets.QHBoxLayout(self.button_frame)

        self.current_button = QtWidgets.QPushButton("Current")
        self.local_button = QtWidgets.QPushButton("Local")
        self.site_button = QtWidgets.QPushButton("Site")

        self.current_button.clicked.connect(self.showCurrent)
        self.local_button.clicked.connect(self.showLocal)
        self.site_button.clicked.connect(self.showSite)

        button_layout.addWidget(self.current_button)
        button_layout.addWidget(self.local_button)
        button_layout.addWidget(self.site_button)

        self.main_layout.addWidget(self.button_frame)

    # Site (PAGASA Site) Frame
    def initializeSiteFrame(self):
        labels = ["datetime", "tempmax", "tempmin", "temp", "humidity", "windspeed", "sealevelpressure", "conditions"]
        print(self.concatenated_data[labels])

        self.data_frame = QtWidgets.QFrame()
        self.data_layout = QtWidgets.QVBoxLayout(self.data_frame)

        table_widget = QtWidgets.QTableWidget()
        table_widget.setRowCount(len(self.concatenated_data))
        table_widget.setColumnCount(len(labels))
        table_widget.setHorizontalHeaderLabels(labels)

        for i, row in self.concatenated_data.iterrows():
            for j, label in enumerate(labels):
                item = QtWidgets.QTableWidgetItem(str(row[label]))
                table_widget.setItem(i, j, item)

        self.data_layout.addWidget(table_widget)
        self.main_layout.addWidget(self.data_frame)

    # Remove Site Frame
    def deinitializeSiteFrame(self):
        self.data_frame.hide()

    # Local Frame (Sensor Readings Frame)
    def initializeLocalFrame(self):

        self.current_preview_frame = QtWidgets.QFrame()
        preview_layout = QtWidgets.QHBoxLayout(self.current_preview_frame)

        # Temperature Preview
        self.temp_preview_frame = QtWidgets.QFrame()
        temp_layout = QtWidgets.QVBoxLayout(self.temp_preview_frame)
        temp_label = QtWidgets.QLabel("TEMPERATURE (C)")
        temp_label.setAlignment(QtCore.Qt.AlignCenter)
        temp_layout.addWidget(temp_label)
        self.temp_min_preview = QtWidgets.QLabel(f"MIN: {min(self.corrected_temp_data):.2f}")
        self.temp_cur_preview = QtWidgets.QLabel(f"CUR: {self.corrected_temp_data[-1]:.2f}")
        self.temp_max_preview = QtWidgets.QLabel(f"MAX: {max(self.corrected_temp_data):.2f}")
        temp_layout.addWidget(self.temp_min_preview)
        temp_layout.addWidget(self.temp_cur_preview)
        temp_layout.addWidget(self.temp_max_preview)
        preview_layout.addWidget(self.temp_preview_frame)

        # Humidity Preview
        self.humid_preview_frame = QtWidgets.QFrame()
        humid_layout = QtWidgets.QVBoxLayout(self.humid_preview_frame)
        humid_label = QtWidgets.QLabel("HUMIDITY (%)")
        humid_label.setAlignment(QtCore.Qt.AlignCenter)
        humid_layout.addWidget(humid_label)
        self.humid_min_preview = QtWidgets.QLabel(f"MIN: {min(self.corrected_humid_data):.2f}")
        self.humid_cur_preview = QtWidgets.QLabel(f"CUR: {self.corrected_humid_data[-1]:.2f}")
        self.humid_max_preview = QtWidgets.QLabel(f"MAX: {max(self.corrected_humid_data):.2f}")
        humid_layout.addWidget(self.humid_min_preview)
        humid_layout.addWidget(self.humid_cur_preview)
        humid_layout.addWidget(self.humid_max_preview)
        preview_layout.addWidget(self.humid_preview_frame)

        # Pressure Preview
        self.pressure_preview_frame = QtWidgets.QFrame()
        pressure_layout = QtWidgets.QVBoxLayout(self.pressure_preview_frame)
        pressure_label = QtWidgets.QLabel("PRESSURE (mb)")
        pressure_label.setAlignment(QtCore.Qt.AlignCenter)
        pressure_layout.addWidget(pressure_label)
        self.pressure_min_preview = QtWidgets.QLabel(f"MIN: {min(self.corrected_pressure_data):.2f}")
        self.pressure_cur_preview = QtWidgets.QLabel(f"CUR: {self.corrected_pressure_data[-1]:.2f}")
        self.pressure_max_preview = QtWidgets.QLabel(f"MAX: {max(self.corrected_pressure_data):.2f}")
        pressure_layout.addWidget(self.pressure_min_preview)
        pressure_layout.addWidget(self.pressure_cur_preview)
        pressure_layout.addWidget(self.pressure_max_preview)
        preview_layout.addWidget(self.pressure_preview_frame)

        # Wind Preview
        self.wind_preview_frame = QtWidgets.QFrame()
        wind_layout = QtWidgets.QVBoxLayout(self.wind_preview_frame)
        wind_label = QtWidgets.QLabel("WIND SPEED (kph)")
        wind_label.setAlignment(QtCore.Qt.AlignCenter)
        wind_layout.addWidget(wind_label)
        self.wind_min_preview = QtWidgets.QLabel(f"MIN: {min(self.corrected_wind_data):.2f}")
        self.wind_cur_preview = QtWidgets.QLabel(f"CUR: {self.corrected_wind_data[-1]:.2f}")
        self.wind_max_preview = QtWidgets.QLabel(f"MAX: {max(self.corrected_wind_data):.2f}")
        wind_layout.addWidget(self.wind_min_preview)
        wind_layout.addWidget(self.wind_cur_preview)
        wind_layout.addWidget(self.wind_max_preview)
        preview_layout.addWidget(self.wind_preview_frame)

        self.main_layout.addWidget(self.current_preview_frame)

        # Data Table
        self.data_frame = QtWidgets.QFrame()
        self.data_layout = QtWidgets.QVBoxLayout(self.data_frame)

        table_widget = QtWidgets.QTableWidget()
        table_widget.setRowCount(len(self.date))
        table_widget.setColumnCount(10)
        labels = ["Date" , "Time", "Temperature", "Corrected", "Humidity", "Corrected", "Pressure", "Corrected", "Wind Speed", "Corrected"]
        table_widget.setHorizontalHeaderLabels(labels)

        current_snapshot = [self.date, self.time, self.temp_data, self.corrected_temp_data, self.humid_data, self.corrected_humid_data, self.pressure_data, self.corrected_pressure_data, self.wind_data, self.corrected_wind_data]

        for i in range(len(self.date)):
            for j in range(10):
                try:
                    if j > 1:
                        item = QtWidgets.QTableWidgetItem(f"{current_snapshot[j][i]:.2f}")
                    else:
                        item = QtWidgets.QTableWidgetItem(current_snapshot[j][i])
                    table_widget.setItem(i, j, item)
                except Exception as E:
                    print("Error printing on ", i, j, E)
                    item = QtWidgets.QTableWidgetItem(" ")
                    table_widget.setItem(i, j, item)

        self.data_layout.addWidget(table_widget)
        self.main_layout.addWidget(self.data_frame)

    # Remove Local Frame
    def deinitializeLocalFrame(self):
        self.current_preview_frame.hide()
        self.data_frame.hide()

    def get_stat_data(self, data_arr):
        return (max(data_arr), min(data_arr), data_arr[-1])

    def animate_and_set_data(self, max_widget, min_widget, cur_widget, data_axis, data_arr):
        max_reading, min_reading, cur_reading = self.get_stat_data(data_arr)
        max_widget.setText(f"MAX: {max_reading}")
        min_widget.setText(f"MIN: {min_reading}")
        cur_widget.setText(f"NOW: {cur_reading}")


    def update(self):
        global BMP, HALL, DHT
        # temp_new_val = DHT.readTemperature()
        # humid_new_val = DHT.readHumidity()
        # wind_new_val = HALL.readSpeed()
        # pressure_new_val = BMP.readPressure() * -1 / 1000
        temp_new_val = randint(1, 100)
        humid_new_val = randint(1, 100)
        wind_new_val = randint(1, 10)
        pressure_new_val = randint(1, 1000)

        now = datetime.now()
        self.date.append(now.strftime("%m/%d/%Y"))
        self.time.append(now.strftime("%H:%M:%S"))
        diff = now - self.last_check
        dc = round((diff.microseconds + diff.seconds * 1000000) / 1000, 0)
        self.total_delay += dc
        self.delay_count += 1
        adc = round(self.total_delay / self.delay_count, 0)

        temp_new_val = temp_new_val if temp_new_val != None else 0
        humid_new_val = humid_new_val if humid_new_val != None else 0

        print(f"{now.strftime('%H:%M:%S.%f')[:-3]} - {dc}ms = {adc}ms - Temp: {temp_new_val}C, Humid: {humid_new_val}%, Wind: {round(wind_new_val, 2)}, Pressure: {round(pressure_new_val, 3)}KPa")

        self.last_check = now

        self.temp_data.append(temp_new_val)
        self.humid_data.append(humid_new_val)
        self.wind_data.append(wind_new_val)
        self.pressure_data.append(pressure_new_val)

        self.corrected_temp_data.append(temp_new_val * 0.8)
        self.corrected_humid_data.append(humid_new_val * 0.9)
        self.corrected_wind_data.append(wind_new_val)
        self.corrected_pressure_data.append(pressure_new_val)

        self.updateGraphData(self.temp_data, self.corrected_temp_data, self.temp_plot, 0, 100)
        self.updateGraphData(self.humid_data, self.corrected_humid_data, self.humid_plot, 0, 100)
        self.updateGraphData(self.pressure_data, self.corrected_pressure_data, self.pressure_plot, 0, 1500)
        self.updateGraphData(self.wind_data, self.corrected_wind_data, self.wind_plot, 0, 15)

        self.animate_and_set_data(self.temp_max_preview, self.temp_min_preview, self.temp_cur_preview, self.temp_plot, self.corrected_temp_data)
        self.animate_and_set_data(self.humid_max_preview, self.humid_min_preview, self.humid_cur_preview, self.humid_plot, self.corrected_humid_data)
        self.animate_and_set_data(self.pressure_max_preview, self.pressure_min_preview, self.pressure_cur_preview, self.pressure_plot, self.corrected_pressure_data)
        self.animate_and_set_data(self.wind_max_preview, self.wind_min_preview, self.wind_cur_preview, self.wind_plot, self.corrected_wind_data)

    def updateGraphData(self, show_data, show_corrected, plot_widget, lower_limit = 0, higher_limit = 150, data_length = 20):
        plot_data = show_data[-1 * data_length:]
        plot_corrected = show_corrected[-1 * data_length:]
        plot_widget.clear()
        plot_widget.setYRange(lower_limit, higher_limit)
        plot_widget.plot(plot_data, pen='r', name="Sensor")
        plot_widget.plot(plot_corrected, pen='b', name="Corrected")
        plot_widget.addLegend()

    def setupAnimationAndExecute(self):
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)  # Update every 100 ms
        self.timer.timeout.connect(self.update)
        self.timer.start()
        self.show()

    def execute(self):
        self.show()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ThesisMG = MainGUI()
    ThesisMG.setupAnimationAndExecute()
    sys.exit(app.exec_())
