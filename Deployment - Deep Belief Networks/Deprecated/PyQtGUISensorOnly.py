import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QSizePolicy, QTextEdit, QSpacerItem, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui  # Import QtGui
import numpy as np
from datetime import datetime

app = QApplication(sys.argv)


# Condition - Not Yet Predicting 
# Local Records
# Site Records
# Dapat Hourly Records 


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
last_update_time = datetime.now()

def setPlotLimits(plot, ymin, ymax):
    plot.setYRange(ymin, ymax)
    plot.getAxis('left').setPen('k')
    plot.showAxis('left', True)
    plot.getAxis('bottom').setTicks([[]])
    plot.getViewBox().setBackgroundColor('w')

def roll_arr_and_append(arr, val):
    arr = np.roll(arr, -1)
    arr[-1] = val
    return arr

def rt_button_clicked():
    print("RT Button Clicked")

def h1_button_clicked():
    print("H1 Button Clicked")

def d1_button_clicked():
    print("D1 Button Clicked")

def local_button_clicked():
    print("Local Button is Clicked")

def site_button_clicked():
    print("Site Button is Clicked")

def current_button_clicked():
    print("Current Button Clicked")

############################################
# Main Window
############################################
window = QMainWindow()
window.setWindowTitle("DeepBelief Networks - Weather Prediction")
window.setGeometry(0, 0, 800, 400)
mainWidget = QWidget(window)
mainWidget.setStyleSheet("background-color: white;")
mainWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
window.setCentralWidget(mainWidget)

mainLayout = QGridLayout(mainWidget)

############################################
# Label
############################################
spanning_label = QLabel("Implementation of a Deep Belief Network with Sensor Correction Algorithm to predict Weather on a Raspberry Pi")
spanning_label.setAlignment(Qt.AlignCenter)
mainLayout.addWidget(spanning_label, 0, 0, 1, 3)

############################################
# Weather Icons
############################################
weatherGroupWidget = QWidget(mainWidget)
mainLayout.addWidget(weatherGroupWidget, 1, 2, 1, 1)

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

############################################
# Sensor Graphs
############################################
graphicsView = pg.GraphicsLayoutWidget()
graphicsView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
mainLayout.addWidget(graphicsView, 1, 0, 2, 2)

temp_plot = graphicsView.addPlot(row=0, col=0, title="Temperature (°C)")
humid_plot = graphicsView.addPlot(row=0, col=1, title="Humidity (%)")
graphicsView.nextRow()
pressure_plot = graphicsView.addPlot(row=1, col=0, title="Pressure (mBar)")
wind_plot = graphicsView.addPlot(row=1, col=1, title="Wind Speed (kph)")

for plot in [temp_plot, humid_plot, pressure_plot, wind_plot]:
    plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

############################################
# Console
############################################
cw = QWidget(mainWidget)
mainLayout.addWidget(cw, 2, 2, 1, 1)
cwLayout = QVBoxLayout(cw)

console = QTextEdit(cw)
console.setReadOnly(True)  # Make it read-only
console.setStyleSheet("background-color: black; color: white;") 
console.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff) 
console.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
cwLayout.addWidget(console)

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



############################################
# Frame Nav
############################################
fbs = 50
frameControlsWidget = QWidget(mainWidget)
fCWLayout = QHBoxLayout(frameControlsWidget)
frameControlsWidget.setLayout(fCWLayout)

localButton = QPushButton("Local")
siteButton = QPushButton("Site")
currentButton = QPushButton("Current")
localButton.clicked.connect(local_button_clicked)
siteButton.clicked.connect(site_button_clicked)
currentButton.clicked.connect(current_button_clicked)   
localButton.setFixedWidth(fbs)
siteButton.setFixedWidth(fbs)
currentButton.setFixedWidth(fbs)
fCWLayout.addWidget(currentButton)
fCWLayout.addWidget(localButton)
fCWLayout.addWidget(siteButton)

mainLayout.addWidget(frameControlsWidget, 3, 0, 1, 3)





############################################
# Seed Data
############################################
x1 = np.linspace(0, 10, 100)
x2 = np.linspace(0, 10, 100)
x3 = np.linspace(0, 10, 100)
x4 = np.linspace(0, 10, 100)

temp_data = np.sin(x1) if not in_board else np.zeros(100)
humid_data = np.cos(x2) if not in_board else np.zeros(100)
pressure_data = np.tan(x3) if not in_board else np.zeros(100)
wind_data = np.sin(2*x4) if not in_board else np.zeros(100)

curve1 = temp_plot.plot(x1, temp_data, pen='b', name='Temperature')
curve2 = humid_plot.plot(x2, humid_data, pen='b', name='Humidity')
curve3 = pressure_plot.plot(x3, pressure_data, pen='b', name='Pressure')
curve4 = wind_plot.plot(x4, wind_data, pen='b', name='Wind Speed')

setPlotLimits(temp_plot, temp_data.min() * 0.9, temp_data.max() * 1.1)
setPlotLimits(humid_plot, humid_data.min() * 0.9, humid_data.max() * 1.1)
setPlotLimits(pressure_plot, pressure_data.min() * 0.9, pressure_data.max() * 1.1)
setPlotLimits(wind_plot, wind_data.min() * 0.9, wind_data.max() * 1.1)


############################################
# Main Driver Function
############################################
def update_plots():
    global x1, temp_data, x2, humid_data, x3, pressure_data, x4, wind_data, last_update_time

    # Measure time difference
    now = datetime.now()
    time_diff = now - last_update_time
    elapsed_ms = time_diff.total_seconds() * 1000  # Convert to milliseconds

    # Update data for each plot (replace with your actual data sources)
    x1 = x1 + 1
    if in_board:
        temp_data = roll_arr_and_append(temp_data, DHT.readTemperature())
    else:
        temp_data = np.sin(x1)
    curve1.setData(x1, temp_data)
    current_temp_data = temp_data[-1]  # Get the last y-value

    x2 = x2 + 1  # Different update rate for each plot
    if in_board:
        humid_data = roll_arr_and_append(humid_data, DHT.readHumidity())
    else:
        humid_data = np.cos(x2)
    curve2.setData(x2, humid_data)
    current_humid_data = humid_data[-1]

    x3 = x3 + 1
    if in_board:
        pressure_data = roll_arr_and_append(pressure_data, BMP.readPressure() * -1 / 1000 * 10)
    else:
        pressure_data = np.tan(x3)
    curve3.setData(x3, pressure_data)
    current_pressure_data = pressure_data[-1]

    x4 = x4 + 1
    if in_board:
        wind_data = roll_arr_and_append(wind_data, HALL.readSpeed())
    else:
        wind_data = np.sin(2*x4)
    curve4.setData(x4, wind_data)
    current_wind_data = wind_data[-1]
    
    try:
        if current_temp_data != 0:
            temp_plot.setYRange(temp_data.min() * 0.9, temp_data.max() * 1.1)
        if current_humid_data != 0:
            humid_plot.setYRange(humid_data.min() * 0.9, humid_data.max() * 1.1)
        if current_pressure_data != 0:
            pressure_plot.setYRange(pressure_data.min() * 0.9, pressure_data.max() * 1.1)
        if current_wind_data != 0:
            wind_plot.setYRange(wind_data.min() * 0.9, wind_data.max() * 1.1)
        
    except Exception as E:
        print("Failed Setting Limits", E)

    # Update last_update_time
    last_update_time = now

    # Create the message
    message = f"Temperature: {current_temp_data:.2f}°C\nHumidity: {current_humid_data:.2f}%\nPressure:{current_pressure_data:.2f}mBar\nWind:{current_wind_data:.2f}kph"
    # print(f"[{elapsed_ms:.2f} ms]\n{message}")
    # Append the message to the console
    console.setText(message)

num_columns = mainLayout.columnCount()
num_rows = mainLayout.rowCount()

# Set the same stretch factor for all columns
for column in range(num_columns):
    mainLayout.setColumnStretch(column, 1)  # Give each column a stretch factor of 1

# for row in range(num_rows):
#     mainLayout.setRowStretch(row, 1)

############################################
# Timer
############################################
timer = QtCore.QTimer()
timer.timeout.connect(update_plots)
timer.start(250)  # Update every 20 milliseconds

############################################
# Main Loop
############################################
window.show()
sys.exit(app.exec_())