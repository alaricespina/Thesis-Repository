import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt5 import QtWidgets  # Import QtWidgets
import numpy as np
from datetime import datetime
import time

# 1. Create a PyQtGraph Application and Window
app = QtWidgets.QApplication([])  # Create a QtWidgets.QApplication
win = pg.GraphicsLayoutWidget(show=True, title="Dynamic PyQtGraph Plot")  # Create a window

# 2. Add a Plot to the Window
plot = win.addPlot(title="Sine Wave")  # Add a plot widget
curve = plot.plot()  # Create a curve (line) object

# 3. Initialize Data
x_data = np.linspace(-50, 0, 50)  # Initialize x-data
y_data = np.zeros(50)  # Initialize y-data
i = 0

# 4. Define the Update Function
def update_plot():
    global i, x_data, y_data

    start_time = time.time()

    # Generate new data (replace with your actual data source)
    x_data[:-1] = x_data[1:]  # Efficient shifting using slicing
    x_data[-1] = i
    y_data[:-1] = y_data[1:]
    y_data[-1] = np.sin(i / 10)

    # Update the plot data
    curve.setData(x_data, y_data)  # Update the curve data

    # Adjust the axes limits (optional, but can improve performance)
    plot.setXRange(i - 50, i)  # Keep x-axis showing the last 50 values
    #plot.setYRange(-1.2, 1.2) # Set y-axis range if needed

    i += 1

    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # in milliseconds

    print(f"Update Time: {elapsed_time:.2f} ms")

# 5. Create a Timer to Update the Plot
timer = QtCore.QTimer()  # Create a Qt timer
timer.timeout.connect(update_plot)  # Connect the timer to the update function
timer.start(10)  # Start the timer with a 10ms interval

# 6. Start the PyQtGraph Application Event Loop
if __name__ == '__main__':
    app.exec_()  # Start the Qt event loop