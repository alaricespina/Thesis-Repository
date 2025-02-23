import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime
import time

root = tk.Tk()
root.title("Dynamic Matplotlib Chart in Tkinter")

# 1. Create a Matplotlib Figure and Axes
fig, ax = plt.subplots()
line, = ax.plot([], [])  # Create an empty line object for updating

# 2. Embed the Matplotlib Figure in Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# 3. Define the Update Function
x_data = list(range(-50, 0))  # Initialize with a range of values
y_data = [0] * 50  # Initialize with zeros
i = 0
last_check = datetime.now()
update_interval = 0.01  # seconds (10 milliseconds)

def update_plot():
    global i, x_data, y_data, last_check

    # Generate new data (replace with your actual data source)
    x_data = x_data[1:] + [i]  # Efficient shifting using slicing
    y_data = y_data[1:] + [np.sin(i / 10)]

    # Update the plot data
    line.set_xdata(x_data)
    line.set_ydata(y_data)

    # Adjust the axes limits (important for dynamic plots)
    ax.set_xlim(i - 50, i)  # Keep x-axis showing the last 50 values
    ax.relim()  # Recalculate limits
    ax.autoscale_view()  # Autoscale based on new data

    # Redraw the canvas
    canvas.draw()

    i += 1

    now = datetime.now()
    diff = now - last_check
    dc = round((diff.microseconds + diff.seconds * 1000000) / 1000, 0)
    last_check = now
    print(f"{dc}ms - X: {x_data[-1]} Y: {y_data[-1]}, L: {len(x_data)}")

# 4. Custom Event Loop (Replacing mainloop())
running = True
while running:
    try:
        update_plot()  # Call the update function
        root.update()  # Process events and update the GUI
        time.sleep(0.25)  # Small delay to avoid excessive CPU usage
    except tk.TclError:
        # Window has been destroyed
        running = False
        break

print("Tkinter window closed.")