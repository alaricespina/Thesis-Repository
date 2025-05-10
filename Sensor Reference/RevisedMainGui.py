import customtkinter as ctk 
import tkinter 

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 

import numpy as np 

import adafruit_dht
import board 

temp_humid_sensor = adafruit_dht.DHT11(board.D4)


import random 




WIDTH = 1280
HEIGHT = 720 

app = ctk.CTk()
app.geometry(f"{WIDTH}x{HEIGHT}")
app.title("DBN Implementation on Weather Prediction using RPI")

arial_font = ctk.CTkFont(family="Arial", size=12, weight="normal")
arial_small_font = ctk.CTkFont(family="Arial", size=8, weight="normal")
arial_bold_font = ctk.CTkFont(family="Arial", size=12, weight="bold")
arial_title_font = ctk.CTkFont(family="Arial", size=15, weight="bold")

temp_y = []
humid_y = []
wind_y = []
pressure_y = []


_ = ctk.CTkLabel(master=app, text="Implementation of a Deep Belief Network with Sensor \nCorrection Algorithm to predict Weather on a Raspberry Pi", font=arial_title_font)
_.place(relx=0, rely=0, relwidth=1.0, relheight=0.1)


_ = ctk.CTkLabel(master=app, text="TEMPERATURE", font=arial_font, anchor=tkinter.CENTER)
_.place(relx=0.0, rely=0.10, relwidth=0.4, relheight=0.05)

_ = ctk.CTkLabel(master=app, text="HUMIDITY", font=arial_font, anchor=tkinter.CENTER)
_.place(relx=0.4, rely=0.10, relwidth=0.4, relheight=0.05)

_ = ctk.CTkLabel(master=app, text="WIND SPEED", font=arial_font, anchor=tkinter.CENTER)
_.place(relx=0, rely=0.55, relwidth=0.4, relheight=0.05)

_ = ctk.CTkLabel(master=app, text="PRESSURE", font=arial_font, anchor=tkinter.CENTER)
_.place(relx=0.4, rely=0.55, relwidth=0.4, relheight=0.05)

_ = ctk.CTkLabel(master=app, text="PREDICTION", font=arial_font, anchor=tkinter.CENTER)
_.place(relx=0.8, rely=0.10, relwidth=0.2, relheight=0.05)

# Frames

temp_frame = ctk.CTkFrame(master=app, fg_color="red")
temp_frame.place(relx=0, rely=0.15, relwidth=0.4, relheight=0.3)

humidity_frame = ctk.CTkFrame(master=app, fg_color="green")
humidity_frame.place(relx=0.4, rely=0.15, relwidth=0.4, relheight=0.3)

windspeed_frame = ctk.CTkFrame(master=app, fg_color="blue")
windspeed_frame.place(relx=0, rely=0.6, relwidth=0.4, relheight=0.3)

pressure_frame = ctk.CTkFrame(master=app, fg_color="purple")
pressure_frame.place(relx=0.4, rely=0.6, relwidth=0.4, relheight=0.3)

# Figures
start_i = 0 

temp_fig = plt.Figure()
humidity_fig = plt.Figure()
windspeed_fig = plt.Figure()
pressure_fig = plt.Figure()

plt.ylim(0, 100)

temp_canvas = FigureCanvasTkAgg(temp_fig, temp_frame)
temp_canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)

humidity_canvas = FigureCanvasTkAgg(humidity_fig, humidity_frame)
humidity_canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)

windspeed_canvas = FigureCanvasTkAgg(windspeed_fig, windspeed_frame)
windspeed_canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)

pressure_canvas = FigureCanvasTkAgg(pressure_fig, pressure_frame)
pressure_canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)

temp_ax = temp_fig.add_subplot(1, 1, 1)
humid_ax = humidity_fig.add_subplot(1, 1, 1)
windspeed_ax = windspeed_fig.add_subplot(1, 1, 1)
pressure_ax = pressure_fig.add_subplot(1, 1, 1)
windspeed_ax.set_ylim(-100, 100)


# Function
def temp_animate(i, ys, ax):
    
    ys.append(temp_humid_sensor.temperature)
    #ys.append(random.randint(1, 100))

    s_y = ys[-20:]


    ax.clear()
    ax.plot(s_y)
    max_temp.configure(text=f"MAX: {max(ys)}")
    min_temp.configure(text=f"MIN: {min(ys)}")
    current_temp.configure(text=f"NOW: {ys[-1]}")
    

def humidity_animate(i, ys, ax):
    
    ys.append(temp_humid_sensor.humidity)
    #ys.append(random.randint(1, 100))

    s_y = ys[-20:]

    ax.clear()
    ax.plot(s_y)
    max_humid.configure(text=f"MAX: {max(ys)}")
    min_humid.configure(text=f"MIN: {min(ys)}")
    current_humid.configure(text=f"NOW: {ys[-1]}")


def windspeed_animate(i, ys, ax):
    
    ys.append(0)

    s_y = ys[-20:]

    ax.clear()
    ax.plot(s_y)
    max_windspeed.configure(text=f"MAX: {max(ys)}")
    min_windspeed.configure(text=f"MIN: {min(ys)}")
    current_windspeed.configure(text=f"NOW: {ys[-1]}")

def pressure_animate(i, ys, ax):
    
    ys.append(0)

    s_y = ys[-20:]

    ax.clear()
    ax.plot(s_y)
    max_pressure.configure(text=f"MAX: {max(ys)}")
    min_pressure.configure(text=f"MIN: {min(ys)}")
    current_pressure.configure(text=f"NOW: {ys[-1]}")



ani1 = animation.FuncAnimation(temp_fig, temp_animate, fargs=(temp_y, temp_ax), interval=1000)
ani2 = animation.FuncAnimation(humidity_fig, humidity_animate, fargs=(humid_y, humid_ax), interval=1000)
ani3 = animation.FuncAnimation(windspeed_fig, windspeed_animate, fargs=(wind_y, windspeed_ax), interval=1000)
ani4 = animation.FuncAnimation(pressure_fig, pressure_animate, fargs=(pressure_y, pressure_ax), interval=1000)




# Animate Call



# Frame Stats

temp_stats_frame = ctk.CTkFrame(master=app)
temp_stats_frame.place(relx=0, rely=0.45, relwidth=0.4, relheight=0.10)

humidity_stats_frame = ctk.CTkFrame(master=app)
humidity_stats_frame.place(relx=0.4, rely=0.45, relwidth=0.4, relheight=0.10)

windspeed_stats_frame = ctk.CTkFrame(master=app)
windspeed_stats_frame.place(relx=0, rely=0.9, relwidth=0.4, relheight=0.10)

pressure_stats_frame = ctk.CTkFrame(master=app)
pressure_stats_frame.place(relx=0.4, rely=0.9, relwidth=0.4, relheight=0.10)

# Temperature Stats Display

max_temp = ctk.CTkLabel(master=temp_stats_frame, text="MAX:", font=arial_font)
max_temp.place(relx=0, rely=0, relwidth=0.3, relheight=1)

min_temp = ctk.CTkLabel(master=temp_stats_frame, text="MIN:", font=arial_font)
min_temp.place(relx=0.3, rely=0, relwidth=0.3, relheight=1)

current_temp = ctk.CTkLabel(master=temp_stats_frame, text="NOW:", font=arial_font)
current_temp.place(relx=0.6, rely=0, relwidth=0.4, relheight=1)

# Humidity Stats display

max_humid = ctk.CTkLabel(master=humidity_stats_frame, text="MAX:", font=arial_font)
max_humid.place(relx=0, rely=0, relwidth=0.3, relheight=1)

min_humid = ctk.CTkLabel(master=humidity_stats_frame, text="MIN:", font=arial_font)
min_humid.place(relx=0.3, rely=0, relwidth=0.3, relheight=1)

current_humid = ctk.CTkLabel(master=humidity_stats_frame, text="NOW:", font=arial_font)
current_humid.place(relx=0.6, rely=0, relwidth=0.4, relheight=1)

# Wind speed stats display

max_windspeed = ctk.CTkLabel(master=windspeed_stats_frame, text="MAX:", font=arial_font)
max_windspeed.place(relx=0, rely=0, relwidth=0.3, relheight=1)

min_windspeed = ctk.CTkLabel(master=windspeed_stats_frame, text="MIN:", font=arial_font)
min_windspeed.place(relx=0.3, rely=0, relwidth=0.3, relheight=1)

current_windspeed = ctk.CTkLabel(master=windspeed_stats_frame, text="NOW:", font=arial_font)
current_windspeed.place(relx=0.6, rely=0, relwidth=0.4, relheight=1)

# Pressure stats display

max_pressure = ctk.CTkLabel(master=pressure_stats_frame, text="MAX:", font=arial_font)
max_pressure.place(relx=0, rely=0, relwidth=0.3, relheight=1)

min_pressure = ctk.CTkLabel(master=pressure_stats_frame, text="MIN:", font=arial_font)
min_pressure.place(relx=0.3, rely=0, relwidth=0.3, relheight=1)

current_pressure = ctk.CTkLabel(master=pressure_stats_frame, text="NOW:", font=arial_font)
current_pressure.place(relx=0.6, rely=0, relwidth=0.4, relheight=1)


# _ = ctk.CTkLabel(master=app, text="MIN", font=arial_font)
# _.place(relx=0.35, rely=0.2, relwidth=0.15, relheight=0.1)

# _ = ctk.CTkLabel(master=app, text="MAX", font=arial_font)
# _.place(relx=0.5, rely=0.2, relwidth=0.15, relheight=0.1)

# _ = ctk.CTkLabel(master=app, text="TOMORROW WEATHER", font=arial_bold_font)
# _.place(relx=0.65, rely=0.2, relwidth=0.35, relheight=0.1)

# _ = ctk.CTkLabel(master=app, text="TEMPERATURE", font=arial_bold_font)
# _.place(relx=0, rely=0.3, relwidth=0.2, relheight=0.15)

# _ = ctk.CTkLabel(master=app, text="HUMIDITY", font=arial_bold_font)
# _.place(relx=0, rely=0.45, relwidth=0.2, relheight=0.15)

# _ = ctk.CTkLabel(master=app, text="WIND SPEED", font=arial_bold_font)
# _.place(relx=0, rely=0.60, relwidth=0.2, relheight=0.15)

# cur_temp_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
# cur_temp_ent.place(relx=0.2, rely=0.3, relheight=0.15, relwidth=0.15)

# max_temp_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
# max_temp_ent.place(relx=0.35, rely=0.3, relheight=0.15, relwidth=0.15)

# min_temp_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
# min_temp_ent.place(relx=0.5, rely=0.3, relheight=0.15, relwidth=0.15)

# cur_humid_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
# cur_humid_ent.place(relx=0.2, rely=0.45, relheight=0.15, relwidth=0.15)

# max_humid_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
# max_humid_ent.place(relx=0.35, rely=0.45, relheight=0.15, relwidth=0.15)

# min_humid_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
# min_humid_ent.place(relx=0.5, rely=0.45, relheight=0.15, relwidth=0.15)

# cur_wind_speed_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
# cur_wind_speed_ent.place(relx=0.2, rely=0.6, relheight=0.15, relwidth=0.15)

# max_wind_speed_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
# max_wind_speed_ent.place(relx=0.35, rely=0.6, relheight=0.15, relwidth=0.15)

# min_wind_speed_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
# min_wind_speed_ent.place(relx=0.5, rely=0.6, relheight=0.15, relwidth=0.15)

# weather_frame = ctk.CTkFrame(master=app)
# weather_frame.place(relx=0.65, rely=0.3, relwidth=0.35,  relheight = 0.6)

# weather_prediction_ent = ctk.CTkEntry(master=app, font=arial_font)
# weather_prediction_ent.place(relx=0.65, rely=0.9, relwidth=0.35, relheight=0.1)
# weather_prediction_ent.insert(0, "ERROR: NO SENSOR CONNECTED")
# weather_prediction_ent.configure(state="disable")

# command_output_textbox = ctk.CTkTextbox(master=app, corner_radius=0, font=arial_small_font)
# command_output_textbox.place(relx=0.35, rely=0.75, relwidth=0.3, relheight=0.25)
# command_output_textbox.insert("0.0", "DHT11 not connected\nBMP180 not connected\nA3144 not connected")

# def reload_sensor_button_clicked():
#     command_output_textbox.insert('end', "\n===REFRESH CLICKED===\nDHT11 not connected\nBMP180 not connected\nA3144 not connected")
#     command_output_textbox.see('end')

# reload_img = ctk.CTkImage(dark_image=Image.open("icon - refresh.png"), size=(10, 10))

# reload_sensor_button = ctk.CTkButton(master=app, corner_radius=0, text="REFRESH", command=reload_sensor_button_clicked, font=arial_bold_font)
# reload_sensor_button.place(relx=0.2, rely=0.75, relwidth=0.15, relheight=0.25)

app.mainloop()


