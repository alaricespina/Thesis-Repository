
import customtkinter as ctk 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import numpy as np 
import pandas as pd 
import os 
from datetime import datetime 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ImageHandler import WeatherImageIcons, IndicatorIcons

board_connected = False 
temp_humid_sensor = None 
testing_change = True

try:
    from HALL import HALL_EFFECT
    from BMP180 import BMP180 
    from DHT11 import DHT11 
    HALL = HALL_EFFECT()
    BMP = BMP180()
    DHT = DHT11()
    print("Succesfully imported Necessary Packages in RPI")

except Exception as E:
    print("Error: ", E)


class MainGUI():
    def __init__(self, w = 800, h = 480, title = "DBN Implementation on Weather Prediction using RPI"):
        self.WIDTH = w
        self.HEIGHT = h
        self.TITLE = title 

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

        self.last_check = datetime.now()
        self.total_delay = 0
        self.delay_count = 0
        

    def initializeGUI(self):
        self.app = ctk.CTk()
        print(f"WIDTH: {self.WIDTH}, HEIGHT: {self.HEIGHT}")
        self.app.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.app.title(self.TITLE)

        self.arial_font = ctk.CTkFont(family="Arial", size=12, weight="normal")
        self.arial_small_font = ctk.CTkFont(family="Arial", size=8, weight="normal")
        self.arial_bold_font = ctk.CTkFont(family="Arial", size=12, weight="bold")
        self.arial_title_font = ctk.CTkFont(family="Arial", size=15, weight="bold")

        self.initializeCurrentFrame(self.app)

    def clearScreen(self):
        try:
            self.deintializeCurrentFrames()
        except:
            pass
   
        print("Cleared")

    def showCurrent(self):
        self.clearScreen()
        self.initializeCurrentFrame(self.app)
        self.setupAnimationAndExecute()


    # ========================================================================================================================
    # FRAME SECTION
    # ========================================================================================================================

    # Current (Graph Readings - with sensor corerction frame) Frame
    def initializeCurrentFrame(self, app):
        # Title Label
        self.title_label = ctk.CTkLabel(master=app, text="Implementation of a Deep Belief Network with Sensor \nCorrection Algorithm to predict Weather on a Raspberry Pi", font=self.arial_title_font)
        self.title_label.place(relx=0, rely=0, relwidth=1.0, relheight=0.1)

        # Main Frames
        self.sensor_frame = ctk.CTkFrame(master=app)
        self.prediction_frame = ctk.CTkFrame(master=app, fg_color="transparent")
        self.prediction_frame.place(relx=0.7, rely=0.15, relheight = 0.7, relwidth = 0.25)

        
        # Console Frame - For Generic Console Logs
        generic_console_frame = ctk.CTkFrame(master=self.prediction_frame)
        self.generic_textbox = ctk.CTkTextbox(master=generic_console_frame, fg_color="black", corner_radius=0)
        generic_console_frame.place(relx=0, rely=0.6, relwidth=1.0, relheight=0.4)
        self.generic_textbox.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
        
        # Sensor Frame
        self.sensor_frame.place(relx=0.05, rely=0.15, relheight = 0.7, relwidth=0.6)
        self.sensor_fig, self.sensor_axs = plt.subplots(2, 2)
        plt.tight_layout()
        sensor_canvas = FigureCanvasTkAgg(self.sensor_fig, self.sensor_frame)
        sensor_canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)

        self.sensor_axs[0, 0].set_title("Temperature")
        self.sensor_axs[0, 1].set_title("Humidity")
        self.sensor_axs[1, 0].set_title("Pressure")
        self.sensor_axs[1, 1].set_title("Wind Speed")

    def insertGenericConsole(self, text, location = "end"):
        print(f"Generic Console: {text}")
        self.generic_textbox.insert(location, "\n" + text)
        self.generic_textbox.see(location)

    def insertWeatherConsole(self, text, location = "end"):
        print(f"Weather Console: {text}")
        self.weather_textbox.delete("0.0", "end")
        self.weather_textbox.insert(location, text)
        self.weather_textbox.see(location)

    def clearConditions(self):
        self.cloudy_indicator.configure(image = self.WII.CLOUDY_INACTIVE)
        self.rainy_indicator.configure(image = self.WII.RAINY_INACTIVE)
        self.sunny_indicator.configure(image = self.WII.SUNNY_INACTIVE)
        self.rainy_and_sunny_indicator.configure(image = self.WII.RAINY_AND_SUNNY_INACTIVE)

    def ToggleCloudy(self):
        if self.DEMO_MODE:
            self.clearConditions()
            self.cloudy_indicator.configure(image = self.WII.CLOUDY_ACTIVE)
            self.insertWeatherConsole("DEMO - CLOUDY WEATHER CONDITION")

    def ToggleSunny(self):
        if self.DEMO_MODE:
            self.clearConditions()
            self.sunny_indicator.configure(image = self.WII.SUNNY_ACTIVE)
            self.insertWeatherConsole("DEMO - SUNNY WEATHER CONDITION")

    def ToggleRainy(self):
        if self.DEMO_MODE:
            self.clearConditions()
            self.rainy_indicator.configure(image = self.WII.RAINY_ACTIVE)
            self.insertWeatherConsole("DEMO - RAINY WEATHER CONDITION") 

    def ToggleRainySunny(self):
        if self.DEMO_MODE:
            self.clearConditions()
            self.rainy_and_sunny_indicator.configure(image = self.WII.RAINY_AND_SUNNY_ACTIVE)
            self.insertWeatherConsole("DEMO - RAINY & SUNNY WEATHER CONDITION") 

    # Remove Current Frame
    def deintializeCurrentFrames(self):
        self.sensor_frame.place_forget()
        self.prediction_frame.place_forget()
        self.title_label.place_forget()
    
    # Frame Controls on bottom of screen
    def initializeFrameControls(self, app):
        self.button_frame = ctk.CTkFrame(master=app, fg_color="transparent")
        self.button_frame.place(relx=0.05, rely=0.9, relwidth=0.9, relheight=0.1)

        self.current_button = ctk.CTkButton(master=self.button_frame, text="Current", command=self.showCurrent, font = self.arial_bold_font)
        self.current_button.place(relx=0, rely=0, relwidth=0.32, relheight=1)

        self.local_button = ctk.CTkButton(master=self.button_frame, text="Local", command=self.showLocal, font = self.arial_bold_font)
        self.local_button.place(relx=0.34, rely=0, relwidth=0.32, relheight=1)

        self.site_button = ctk.CTkButton(master=self.button_frame, text="Site", command=self.showSite, font = self.arial_bold_font)
        self.site_button.place(relx=0.68, rely=0, relwidth=0.32, relheight=1)

    # Site (PAGASA Site) Frame
    def initializeSiteFrame(self, app):
        labels = ["datetime", "tempmax", "tempmin", "temp", "humidity", "windspeed", "sealevelpressure", "conditions"]
        print(self.concatenated_data[labels])

        def populate(frame):
            num_cols = len(labels)
            frame.grid_columnconfigure(tuple(x for x in range(num_cols)), weight=1, uniform="x")

            for a in range(num_cols):
                ctk.CTkLabel(master = frame, text = labels[a]).grid(row=0, column=a)

            for i, row in self.concatenated_data.iterrows():
                if (i < len(self.concatenated_data) - self.site_data_viewing):
                    continue 

                for j in range(num_cols):
                    ctk.CTkLabel(master = frame, text = str(row[labels[j]])).grid(row=i+1, column=j)
        
        def onFrameConfigure(canvas):
            '''Reset the scroll region to encompass the inner frame'''
            canvas.configure(scrollregion=canvas.bbox("all"))

        def setFrameWidth(event):
            self.canvas.itemconfig(self.canvas_frame, width = event.width)
            # self.canvas.itemconfig(self.canvas_frame, height = event.height)
        
        self.data_frame = ctk.CTkFrame(master=app, bg_color="black")
        self.data_frame.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.8)

        self.canvas = ctk.CTkCanvas(master=self.data_frame, border=0)
        self.frame = ctk.CTkFrame(master=self.canvas)
        self.vsb = ctk.CTkScrollbar(master=self.data_frame, orientation="vertical", command=self.canvas.yview)
        self.hsb = ctk.CTkScrollbar(master=self.data_frame, orientation="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.configure(xscrollcommand=self.hsb.set)

        self.vsb.pack(side="right", fill="y")
        self.hsb.pack(side="bottom", fill="x")
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.canvas_frame = self.canvas.create_window((0 , 0), window=self.frame, anchor="nw")

        self.frame.bind("<Configure>", lambda event, canvas = self.canvas: onFrameConfigure(canvas))
        self.canvas.bind('<Configure>', setFrameWidth)
        populate(self.frame)

    # Remove Site Frame
    def deinitializeSiteFrame(self):
        self.data_frame.place_forget()
        self.canvas.desroy()
        self.vsb.pack_forget()

        # self.canvas.place_forget()
        
    # Local Frame (Sensor Readings Frame)
    def initializeLocalFrame(self, app):

        def populate(frame):
            labels = ["Date" , "Time", "Temperature", "Corrected", "Humidity", "Corrected", "Pressure", "Corrected", "Wind Speed", "Corrected"]
            num_cols = len(labels)
            current_snapshot = [self.date, self.time, self.temp_data, self.corrected_temp_data, self.humid_data, self.corrected_humid_data, self.pressure_data, self.corrected_pressure_data, self.wind_data, self.corrected_wind_data]

            frame.grid_columnconfigure(tuple(x for x in range(num_cols)), weight=1, uniform="x")

            for a in range(num_cols):
                ctk.CTkLabel(master = frame, text = labels[a], font = self.arial_bold_font).grid(row=0, column=a)

            for i in range(0, len(current_snapshot[0])):
                if i < len(current_snapshot[0]) - self.local_data_viewing:
                    continue 
                
                for j in range(num_cols):
                    try:
                        if( j > 1) :
                            ctk.CTkLabel(master = frame, text = f"{current_snapshot[j][i]:.2f}").grid(row=i+1, column=j)
                        else:
                            ctk.CTkLabel(master = frame, text = f"{current_snapshot[j][i]}").grid(row=i+1, column=j)
                    except Exception as E:
                        print("Error printing on ", i, j, E)
                        ctk.CTkLabel(master = frame, text = " ").grid(row=i+1, column=j)
        
        
        def onFrameConfigure(canvas):
            '''Reset the scroll region to encompass the inner frame'''
            canvas.configure(scrollregion=canvas.bbox("all"))

        def setFrameWidth(event):
            self.canvas.itemconfig(self.canvas_frame, width = event.width)
            # self.canvas.itemconfig(self.canvas_frame, height = event.height)

        self.current_preview_frame = ctk.CTkFrame(master=app, bg_color="blue")
        self.current_preview_frame.place(relx=0.05, rely=0.05, relwidth=0.9, relheight=0.1)
        
        
        self.temp_preview_frame = ctk.CTkFrame(master=self.current_preview_frame)
        self.temp_preview_frame.place(relx=0, rely=0, relwidth=0.25, relheight=1)
        _ = ctk.CTkLabel(master=self.temp_preview_frame, text="TEMPERATURE (C)", anchor="center").place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self.temp_min_preview = ctk.CTkLabel(master=self.temp_preview_frame, text=f"MIN: {min(self.corrected_temp_data):.2f}")
        self.temp_min_preview.place(relx=0, rely=0.5, relwidth=0.33, relheight=0.5)
        self.temp_cur_preview = ctk.CTkLabel(master=self.temp_preview_frame, text=f"CUR: {self.corrected_temp_data[-1]:.2f}")
        self.temp_cur_preview.place(relx=0.33, rely=0.5, relwidth=0.33, relheight=0.5)
        self.temp_max_preview = ctk.CTkLabel(master=self.temp_preview_frame, text=f"MAX: {min(self.corrected_temp_data):.2f}")
        self.temp_max_preview.place(relx=0.66, rely=0.5, relwidth=0.33, relheight=0.5)
        
        self.humid_preview_frame = ctk.CTkFrame(master=self.current_preview_frame)
        self.humid_preview_frame.place(relx=0.25, rely=0, relwidth=0.25, relheight=1)
        _ = ctk.CTkLabel(master=self.humid_preview_frame, text="HUMIDITY (%)", anchor="center").place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self.humid_min_preview = ctk.CTkLabel(master=self.humid_preview_frame, text=f"MIN: {min(self.corrected_humid_data):.2f}")
        self.humid_min_preview.place(relx=0, rely=0.5, relwidth=0.33, relheight=0.5)
        self.humid_cur_preview = ctk.CTkLabel(master=self.humid_preview_frame, text=f"CUR: {self.corrected_humid_data[-1]:.2f}")
        self.humid_cur_preview.place(relx=0.33, rely=0.5, relwidth=0.33, relheight=0.5)
        self.humid_max_preview = ctk.CTkLabel(master=self.humid_preview_frame, text=f"MAX: {max(self.corrected_humid_data):.2f}")
        self.humid_max_preview.place(relx=0.66, rely=0.5, relwidth=0.33, relheight=0.5)

        self.pressure_preview_frame = ctk.CTkFrame(master=self.current_preview_frame)
        self.pressure_preview_frame.place(relx=0.5, rely=0, relwidth=0.25, relheight=1)
        _ = ctk.CTkLabel(master=self.pressure_preview_frame, text="PRESSURE (mb)", anchor="center").place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self.pressure_min_preview = ctk.CTkLabel(master=self.pressure_preview_frame, text=f"MIN: {min(self.corrected_pressure_data)}")
        self.pressure_min_preview.place(relx=0, rely=0.5, relwidth=0.33, relheight=0.5)
        self.pressure_cur_preview = ctk.CTkLabel(master=self.pressure_preview_frame, text=f"CUR: {self.corrected_pressure_data[-1]}")
        self.pressure_cur_preview.place(relx=0.33, rely=0.5, relwidth=0.33, relheight=0.5)
        self.pressure_max_preview = ctk.CTkLabel(master=self.pressure_preview_frame, text=f"MAX: {max(self.corrected_pressure_data)}")
        self.pressure_max_preview.place(relx=0.66, rely=0.5, relwidth=0.33, relheight=0.5)

        self.wind_preview_frame = ctk.CTkFrame(master=self.current_preview_frame)
        self.wind_preview_frame.place(relx=0.75, rely=0, relwidth=0.25, relheight=1)
        _ = ctk.CTkLabel(master=self.wind_preview_frame, text="WIND SPEED (kph)", anchor="center").place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self.wind_min_preview = ctk.CTkLabel(master=self.wind_preview_frame, text=f"MIN: {min(self.corrected_wind_data):.2f}")
        self.wind_min_preview.place(relx=0, rely=0.5, relwidth=0.33, relheight=0.5)
        self.wind_cur_preview = ctk.CTkLabel(master=self.wind_preview_frame, text=f"CUR: {self.corrected_wind_data[-1]:.2f}")
        self.wind_cur_preview.place(relx=0.33, rely=0.5, relwidth=0.33, relheight=0.5)
        self.wind_max_preview = ctk.CTkLabel(master=self.wind_preview_frame, text=f"MAX: {max(self.corrected_wind_data):.2f}")
        self.wind_max_preview.place(relx=0.66, rely=0.5, relwidth=0.33, relheight=0.5)

        self.data_frame = ctk.CTkFrame(master=app, bg_color="black")
        self.data_frame.place(relx=0.05, rely=0.2, relwidth=0.9, relheight=0.65)


        self.canvas = ctk.CTkCanvas(master=self.data_frame, border=0)
        self.frame = ctk.CTkFrame(master=self.canvas)
        self.vsb = ctk.CTkScrollbar(master=self.data_frame, orientation="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        # self.canvas.pack(fill="both")
        self.canvas_frame = self.canvas.create_window((0 , 0), window=self.frame, anchor="nw")

        self.frame.bind("<Configure>", lambda event, canvas = self.canvas: onFrameConfigure(canvas))
        self.canvas.bind('<Configure>', setFrameWidth)

        

        populate(self.frame)

    # Remove Local Frame
    def deinitializeLocalFrame(self):
        self.current_preview_frame.place_forget()
        self.data_frame.place_forget()
        self.canvas.destroy()
        # self.canvas.place_forget()
        self.vsb.pack_forget()

    def get_stat_data(self, data_arr):
        return (max(data_arr), min(data_arr), data_arr[-1])

    def animate_and_set_data(self, max_widget, min_widget, cur_widget, data_axis, data_arr):
        max_reading, min_reading, cur_reading = self.get_stat_data(data_arr)
        max_widget.configure(text = f"MAX: {max_reading}")
        min_widget.configure(text = f"MIN: {min_reading}")
        cur_widget.configure(text = f"NOW: {cur_reading}")
        
    
    def update(self):
        global BMP, HALL, DHT 
        temp_new_val = DHT.readTemperature() 
        humid_new_val = DHT.readHumidity()
        wind_new_val = HALL.readSpeed()
        pressure_new_val = BMP.readPressure() * -1 / 1000
        
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

        self.changeGraphData(self.temp_data, self.corrected_temp_data, self.sensor_axs[0, 0], 0, 100)
        self.changeGraphData(self.humid_data, self.corrected_humid_data, self.sensor_axs[0, 1], 0, 100)
        self.changeGraphData(self.pressure_data, self.corrected_pressure_data, self.sensor_axs[1, 0], 0, 1500)
        self.changeGraphData(self.wind_data, self.corrected_wind_data, self.sensor_axs[1, 1], 0, 15)

        self.app.update()

        self.app.after(10, self.update)
    
    def changeGraphData(self, show_data, show_corrected, plot_axis, lower_limit = 0, higher_limit = 150, data_length = 20):
        plot_data = show_data[-1 * data_length:]
        plot_corrected = show_corrected[-1 * data_length:]
        title = plot_axis.get_title()
        plot_axis.clear()
        plot_axis.set_ylim(lower_limit, higher_limit)
        plot_axis.plot(plot_data, label="Sensor")
        plot_axis.plot(plot_corrected, label="Corrected")
        plot_axis.legend(loc="upper left")
        plot_axis.set_title(title)

    def setupAnimationAndExecute(self):
        # a = animation.FuncAnimation(self.sensor_fig, self.animate_group, interval=10, cache_frame_data=False)
        self.execute()
    
    def execute(self):
        self.update()

if __name__ == "__main__":

    ThesisMG = MainGUI()
    ThesisMG.initializeGUI()
    ThesisMG.execute()