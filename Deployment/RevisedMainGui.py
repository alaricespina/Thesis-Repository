
import customtkinter as ctk 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import numpy as np 
import pandas as pd 
import random 
# import adafruit_dht
# import board 


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure 
from PIL import Image

# DHT 11 Temperature and Humidity
DHT_LOADED = False

# BMP 180 Barometric Pressure
BMP_LOADED = False 

# Hall Effect A3144 Magnetic Field Sensor
HES_LOADED = False 

'''
Green : 00ff24
Yellow : ffc600
Red : ff0000
'''

class MainGUI():
    def __init__(self, w = 1280, h = 720, title = "DBN Implementation on Weather Prediction using RPI"):
        self.WIDTH = w
        self.HEIGHT = h
        self.TITLE = title 
        self.temp_data = [0]
        self.corrected_temp_data = [0]
        self.humid_data = [0]
        self.corrected_humid_data = [0]
        self.wind_data = [0]
        self.corrected_wind_data = [0]
        self.pressure_data = [0]
        self.corrected_pressure_data = [0]
        self.prediction_data = []
        self.data_2011 = pd.read_csv("Data/2011.csv")
        self.data_2012 = pd.read_csv("Data/2012.csv")
        self.data_2013 = pd.read_csv("Data/2013.csv")
        self.data_2014 = pd.read_csv("Data/2014.csv")
        self.concatenated_data = pd.concat([self.data_2011, self.data_2012, self.data_2013, self.data_2014], ignore_index=True, sort=False)

    def initializeGUI(self):
        self.app = ctk.CTk()
        print(f"WIDTH: {self.WIDTH}, HEIGHT: {self.HEIGHT}")
        # self.app.geometry("f{self.WIDTH}x{self.HEIGHT}")
        self.app.geometry("1280x720")
        self.app.title(self.TITLE)


        self.arial_font = ctk.CTkFont(family="Arial", size=12, weight="normal")
        self.arial_small_font = ctk.CTkFont(family="Arial", size=8, weight="normal")
        self.arial_bold_font = ctk.CTkFont(family="Arial", size=12, weight="bold")
        self.arial_title_font = ctk.CTkFont(family="Arial", size=15, weight="bold")

        self.initializeCurrentFrame(self.app)
        self.initializeFrameControls(self.app)
        self.setupAnimations()
        

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
        self.initializeCurrentFrame(self.app)
        self.setupAnimations()

    def showLocal(self):
        self.clearScreen()
        self.initializeLocalFrame(self.app)

    def showSite(self):
        self.clearScreen()
        self.initializeSiteFrame(self.app)

    # Frame Controls on bottom of screen
    def initializeFrameControls(self, app):
        self.button_frame = ctk.CTkFrame(master=app, fg_color="#242424")
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
                if (i < len(self.concatenated_data) - 100):
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
        self.canvas.desroy()
        self.data_frame.place_forget()

    # Local Frame (Sensor Readings Frame)
    def initializeLocalFrame(self, app):

        def populate(frame):
            labels = ["Date" , "Time", "Temperature", "Humidity", "Pressure", "Wind Speed"]
            num_cols = len(labels)
            num_rows = 100
            frame.grid_columnconfigure(tuple(x for x in range(num_cols)), weight=1, uniform="x")

            for a in range(num_cols):
                ctk.CTkLabel(master = frame, text = labels[a]).grid(row=0, column=a)

            for i in range(1, num_rows):
                for j in range(num_cols):
                    ctk.CTkLabel(master = frame, text = "69.69").grid(row=i, column=j)
        
        
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
        _ = ctk.CTkLabel(master=self.temp_preview_frame, text="TEMPERATURE", anchor="center").place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self.temp_min_preview = ctk.CTkLabel(master=self.temp_preview_frame, text="MIN: ").place(relx=0, rely=0.5, relwidth=0.33, relheight=0.5)
        self.temp_cur_preview = ctk.CTkLabel(master=self.temp_preview_frame, text="CUR: ").place(relx=0.33, rely=0.5, relwidth=0.33, relheight=0.5)
        self.temp_max_preview = ctk.CTkLabel(master=self.temp_preview_frame, text="MAX: ").place(relx=0.66, rely=0.5, relwidth=0.33, relheight=0.5)
        
        self.humid_preview_frame = ctk.CTkFrame(master=self.current_preview_frame)
        self.humid_preview_frame.place(relx=0.25, rely=0, relwidth=0.25, relheight=1)
        _ = ctk.CTkLabel(master=self.humid_preview_frame, text="HUMIDITY", anchor="center").place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self.humid_min_preview = ctk.CTkLabel(master=self.humid_preview_frame, text="MIN: ").place(relx=0, rely=0.5, relwidth=0.33, relheight=0.5)
        self.humid_cur_preview = ctk.CTkLabel(master=self.humid_preview_frame, text="CUR: ").place(relx=0.33, rely=0.5, relwidth=0.33, relheight=0.5)
        self.humid_max_preview = ctk.CTkLabel(master=self.humid_preview_frame, text="MAX: ").place(relx=0.66, rely=0.5, relwidth=0.33, relheight=0.5)

        self.pressure_preview_frame = ctk.CTkFrame(master=self.current_preview_frame)
        self.pressure_preview_frame.place(relx=0.5, rely=0, relwidth=0.25, relheight=1)
        _ = ctk.CTkLabel(master=self.pressure_preview_frame, text="PRESSURE", anchor="center").place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self.pressure_min_preview = ctk.CTkLabel(master=self.pressure_preview_frame, text="MIN: ").place(relx=0, rely=0.5, relwidth=0.33, relheight=0.5)
        self.pressure_cur_preview = ctk.CTkLabel(master=self.pressure_preview_frame, text="CUR: ").place(relx=0.33, rely=0.5, relwidth=0.33, relheight=0.5)
        self.pressure_max_preview = ctk.CTkLabel(master=self.pressure_preview_frame, text="MAX: ").place(relx=0.66, rely=0.5, relwidth=0.33, relheight=0.5)

        self.wind_preview_frame = ctk.CTkFrame(master=self.current_preview_frame)
        self.wind_preview_frame.place(relx=0.75, rely=0, relwidth=0.25, relheight=1)
        _ = ctk.CTkLabel(master=self.wind_preview_frame, text="WIND SPEED", anchor="center").place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self.wind_min_preview = ctk.CTkLabel(master=self.wind_preview_frame, text="MIN: ").place(relx=0, rely=0.5, relwidth=0.33, relheight=0.5)
        self.wind_cur_preview = ctk.CTkLabel(master=self.wind_preview_frame, text="CUR: ").place(relx=0.33, rely=0.5, relwidth=0.33, relheight=0.5)
        self.wind_max_preview = ctk.CTkLabel(master=self.wind_preview_frame, text="MAX: ").place(relx=0.66, rely=0.5, relwidth=0.33, relheight=0.5)

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

    # Current (Graph Readings - with sensor corerction frame) Frame
    def initializeCurrentFrame(self, app):
        self.title_label = ctk.CTkLabel(master=app, text="Implementation of a Deep Belief Network with Sensor \nCorrection Algorithm to predict Weather on a Raspberry Pi", font=self.arial_title_font)
        self.title_label.place(relx=0, rely=0, relwidth=1.0, relheight=0.1)

        self.sensor_frame = ctk.CTkFrame(master=app)
        self.prediction_frame = ctk.CTkFrame(master=app, fg_color="transparent")
        self.prediction_frame.place(relx=0.7, rely=0.15, relheight = 0.7, relwidth = 0.25)

        self.weather_prediction_frame = ctk.CTkFrame(master=self.prediction_frame, fg_color="red")
        self.weather_prediction_frame.place(relx=0, rely=0, relwidth=1.0, relheight=0.4)
        
        
        self.cloudy_indicator = ctk.CTkButton(master=self.weather_prediction_frame, text="", fg_color="#242424", corner_radius=0)
        self.rainy_indicator = ctk.CTkButton(master=self.weather_prediction_frame, text="", fg_color="#242424", corner_radius=0)
        self.sunny_indicator = ctk.CTkButton(master=self.weather_prediction_frame, text="", fg_color="#242424", corner_radius=0)
        self.rainy_and_sunny_indicator = ctk.CTkButton(master=self.weather_prediction_frame, text="", fg_color="#242424", corner_radius=0)

        self.cloudy_indicator.place(relx=0, rely=0, relwidth=0.5, relheight=0.5)
        self.rainy_indicator.place(relx=0.5, rely=0, relwidth=0.5, relheight=0.5)
        self.sunny_indicator.place(relx=0, rely=0.5, relwidth=0.5, relheight=0.5)
        self.rainy_and_sunny_indicator.place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.5)

        w, h = self.cloudy_indicator.cget("width"), self.cloudy_indicator.cget("height")
        print(w, h)

        cloudy_image = ctk.CTkImage(dark_image=Image.open("icon - cloudy.png"), size=(w * 0.7, w * 0.7))
        rainy_sunny_image = ctk.CTkImage(dark_image=Image.open("icon - rainy and sunny.png"), size=(w * 0.7, w * 0.7))
        rainy_image = ctk.CTkImage(dark_image=Image.open("icon - rainy.png"), size=(w * 0.7, w * 0.7))
        sunny_image = ctk.CTkImage(dark_image=Image.open("icon - sunny.png"), size=(w * 0.7, w * 0.7))

        self.cloudy_indicator.configure(image = cloudy_image)
        self.rainy_indicator.configure(image = rainy_sunny_image)
        self.sunny_indicator.configure(image = rainy_image)
        self.rainy_and_sunny_indicator.configure(image = sunny_image)
        

        self.indicator_frame = ctk.CTkFrame(master=self.prediction_frame, fg_color="black")
        self.indicator_frame.place(relx=0, rely=0.45, relwidth=1.0, relheight=0.15)

        self.demo_real_mode = ctk.CTkButton(master=self.indicator_frame, text="", fg_color="black", corner_radius=0)
        self.anemo_status = ctk.CTkButton(master=self.indicator_frame, text="", fg_color="black",  corner_radius=0)
        self.temp_status = ctk.CTkButton(master=self.indicator_frame, text="", fg_color="black",  corner_radius=0)
        self.humid_status = ctk.CTkButton(master=self.indicator_frame, text="", fg_color="black",   corner_radius=0)
        self.bmp_status = ctk.CTkButton(master=self.indicator_frame, text="", fg_color="black",   corner_radius=0)

        w, self.demo_real_mode.cget("width")

        scaling_factor = 0.3

        demo_image = ctk.CTkImage(dark_image=Image.open("DEMO ICON.png"), size = (w * scaling_factor, w * scaling_factor)) 
        real_image = ctk.CTkImage(dark_image=Image.open("REAL ICON.png"), size = (w * scaling_factor, w * scaling_factor)) 
        anemo_demo = ctk.CTkImage(dark_image=Image.open("ANEMO DEMO.png"), size = (w * scaling_factor, w * scaling_factor)) 
        anemo_discon = ctk.CTkImage(dark_image=Image.open("ANEMO DISCONNECTED.png"), size = (w * scaling_factor, w * scaling_factor)) 
        anemo_working = ctk.CTkImage(dark_image=Image.open("ANEMO WORKING.png"), size = (w * scaling_factor, w * scaling_factor)) 
        baro_demo = ctk.CTkImage(dark_image=Image.open("BAROMETER DEMO.png"), size = (w * scaling_factor, w * scaling_factor)) 
        baro_discon = ctk.CTkImage(dark_image=Image.open("BAROMETER DISCONNECTED.png"), size = (w * scaling_factor, w * scaling_factor)) 
        baro_working = ctk.CTkImage(dark_image=Image.open("BAROMETER WORKING.png"), size = (w * scaling_factor, w * scaling_factor)) 
        humid_demo = ctk.CTkImage(dark_image=Image.open("HUMIDITY DEMO.png"), size = (w * scaling_factor, w * scaling_factor)) 
        humid_discon = ctk.CTkImage(dark_image=Image.open("HUMIDITY DISCONNECTED.png"), size = (w * scaling_factor, w * scaling_factor)) 
        humid_working = ctk.CTkImage(dark_image=Image.open("HUMIDITY WORKING.png"), size = (w * scaling_factor, w * scaling_factor)) 
        temp_demo = ctk.CTkImage(dark_image=Image.open("TEMPERATURE DEMO.png"), size = (w * scaling_factor, w * scaling_factor)) 
        temp_discon = ctk.CTkImage(dark_image=Image.open("TEMPERATURE DISCONNECTED.png"), size = (w * scaling_factor, w * scaling_factor)) 
        temp_working = ctk.CTkImage(dark_image=Image.open("TEMPERATURE WORKING.png"), size = (w * scaling_factor, w * scaling_factor)) 

        self.demo_real_mode.configure(image = demo_image)
        self.anemo_status.configure(image = anemo_demo)
        self.temp_status.configure(image = temp_demo)
        self.humid_status.configure(image = humid_demo)
        self.bmp_status.configure(image = baro_demo)

        self.demo_real_mode.place(relx=0, rely=0, relwidth=0.2, relheight=1)
        self.anemo_status.place(relx=0.2, rely=0, relwidth=0.2, relheight=1)
        self.temp_status.place(relx=0.4, rely=0, relwidth=0.2, relheight=1)
        self.humid_status.place(relx=0.6, rely=0, relwidth=0.2, relheight=1)
        self.bmp_status.place(relx=0.8, rely=0, relwidth=0.2, relheight=1)
        
        self.current_time_frame = ctk.CTkFrame(master=self.prediction_frame, fg_color="#242424")
        self.current_time_frame.place(relx=0, rely=0.70, relwidth=1.0, relheight=0.15)

        self.current_time_label = ctk.CTkLabel(master = self.current_time_frame, text = "Time: ", fg_color="#242424")
        self.current_time_label.place(relx=0, rely=0, relwidth=0.5, relheight = 0.5)
        self.save_time_label = ctk.CTkLabel(master = self.current_time_frame, text="Save @:", fg_color="#242424")
        self.save_time_label.place(relx=0, rely=0.5, relwidth=0.5, relheight=0.5)
        self.force_save_button = ctk.CTkButton(master = self.current_time_frame, text="", fg_color="#242424")
        self.force_save_button.place(relx=0.5, rely=0, relheight=1, relwidth=0.5)

        w = self.force_save_button.cget("width")
        force_save_image = ctk.CTkImage(dark_image=Image.open("diskette.png"), size = (w * 0.5, w * 0.5))
        self.force_save_button.configure(image = force_save_image)

        self.toggle_frame = ctk.CTkFrame(master=self.prediction_frame, fg_color="#242424")
        self.toggle_frame.place(relx=0, rely=0.9, relwidth=1.0, relheight=0.1)
        self.activate_demo = ctk.CTkButton(master = self.toggle_frame, text = "DEMO", corner_radius = 0, fg_color="#c79a00")
        self.activate_demo.place(relx = 0, rely = 0, relwidth = 0.5, relheight = 1)
        self.activate_real = ctk.CTkButton(master = self.toggle_frame, text = "REAL", corner_radius = 0, fg_color="#029917")
        self.activate_real.place(relx = 0.5, rely = 0, relwidth = 0.5, relheight = 1)


        self.temp_frame = ctk.CTkFrame(master=self.sensor_frame)
        self.humidity_frame = ctk.CTkFrame(master=self.sensor_frame)
        self.windspeed_frame = ctk.CTkFrame(master=self.sensor_frame)
        self.pressure_frame = ctk.CTkFrame(master=self.sensor_frame)

        self.sensor_frame.place(relx=0.05, rely=0.15, relheight = 0.7, relwidth=0.6)
        self.temp_frame.place(relx=0, rely=0, relwidth=0.5, relheight=0.5)       
        self.humidity_frame.place(relx=0.5, rely=0, relwidth=0.5, relheight=0.5)
        self.windspeed_frame.place(relx=0, rely=0.5, relwidth=0.5, relheight=0.5)
        self.pressure_frame.place(relx=0.5, rely=0.5, relwidth=0.5, relheight=0.5)

        self.temp_fig = plt.Figure()
        self.humidity_fig = plt.Figure()
        self.windspeed_fig = plt.Figure()
        self.pressure_fig = plt.Figure()

        plt.ylim(0, 100)

        self.temp_canvas = FigureCanvasTkAgg(self.temp_fig, self.temp_frame)
        self.humidity_canvas = FigureCanvasTkAgg(self.humidity_fig, self.humidity_frame)
        self.windspeed_canvas = FigureCanvasTkAgg(self.windspeed_fig, self.windspeed_frame)
        self.pressure_canvas = FigureCanvasTkAgg(self.pressure_fig, self.pressure_frame)

        self.temp_canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)
        self.humidity_canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)
        self.windspeed_canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)
        self.pressure_canvas.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=1)

        self.temp_ax = self.temp_fig.add_subplot(1, 1, 1)
        self.humid_ax = self.humidity_fig.add_subplot(1, 1, 1)
        self.windspeed_ax = self.windspeed_fig.add_subplot(1, 1, 1)
        self.pressure_ax = self.pressure_fig.add_subplot(1, 1, 1)

        self.temp_ax.set_title("Temperature")
        self.humid_ax.set_title("Humidity")
        self.pressure_ax.set_title("Pressure")
        self.windspeed_ax.set_title("Wind Speed")

    # Remove Current Frame
    def deintializeCurrentFrames(self):
        # self.temp_frame.place_forget()
        # self.humidity_frame.place_forget()
        # self.windspeed_frame.place_forget()
        # self.pressure_frame.place_forget()
        self.sensor_frame.place_forget()
        self.prediction_frame.place_forget()
        self.title_label.place_forget()
    

    def get_stat_data(self, data_arr):
        return (max(data_arr), min(data_arr), data[-1])

    def append_sensor_data(self, data_arr):
        return 0

    def animate_and_set_data(self, max_widget, min_widget, cur_widget, data_axis, data_arr):
        self.animate_data(data_arr, data_axis)
        max_reading, min_reading, cur_reading = self.get_stat_data(data_arr)
        max_widget.configure(text = f"MAX: {max_reading}")
        min_widget.configure(text = f"MIN: {min_reading}")
        cur_widget.configure(text = f"NOW: {cur_reading}")
    
    def setupAnimations(self):

        def animate_data(self, data_arr, corrected_data, plot_axis, data_length = 20):

            new_val = random.randint(0, 100)
            data_arr.append(new_val)
            corrected_data.append(new_val * 0.75)
            show_data = data_arr[-1 * data_length:]
            show_corrected = corrected_data[-1 * data_length:]
            title = plot_axis.get_title()
            plot_axis.clear()
            plot_axis.set_ylim(0, 150)
            plot_axis.plot(show_data, label="Sensor")
            plot_axis.plot(show_corrected, label="Corrected")
            plot_axis.legend(loc="upper left")
            display_stats = f"MAX: {max(data_arr)}\nMIN: {min(data_arr)}\nCUR: {data_arr[-1]}"
            ax = plt.gca()
            plot_axis.text(0.7, 0.6, display_stats, horizontalalignment="right", verticalalignment="top", transform=ax.transAxes)
            plot_axis.set_title(title)
            
            # print("Animated ")
            # print(show_data)

        tempAnimation = animation.FuncAnimation(self.temp_fig, animate_data, fargs=(self.temp_data, self.corrected_temp_data, self.temp_ax), interval=1000)
        humidAnimation = animation.FuncAnimation(self.humidity_fig, animate_data, fargs=(self.humid_data, self.corrected_humid_data, self.humid_ax), interval=1000)
        windAnimation = animation.FuncAnimation(self.windspeed_fig, animate_data, fargs=(self.wind_data, self.corrected_wind_data, self.windspeed_ax), interval=1000)
        pressureAnimation = animation.FuncAnimation(self.pressure_fig, animate_data, fargs=(self.pressure_data, self.corrected_pressure_data, self.pressure_ax), interval=1000)
        self.app.mainloop()

if __name__ == "__main__":
    ThesisMG = MainGUI()
    ThesisMG.initializeGUI()

