import customtkinter as ctk 
import tkinter 

from PIL import Image

WIDTH = 1280
HEIGHT = 720 

app = ctk.CTk()
app.geometry(f"{WIDTH}x{HEIGHT}")
app.title("DBN Implementation on Weather Prediction using RPI")

arial_font = ctk.CTkFont(family="Arial", size=12, weight="normal")
arial_small_font = ctk.CTkFont(family="Arial", size=8, weight="normal")
arial_bold_font = ctk.CTkFont(family="Arial", size=12, weight="bold")
arial_title_font = ctk.CTkFont(family="Arial", size=15, weight="bold")



_ = ctk.CTkLabel(master=app, text="Implementation of a Deep Belief Network with Sensor \nCorrection Algorithm to predict Weather on a Raspberry Pi", font=arial_title_font)
_.place(relx=0, rely=0, relwidth=1.0, relheight=0.2)

_ = ctk.CTkLabel(master=app, text="CURRENT", font=arial_font)
_.place(relx=0.2, rely=0.2, relwidth=0.15, relheight=0.1)

_ = ctk.CTkLabel(master=app, text="MIN", font=arial_font)
_.place(relx=0.35, rely=0.2, relwidth=0.15, relheight=0.1)

_ = ctk.CTkLabel(master=app, text="MAX", font=arial_font)
_.place(relx=0.5, rely=0.2, relwidth=0.15, relheight=0.1)

_ = ctk.CTkLabel(master=app, text="TOMORROW WEATHER", font=arial_bold_font)
_.place(relx=0.65, rely=0.2, relwidth=0.35, relheight=0.1)

_ = ctk.CTkLabel(master=app, text="TEMPERATURE", font=arial_bold_font)
_.place(relx=0, rely=0.3, relwidth=0.2, relheight=0.15)

_ = ctk.CTkLabel(master=app, text="HUMIDITY", font=arial_bold_font)
_.place(relx=0, rely=0.45, relwidth=0.2, relheight=0.15)

_ = ctk.CTkLabel(master=app, text="WIND SPEED", font=arial_bold_font)
_.place(relx=0, rely=0.60, relwidth=0.2, relheight=0.15)

cur_temp_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
cur_temp_ent.place(relx=0.2, rely=0.3, relheight=0.15, relwidth=0.15)

max_temp_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
max_temp_ent.place(relx=0.35, rely=0.3, relheight=0.15, relwidth=0.15)

min_temp_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
min_temp_ent.place(relx=0.5, rely=0.3, relheight=0.15, relwidth=0.15)

cur_humid_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
cur_humid_ent.place(relx=0.2, rely=0.45, relheight=0.15, relwidth=0.15)

max_humid_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
max_humid_ent.place(relx=0.35, rely=0.45, relheight=0.15, relwidth=0.15)

min_humid_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
min_humid_ent.place(relx=0.5, rely=0.45, relheight=0.15, relwidth=0.15)

cur_wind_speed_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
cur_wind_speed_ent.place(relx=0.2, rely=0.6, relheight=0.15, relwidth=0.15)

max_wind_speed_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
max_wind_speed_ent.place(relx=0.35, rely=0.6, relheight=0.15, relwidth=0.15)

min_wind_speed_ent = ctk.CTkEntry(master=app, state=tkinter.DISABLED, corner_radius=0, font=arial_bold_font)
min_wind_speed_ent.place(relx=0.5, rely=0.6, relheight=0.15, relwidth=0.15)

weather_frame = ctk.CTkFrame(master=app)
weather_frame.place(relx=0.65, rely=0.3, relwidth=0.35,  relheight = 0.6)

weather_prediction_ent = ctk.CTkEntry(master=app, font=arial_font)
weather_prediction_ent.place(relx=0.65, rely=0.9, relwidth=0.35, relheight=0.1)
weather_prediction_ent.insert(0, "ERROR: NO SENSOR CONNECTED")
weather_prediction_ent.configure(state="disable")

command_output_textbox = ctk.CTkTextbox(master=app, corner_radius=0, font=arial_small_font)
command_output_textbox.place(relx=0.35, rely=0.75, relwidth=0.3, relheight=0.25)
command_output_textbox.insert("0.0", "DHT11 not connected\nBMP180 not connected\nA3144 not connected")

def reload_sensor_button_clicked():
    command_output_textbox.insert('end', "\n===REFRESH CLICKED===\nDHT11 not connected\nBMP180 not connected\nA3144 not connected")
    command_output_textbox.see('end')

reload_img = ctk.CTkImage(dark_image=Image.open("icon - refresh.png"), size=(10, 10))

reload_sensor_button = ctk.CTkButton(master=app, corner_radius=0, text="REFRESH", command=reload_sensor_button_clicked, font=arial_bold_font)
reload_sensor_button.place(relx=0.2, rely=0.75, relwidth=0.15, relheight=0.25)

app.mainloop()


