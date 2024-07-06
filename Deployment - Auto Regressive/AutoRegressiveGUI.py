import customtkinter as ctk 
import tkinter 

from PIL import Image

WIDTH = 1280
HEIGHT = 720 

app = ctk.CTk()
app.geometry(f"{WIDTH}x{HEIGHT}")
app.title("AutoRegressive Modeling for Long-Term Climate Change Projections: Predicting Temperature, Humidity, and Precipitation")

arial_font = ctk.CTkFont(family="Arial", size=12, weight="normal")
arial_small_font = ctk.CTkFont(family="Arial", size=8, weight="normal")
arial_bold_font = ctk.CTkFont(family="Arial", size=12, weight="bold")
arial_title_font = ctk.CTkFont(family="Arial", size=15, weight="bold")

state = "Temperature"

######################################### TEMPERATURE #########################################

def displayTemperature():
    # Temp Labels
    tempLabel = ctk.CTkLabel(master = app, text = "Temperature")
    tempLabel.place(relx = 0.7, rely = 0.05, relwidth = 0.3, relheight = 0.25)

    tempMinLabel = ctk.CTkLabel(master = app, text = "Min Temp")
    tempMinLabel.place(relx = 0.7, rely = 0.31, relwidth = 0.3, relheight = 0.25)

    tempMaxLabel = ctk.CTkLabel(master = app, text = "Max Temp")
    tempMaxLabel.place(relx = 0.7, rely = 0.57, relwidth = 0.3, relheight = 0.25)

    # Temperature Pics
    tempPic = ctk.CTkImage(Image.open("Predictions/Temperature.png"), size = (800, 175))
    tempBtn = ctk.CTkButton(master = app, text = "", image = tempPic, bg_color="transparent", fg_color = "transparent")
    tempBtn.place(relx = 0, rely = 0.05, relwidth = 0.7, relheight = 0.25)

    tempMinPic = ctk.CTkImage(Image.open("Predictions/TempMin.png"), size = (800, 175))
    tempMinBtn = ctk.CTkButton(master = app, text = "", image = tempMinPic, bg_color="transparent", fg_color = "transparent")
    tempMinBtn.place(relx = 0, rely = 0.31, relwidth = 0.7, relheight = 0.25)

    tempMaxPic = ctk.CTkImage(Image.open("Predictions/TempMax.png"), size = (800, 175))
    tempMaxBtn = ctk.CTkButton(master = app, text = "", image = tempMaxPic, bg_color="transparent", fg_color = "transparent")
    tempMaxBtn.place(relx = 0, rely = 0.57, relwidth = 0.7, relheight = 0.25)

######################################### FEELSLIKE #########################################

def displayFeelsLike():
    # FeelsLike
    FeelsLikeLabel = ctk.CTkLabel(master = app, text = "FeelsLike Temp")
    FeelsLikeLabel.place(relx = 0.7, rely = 0.05, relwidth = 0.3, relheight = 0.25)

    FeelsLikeMinLabel = ctk.CTkLabel(master = app, text = "FeelsLike Min Temp")
    FeelsLikeMinLabel.place(relx = 0.7, rely = 0.31, relwidth = 0.3, relheight = 0.25)

    FeelsLikeMaxLabel = ctk.CTkLabel(master = app, text = "FeelsLike Max Temp")
    FeelsLikeMaxLabel.place(relx = 0.7, rely = 0.57, relwidth = 0.3, relheight = 0.25)

    # FeelsLike Pics
    FeelsLikePic = ctk.CTkImage(Image.open("Predictions/FeelsLike.png"), size = (800, 175))
    FeelsLikeBtn = ctk.CTkButton(master = app, text = "", image = FeelsLikePic, bg_color="transparent", fg_color = "transparent")
    FeelsLikeBtn.place(relx = 0, rely = 0.05, relwidth = 0.7, relheight = 0.25)

    FeelsLikeMinPic = ctk.CTkImage(Image.open("Predictions/FeelsLikeMin.png"), size = (800, 175))
    FeelsLikeMinBtn = ctk.CTkButton(master = app, text = "", image = FeelsLikeMinPic, bg_color="transparent", fg_color = "transparent")
    FeelsLikeMinBtn.place(relx = 0, rely = 0.31, relwidth = 0.7, relheight = 0.25)

    FeelsLikeMaxPic = ctk.CTkImage(Image.open("Predictions/FeelsLikeMax.png"), size = (800, 175))
    FeelsLikeMaxBtn = ctk.CTkButton(master = app, text = "", image = FeelsLikeMaxPic, bg_color="transparent", fg_color = "transparent")
    FeelsLikeMaxBtn.place(relx = 0, rely = 0.57, relwidth = 0.7, relheight = 0.25)

######################################### HUMIDITY #########################################

def displayHumidity():
    HumidityLabel = ctk.CTkLabel(master = app, text = "Humidity")
    HumidityLabel.place(relx = 0, rely = 0.05, relwidth = 1.0, relheight = 0.25)

    HumidityPic = ctk.CTkImage(Image.open("Predictions/Humidity.png"), size = (1200, 375))
    HumidityBtn = ctk.CTkButton(master = app, text = "", image = HumidityPic, bg_color="transparent", fg_color = "transparent")
    HumidityBtn.place(relx = 0, rely = 0.26, relwidth = 1.0, relheight = 0.55)

######################################### PRECIPITATION #########################################

def displayPrecipitation():
    PrecipitationLabel = ctk.CTkLabel(master = app, text = "Precipitation")
    PrecipitationLabel.place(relx = 0, rely = 0.05, relwidth = 1.0, relheight = 0.25)

    PrecipitationPic = ctk.CTkImage(Image.open("Predictions/Precipitations.png"), size = (1200, 375))
    PrecipitationBtn = ctk.CTkButton(master = app, text = "", image = PrecipitationPic, bg_color="transparent", fg_color = "transparent")
    PrecipitationBtn.place(relx = 0, rely = 0.26, relwidth = 1.0, relheight = 0.55)


def destroyChildren():
    for children in app.winfo_children():
        children.place_forget()

def updateState(target_state):
    global state
    state = target_state
    checkState()
    pass 

def displayButtons():
    temperatureBtn = ctk.CTkButton(master = app, text = "Temperature", command = lambda : updateState("Temperature"))
    temperatureBtn.place(relx = 0.01, rely = 0.9, relwidth = 0.23, relheight = 0.09)

    feelslikeBtn = ctk.CTkButton(master = app, text = "Feels Like", command = lambda : updateState("Feels Like"))
    feelslikeBtn.place(relx = 0.26, rely = 0.9, relwidth = 0.23, relheight = 0.09)

    humidityBtn = ctk.CTkButton(master = app, text = "Humidity", command = lambda : updateState("Humidity"))
    humidityBtn.place(relx = 0.51, rely = 0.9, relwidth = 0.23, relheight = 0.09)

    precipitationBtn = ctk.CTkButton(master = app, text = "Precipitation", command = lambda : updateState("Precipitation"))
    precipitationBtn.place(relx = 0.76, rely = 0.9, relwidth = 0.23, relheight = 0.09)

def checkState():
    global state 

    destroyChildren()

    if (state == "Temperature"):
        displayTemperature()
    elif (state == "Feels Like"):
        displayFeelsLike()
    elif (state == "Humidity"):
        displayHumidity()
    elif (state == "Precipitation"):
        displayPrecipitation()
    
    displayButtons()


checkState()



app.mainloop()


