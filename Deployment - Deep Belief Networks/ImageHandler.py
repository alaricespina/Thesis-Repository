from customtkinter import CTkImage 
from PIL import Image

class OtherIcons:
    def __init__(self):
        pass 


class IndicatorIcons:
    def __init__(self, w = -1, h = -1):
        self.anemometer_image_demo_path = "Icons/ANEMOMETER DEMO.png"
        self.anemometer_image_disconnected_path = "Icons/ANEMOMETER DISCONNECTED.png"
        self.anemometer_image_working_path = "Icons/ANEMOMETER WORKING.png"

        self.barometer_image_demo_path = "Icons/BAROMETER DEMO.png"
        self.barometer_image_disconnected_path = "Icons/BAROMETER DISCONNECTED.png"
        self.barometer_image_working_path = "Icons/BAROMETER WORKING.png"

        self.thermometer_image_demo_path = "Icons/THERMOMETER DEMO.png"
        self.thermometer_image_disconnected_path = "Icons/THERMOMETER DISCONNECTED.png"
        self.thermometer_image_working_path = "Icons/THERMOMETER WORKING.png"

        self.hygrometer_image_demo_path = "Icons/HYGROMETER DEMO.png"
        self.hygrometer_image_disconnected_path = "Icons/HYGROMETER DISCONNECTED.png"
        self.hygrometer_image_working_path = "Icons/HYGROMETER WORKING.png"

        self.demo_icon_image_path = "Icons/DEMO ICON.png"
        self.real_icon_image_path = "Icons/REAL ICON.png"

        self.w = w 
        self.h = h 

    def setDimensions(self, dimW, dimH):
        self.w = dimW 
        self.h = dimH 

    def makeImages(self):
        if (self.w == -1 or self.h == -1):
            raise ArithmeticError(f"Invalid Image Size {self.w} x {self.h}")

        self.WIND_WORK = CTkImage(dark_image=Image.open(self.anemometer_image_working_path), size = (self.w, self.h))
        self.WIND_DEMO = CTkImage(dark_image=Image.open(self.anemometer_image_demo_path), size = (self.w, self.h)) 
        self.WIND_DISCON = CTkImage(dark_image=Image.open(self.anemometer_image_disconnected_path), size = (self.w, self.h))

        self.HUMID_WORK = CTkImage(dark_image=Image.open(self.hygrometer_image_working_path), size = (self.w, self.h))
        self.HUMID_DEMO = CTkImage(dark_image=Image.open(self.hygrometer_image_demo_path), size = (self.w, self.h)) 
        self.HUMID_DISCON = CTkImage(dark_image=Image.open(self.hygrometer_image_disconnected_path), size = (self.w, self.h))

        self.TEMP_WORK = CTkImage(dark_image=Image.open(self.thermometer_image_working_path), size = (self.w, self.h))
        self.TEMP_DEMO = CTkImage(dark_image=Image.open(self.thermometer_image_demo_path), size = (self.w, self.h)) 
        self.TEMP_DISCON = CTkImage(dark_image=Image.open(self.thermometer_image_disconnected_path), size = (self.w, self.h))

        self.PRESSURE_WORK = CTkImage(dark_image=Image.open(self.barometer_image_working_path), size = (self.w, self.h))
        self.PRESSURE_DEMO = CTkImage(dark_image=Image.open(self.barometer_image_demo_path), size = (self.w, self.h)) 
        self.PRESSURE_DISCON = CTkImage(dark_image=Image.open(self.barometer_image_disconnected_path), size = (self.w, self.h))

        self.DEMO_MODE = CTkImage(dark_image=Image.open(self.demo_icon_image_path), size = (self.w, self.h))
        self.REAL_MODE = CTkImage(dark_image=Image.open(self.real_icon_image_path), size = (self.w, self.h))


class WeatherImageIcons:
    def __init__(self, w = -1, h = -1):
        self.cloudy_image_active_path = "WeatherIcons/CLOUDY ACTIVE.png"
        self.cloudy_image_inactive_path = "WeatherIcons/CLOUDY INACTIVE.png"
        self.rainy_image_active_path = "WeatherIcons/RAINY ACTIVE.png"
        self.rainy_image_inactive_path = "WeatherIcons/RAINY INACTIVE.png"
        self.sunny_image_active_path = "WeatherIcons/SUNNY ACTIVE.png"
        self.sunny_image_inactive_path = "WeatherIcons/SUNNY INACTIVE.png"
        self.rainy_and_sunny_image_active_path = "WeatherIcons/RAINY AND SUNNY ACTIVE.png"
        self.rainy_and_sunny_image_inactive_path = "WeatherIcons/RAINY AND SUNNY INACTIVE.png"
        self.w = w
        self.h = h

    def setDimensions(self, dimW, dimH):
        self.w = dimW 
        self.h = dimH 

    def makeImages(self):
        if (self.w == -1 or self.h == -1):
            raise ArithmeticError(f"Invalid Image Size {self.w} x {self.h}")

        self.CLOUDY_ACTIVE = CTkImage(dark_image = Image.open(self.cloudy_image_active_path), size = (self.w, self.h))
        self.CLOUDY_INACTIVE = CTkImage(dark_image = Image.open(self.cloudy_image_inactive_path), size = (self.w, self.h))

        self.RAINY_ACTIVE = CTkImage(dark_image = Image.open(self.rainy_image_active_path), size = (self.w, self.h))
        self.RAINY_INACTIVE = CTkImage(dark_image = Image.open(self.rainy_image_inactive_path), size = (self.w, self.h))

        self.SUNNY_ACTIVE = CTkImage(dark_image = Image.open(self.sunny_image_active_path), size = (self.w, self.h))
        self.SUNNY_INACTIVE = CTkImage(dark_image = Image.open(self.sunny_image_inactive_path), size = (self.w, self.h))

        self.RAINY_AND_SUNNY_ACTIVE = CTkImage(dark_image = Image.open(self.rainy_and_sunny_image_active_path), size = (self.w, self.h))
        self.RAINY_AND_SUNNY_INACTIVE = CTkImage(dark_image = Image.open(self.rainy_and_sunny_image_inactive_path), size = (self.w, self.h))

if __name__ == "__main__":
    # Should Throw Error
    # WeatherImageIcons().makeImages() 
    WeatherImageIcons(10, 10).makeImages()