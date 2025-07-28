import RPi.GPIO as GPIO
import time 

class RainSensor():
    def __init__(self, pin=23):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN)
    
    def readRain(self, verbose=0):
        return GPIO.input(self.pin)
    
    def read_precipitation(self):
        rain_state = self.readRain()
        return 0.0 if rain_state == 0 else 1.0
    
    def cleanup(self):
        GPIO.cleanup()