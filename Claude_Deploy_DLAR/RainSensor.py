import RPi.GPIO as GPIO
import time 

class RainSensor():
    def __init__(self, pin=23):
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.IN)
    
    def readRain(self, verbose=0):
        return GPIO.input(self.pin)
    
    def cleanup(self):
        GPIO.cleanup()
    