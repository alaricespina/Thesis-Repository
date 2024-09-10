# importing libraries
import numpy as np
import time
import datetime
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt

GPIO.setmode(GPIO.BCM)
GPIO.setup(17 , GPIO.IN)

# Loop
while True:
    print(GPIO.input(17))
    time.sleep(0.1)




