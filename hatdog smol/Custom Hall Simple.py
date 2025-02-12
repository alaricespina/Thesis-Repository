# importing libraries
import numpy as np
import time
import datetime
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt

GPIO.setmode(GPIO.BOARD)
GPIO.setup(15 , GPIO.IN)

# Loop
while True:
    print(GPIO.input(15))
    time.sleep(0.1)




