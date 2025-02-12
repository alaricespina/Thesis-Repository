import time, datetime
import RPi.GPIO as GPIO
sense_pin = 15
GPIO.setmode(GPIO.BOARD)
GPIO.setup(sense_pin, GPIO.IN)

def detect(channel):
    timedate = datetime.datetime.now().strftime("%H:%M:%S %a")
    if GPIO.input(sense_pin) == 1:
        print("Detect off @", timedate)
    else:
        print("Detect on @", timedate)
    
    return()

GPIO.add_event_detect(sense_pin, GPIO.BOTH, callback=detect, bouncetime=20)

print("Ctrl C to quit")

try:
    while (True):
        timedate = datetime.datetime.now().strftime("%H:%M:%S %a")
        print("System Active Check", timedate)
        time.sleep(0.5)
except:
    time.sleep(2)
    GPIO.remove_event_detect(sense_pin)
    GPIO.cleanup()
    print("Done")
