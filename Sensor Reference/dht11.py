import time
import board
import adafruit_dht

sensor = adafruit_dht.DHT11(board.D4)
print("Hatdog")
print("Sensor Reading:", sensor.temperature)
print("Sensor Humidity:", sensor.humidity)
