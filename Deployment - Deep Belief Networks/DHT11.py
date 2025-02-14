import adafruit_dht
import board 
import time 

class DHT11:
    def __init__(self):
        self.sensor = adafruit_dht.DHT11(board.D4, use_pulseio=False)
    
    def readTemperature(self):
        return self.sensor.temperature
    
    def readHumidity(self):
        return self.sensor.humidity
    
if __name__ == "__main__":
    dht11 = DHT11()
    while True:
        print("Temperature: {:.2f} °C".format(dht11.readTemperature()))
        print("Humidity: {:.2f} %".format(dht11.readHumidity()))
        print("Exiting.")
        time.sleep(0.1)