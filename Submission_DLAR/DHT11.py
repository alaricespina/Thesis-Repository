import adafruit_dht
import board 
import time 

class DHT11:
    def __init__(self):
        self.sensor = adafruit_dht.DHT11(board.D4, use_pulseio=False)
        self.last_temp = 0
        self.last_humidity = 0
    
    def readTemperature(self, verbose = 0):
        try:
            self.last_temp = self.sensor.temperature
            return self.sensor.temperature
        except Exception as E:
            if verbose:
                print("Error reading temperature: ", E)
                print("Returning Previous Temperature")
            return self.last_temp
    
    def readHumidity(self, verbose = 0):
        try:
            self.last_humidity = self.sensor.humidity
            return self.sensor.humidity
        except Exception as E:
            if verbose:
                print("Error reading humidity: ", E)
                print("Returning Previous Humidity")
            return self.last_humidity

if __name__ == "__main__":
    dht11 = DHT11()
    while True:
        print("Temperature: {:.2f} °C".format(dht11.readTemperature()))
        print("Humidity: {:.2f} %".format(dht11.readHumidity()))
        print("Exiting.")
        time.sleep(0.1)