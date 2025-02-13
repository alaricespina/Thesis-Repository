class DHT11:
    def __init__(self):
        import adafruit_dht
        import board 

        self.sensor = adafruit_dht.DHT11(board.D4)
    
    def readTemperature(self):
        return self.sensor.temperature
    
    def readHumidity(self):
        return self.sensor.humidity