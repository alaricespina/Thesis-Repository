import sys
import Adafruit_DHT
import time

T_I = 1
X = 30
print(f"Reading in Intervals of {T_I} seconds for {X} times")

# Read_Retry(DHT11 -> 11, GPIO23 -> 23)
for _ in range(X):
    humidity, temperature = Adafruit_DHT.read_retry(11, 23)
    print(f"Temperature : {temperature}, Humidity : {humidity}")
    time.sleep(T_I)


print("Test Complete")