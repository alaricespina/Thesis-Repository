import time, datetime 
import RPi.GPIO as GPIO 

class HALL:
    def __init__(self):        
        self.left_sensor_pin = 15 
        self.right_sensor_pin = 13
        # GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.left_sensor_pin, GPIO.IN)
        GPIO.setup(self.right_sensor_pin, GPIO.IN)
        self.left_last_time = time.time()
        self.right_last_time = 0
        self.left_rotation_time = 0
        self.right_rotation_time = 0
        self.left_sensor_value = 0
        self.right_sensor_value = 0
        self.last_left_sensor_value = 0
        self.last_right_sensor_value = 0

    def readRawLeftSensor(self):
        return GPIO.input(self.left_sensor_pin)
    
    def readRawRightSensor(self):
        return GPIO.input(self.right_sensor_pin)
    
    def readSpeed(self):
        self.left_sensor_value = self.readRawLeftSensor()
        # self.right_sensor_value = self.readRawRightSensor()

        if self.left_sensor_value == GPIO.LOW and self.last_left_sensor_value == GPIO.HIGH:
            current_time = time.time()
            self.left_rotation_time = (current_time - self.left_last_time)
            self.left_last_time = current_time
        
        self.last_left_sensor_value = self.left_sensor_value
        

        # if self.right_sensor_value == GPIO.LOW and self.last_right_sensor_value == GPIO.HIGH:
        #     current_time = time.time()
        #     self.right_rotation_time = (current_time - self.right_last_time)
        #     self.right_last_time = current_time
        #     self.last_right_sensor_value = self.right_sensor_value

        rpm1 = 0
        # rpm2 = 0

        if self.left_rotation_time > 0:
            rpm1 = 60.0 / self.left_rotation_time

        return rpm1 

        # if self.right_rotation_time != 0:
        #     rpm2 = 60.0 / (self.right_rotation_time * 6) 

        # valid_rpm_count = 0
        # total_rpm = 0

        # if rpm1 != 0:
        #     total_rpm += rpm1
        #     valid_rpm_count += 1

        # if rpm2 != 0:
        #     total_rpm += rpm2
        #     valid_rpm_count += 1

        # return total_rpm / valid_rpm_count if valid_rpm_count > 0 else 0


if __name__ == "__main__":
    GPIO.setmode(GPIO.BOARD)
    sensor = HALL()
    while True:
        print(f"RPM: {sensor.readSpeed():.2f} L: {sensor.readRawLeftSensor()} R: {sensor.readRawRightSensor()}")
        time.sleep(0.1)


# import RPi.GPIO as GPIO
# import time

# sensor_pin1 = 17  # GPIO pin connected to Hall sensor 1 output
# sensor_pin2 = 18  # GPIO pin connected to Hall sensor 2 output

# last_time1 = 0
# last_time2 = 0

# rotation_time1 = 0
# rotation_time2 = 0

# magnet_count = 6  # Number of magnets on the disc (CHANGE THIS!)

# GPIO.setmode(GPIO.BCM)  # Use Broadcom SOC channel numbering
# GPIO.setup(sensor_pin1, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Input with pull-up resistor
# GPIO.setup(sensor_pin2, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Input with pull-up resistor

# try:
#     while True:
#         # Read the sensor values
#         sensor_value1 = GPIO.input(sensor_pin1)
#         sensor_value2 = GPIO.input(sensor_pin2)

#         # Check for a magnet passing by sensor 1
#         if sensor_value1 == GPIO.LOW:  # Assuming LOW when magnet is detected
#             current_time = time.time()  # Get current time in seconds
#             rotation_time1 = (current_time - last_time1)
#             last_time1 = current_time

#         # Check for a magnet passing by sensor 2
#         if sensor_value2 == GPIO.LOW:  # Assuming LOW when magnet is detected
#             current_time = time.time()  # Get current time in seconds
#             rotation_time2 = (current_time - last_time2)
#             last_time2 = current_time

#         # Calculate RPM (Revolutions Per Minute)
#         rpm1 = 0
#         rpm2 = 0

#         if rotation_time1 > 0:
#             rpm1 = 60.0 / (rotation_time1 * magnet_count)  # 60 seconds in a minute
#         if rotation_time2 > 0:
#             rpm2 = 60.0 / (rotation_time2 * magnet_count)  # 60 seconds in a minute

#         # Combine RPMs using simple averaging
#         valid_rpm_count = 0
#         total_rpm = 0
#         if rpm1 > 0:
#             total_rpm += rpm1
#             valid_rpm_count += 1
#         if rpm2 > 0:
#             total_rpm += rpm2
#             valid_rpm_count += 1

#         if valid_rpm_count > 0:
#             average_rpm = total_rpm / valid_rpm_count
#         else:
#             average_rpm = 0  # Or some other default value

#         # Print the RPM to the console
#         print(f"RPM 1: {rpm1:.2f}  RPM 2: {rpm2:.2f}  Average RPM: {average_rpm:.2f}")

#         time.sleep(0.1)  # Small delay

# except KeyboardInterrupt:
#     print("Exiting...")
# finally:
#     GPIO.cleanup() 






# import time, datetime
# import RPi.GPIO as GPIO
# sense_pin = 15
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(sense_pin, GPIO.IN)

# def detect(channel):
#     timedate = datetime.datetime.now().strftime("%H:%M:%S %a")
#     if GPIO.input(sense_pin) == 1:
#         print("Detect off @", timedate)
#     else:
#         print("Detect on @", timedate)
    
#     return()

# GPIO.add_event_detect(sense_pin, GPIO.BOTH, callback=detect, bouncetime=20)
# print("Ctrl C to quit")

# try:
#     for i in range(1000):
#         timedate = datetime.datetime.now().strftime("%H:%M:%S %a")
#         print("System Active Check", timedate)
#         time.sleep(0.5)
# except:
#     time.sleep(2)
#     GPIO.remove_event_detect(sense_pin)
#     GPIO.cleanup()
#     print("Done")
