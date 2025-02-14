import smbus
import time

class BMP180:
    def __init__(self):
        # BMP180 address
        self.BMP180_ADDRESS = 0x77

        # Control Register Address
        self.CONTROL = 0xF4

        # Read Temperature Command
        self.READ_TEMP = 0x2E

        # Read Pressure Command
        self.READ_PRESSURE = 0x34

        # Oversampling Setting (OSS):  0, 1, 2, or 3
        # Increasing OSS improves accuracy but increases conversion time.
        self.OSS = 3  # Try higher oversampling for more accuracy

        # Pressure Read Settings, based on OSS value
        self.PRESSURE_OSS = {
            0: 0x34,
            1: 0x74,
            2: 0xB4,
            3: 0xF4
        }

        # Wait times for pressure reads, based on OSS value (in seconds)
        self.PRESSURE_WAIT = {
            0: 0.005,
            1: 0.008,
            2: 0.014,
            3: 0.026
        }

        # Initialize I2C bus
        self.bus = smbus.SMBus(1)  # Use 1 for Raspberry Pi models 1, 2, 3, and 4

    # Function to read a two-byte integer from the I2C bus (MSB first)
    def read_2_bytes(self, address):
        msb = self.bus.read_byte_data(self.BMP180_ADDRESS, address)
        lsb = self.bus.read_byte_data(self.BMP180_ADDRESS, address + 1)
        value = (msb << 8) | lsb  # Use bitwise OR to combine bytes
        if value > 32767:  # Handle signed integers. Important!
            value -= 65536
        return value

    # Function to read the calibration data from the BMP180
    def read_calibration_data(self):
        # Calibration Data Addresses
        CAL_AC1 = 0xAA
        CAL_AC2 = 0xAC
        CAL_AC3 = 0xAE
        CAL_AC4 = 0xB0
        CAL_AC5 = 0xB2
        CAL_AC6 = 0xB4
        CAL_B1  = 0xB6
        CAL_B2  = 0xB8
        CAL_MB  = 0xBA
        CAL_MC  = 0xBC
        CAL_MD  = 0xBE

        cal_data = {}
        cal_data['AC1'] = self.read_2_bytes(CAL_AC1)
        cal_data['AC2'] = self.read_2_bytes(CAL_AC2)
        cal_data['AC3'] = self.read_2_bytes(CAL_AC3)
        cal_data['AC4'] = self.read_2_bytes(CAL_AC4)
        cal_data['AC5'] = self.read_2_bytes(CAL_AC5)
        cal_data['AC6'] = self.read_2_bytes(CAL_AC6)
        cal_data['B1']  = self.read_2_bytes(CAL_B1)
        cal_data['B2']  = self.read_2_bytes(CAL_B2)
        cal_data['MB']  = self.read_2_bytes(CAL_MB)
        cal_data['MC']  = self.read_2_bytes(CAL_MC)
        cal_data['MD']  = self.read_2_bytes(CAL_MD)

        # Print calibration data for inspection
        # print("Calibration Data:")
        # for key, value in cal_data.items():
        #     print(f"  {key}: {value}")

        return cal_data

    # Function to read the uncompensated temperature value
    def read_raw_temperature(self):
        self.bus.write_byte_data(self.BMP180_ADDRESS, self.CONTROL, self.READ_TEMP)
        time.sleep(0.005)  # Wait 4.5 ms
        msb = self.bus.read_byte_data(self.BMP180_ADDRESS, 0xF6)
        lsb = self.bus.read_byte_data(self.BMP180_ADDRESS, 0xF7)
        return (msb << 8) + lsb

    # Function to read the uncompensated pressure value
    def read_raw_pressure(self):
        self.bus.write_byte_data(self.BMP180_ADDRESS, self.CONTROL, self.PRESSURE_OSS[self.OSS])
        time.sleep(self.PRESSURE_WAIT[self.OSS])  # Wait based on OSS setting
        msb = self.bus.read_byte_data(self.BMP180_ADDRESS, 0xF6)
        lsb = self.bus.read_byte_data(self.BMP180_ADDRESS, 0xF7)
        xlsb = self.bus.read_byte_data(self.BMP180_ADDRESS, 0xF8)
        return ((msb << 16) + (lsb << 8) + xlsb) >> (8 - self.OSS)


    # Function to calculate the temperature in degrees Celsius
    def calculate_temperature(self, ut, cal_data):
        X1 = ((ut - cal_data['AC6']) * cal_data['AC5']) >> 15
        X2 = (cal_data['MC'] << 11) // (X1 + cal_data['MD'])
        B5 = X1 + X2
        T = (B5 + 8) >> 4
        return T / 10.0

    # Function to calculate the pressure in Pascals
    def calculate_pressure(self, up, cal_data, b5):
        B6 = b5 - 4000
        X1 = (cal_data['B2'] * (B6 * B6) // 2**12) >> 11
        X2 = (cal_data['AC2'] * B6) >> 11
        X3 = X1 + X2
        B3 = (((cal_data['AC1'] * 4 + X3) << self.OSS) + 2) // 4
        X1 = (cal_data['AC3'] * B6) >> 13
        X2 = (cal_data['B1'] * (B6 * B6) // 2**12) >> 16
        X3 = ((X1 + X2) + 2) >> 2
        B4 = (cal_data['AC4'] * (X3 + 32768)) >> 15
        B7 = (up - B3) * (50000 >> self.OSS)

        if B7 < 0x80000000:
            p = (B7 * 2) // B4
        else:
            p = (B7 // B4) * 2

        X1 = (p >> 8) * (p >> 8)
        X1 = (X1 * 3038) >> 16
        X2 = (-7357 * p) >> 16

        p = p + ((X1 + X2 + 3791) >> 4)
        return p
    
    def get_sensor_data(self, verbose = 0):
        # Main program
        try:
            # Read the calibration data
            calibration_data = self.read_calibration_data()

            # Read the uncompensated temperature and pressure values
            ut = self.read_raw_temperature()
            up = self.read_raw_pressure()

            # Calculate the temperature
            temperature = self.calculate_temperature(ut, calibration_data)

            # Calculate B5 - Intermediate value required for pressure calculation
            x1 = ((ut - int(calibration_data['AC6'])) * int(calibration_data['AC5'])) >> 15
            x2 = (int(calibration_data['MC']) << 11) // (x1 + int(calibration_data['MD']))
            b5 = x1 + x2


            # Calculate the pressure
            pressure = self.calculate_pressure(up, calibration_data, b5)

            # Print the results
            if verbose:
                print("Temperature: {:.2f} °C".format(temperature))
                print("Pressure: {:.2f} Pa".format(pressure))
                print("Pressure: {:.2f} hPa".format(pressure / 100))
                print("-" * 20)

            return {"temperature" : temperature, "pressure" : pressure}

        except IOError as e:
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            print("Make sure I2C is enabled and the BMP180 is connected correctly.")
            return None 
        except KeyboardInterrupt:
            print("Program stopped by user.")
            return None 
    
    def readPressure(self):
        calibration_data = self.read_calibration_data()
        ut = self.read_raw_temperature()
        up = self.read_raw_pressure()
        x1 = ((ut - int(calibration_data['AC6'])) * int(calibration_data['AC5'])) >> 15
        x2 = (int(calibration_data['MC']) << 11) // (x1 + int(calibration_data['MD']))
        b5 = x1 + x2
        pressure = self.calculate_pressure(up, calibration_data, b5)
        return pressure

    def readTemperature(self):
        calibration_data = self.read_calibration_data()
        ut = self.read_raw_temperature()
        temperature = self.calculate_temperature(ut, calibration_data)
        return temperature

if __name__ == "__main__":
    bmp180 = BMP180()
    bmp180.get_sensor_data(verbose = 1)
    print("Exiting.")