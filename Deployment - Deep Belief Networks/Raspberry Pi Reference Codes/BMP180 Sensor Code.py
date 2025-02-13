import smbus
import time

# BMP180 address
BMP180_ADDRESS = 0x77

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

# Control Register Address
CONTROL = 0xF4

# Read Temperature Command
READ_TEMP = 0x2E

# Read Pressure Command
READ_PRESSURE = 0x34

# Oversampling Setting (OSS):  0, 1, 2, or 3
# Increasing OSS improves accuracy but increases conversion time.
OSS = 3  # Try higher oversampling for more accuracy

# Pressure Read Settings, based on OSS value
PRESSURE_OSS = {
    0: 0x34,
    1: 0x74,
    2: 0xB4,
    3: 0xF4
}

# Wait times for pressure reads, based on OSS value (in seconds)
PRESSURE_WAIT = {
    0: 0.005,
    1: 0.008,
    2: 0.014,
    3: 0.026
}

# Initialize I2C bus
bus = smbus.SMBus(1)  # Use 1 for Raspberry Pi models 1, 2, 3, and 4

# Function to read a two-byte integer from the I2C bus (MSB first)
def read_2_bytes(address):
    msb = bus.read_byte_data(BMP180_ADDRESS, address)
    lsb = bus.read_byte_data(BMP180_ADDRESS, address + 1)
    value = (msb << 8) | lsb  # Use bitwise OR to combine bytes
    if value > 32767:  # Handle signed integers. Important!
        value -= 65536
    return value

# Function to read the calibration data from the BMP180
def read_calibration_data():
    cal_data = {}
    cal_data['AC1'] = read_2_bytes(CAL_AC1)
    cal_data['AC2'] = read_2_bytes(CAL_AC2)
    cal_data['AC3'] = read_2_bytes(CAL_AC3)
    cal_data['AC4'] = read_2_bytes(CAL_AC4)
    cal_data['AC5'] = read_2_bytes(CAL_AC5)
    cal_data['AC6'] = read_2_bytes(CAL_AC6)
    cal_data['B1']  = read_2_bytes(CAL_B1)
    cal_data['B2']  = read_2_bytes(CAL_B2)
    cal_data['MB']  = read_2_bytes(CAL_MB)
    cal_data['MC']  = read_2_bytes(CAL_MC)
    cal_data['MD']  = read_2_bytes(CAL_MD)

    # Print calibration data for inspection
    print("Calibration Data:")
    for key, value in cal_data.items():
        print(f"  {key}: {value}")

    return cal_data

# Function to read the uncompensated temperature value
def read_raw_temperature():
    bus.write_byte_data(BMP180_ADDRESS, CONTROL, READ_TEMP)
    time.sleep(0.005)  # Wait 4.5 ms
    msb = bus.read_byte_data(BMP180_ADDRESS, 0xF6)
    lsb = bus.read_byte_data(BMP180_ADDRESS, 0xF7)
    return (msb << 8) + lsb

# Function to read the uncompensated pressure value
def read_raw_pressure():
    bus.write_byte_data(BMP180_ADDRESS, CONTROL, PRESSURE_OSS[OSS])
    time.sleep(PRESSURE_WAIT[OSS])  # Wait based on OSS setting
    msb = bus.read_byte_data(BMP180_ADDRESS, 0xF6)
    lsb = bus.read_byte_data(BMP180_ADDRESS, 0xF7)
    xlsb = bus.read_byte_data(BMP180_ADDRESS, 0xF8)
    return ((msb << 16) + (lsb << 8) + xlsb) >> (8 - OSS)


# Function to calculate the temperature in degrees Celsius
def calculate_temperature(ut, cal_data):
    X1 = ((ut - cal_data['AC6']) * cal_data['AC5']) >> 15
    X2 = (cal_data['MC'] << 11) // (X1 + cal_data['MD'])
    B5 = X1 + X2
    T = (B5 + 8) >> 4
    return T / 10.0

# Function to calculate the pressure in Pascals
def calculate_pressure(up, cal_data, b5):
    B6 = b5 - 4000
    X1 = (cal_data['B2'] * (B6 * B6) // 2**12) >> 11
    X2 = (cal_data['AC2'] * B6) >> 11
    X3 = X1 + X2
    B3 = (((cal_data['AC1'] * 4 + X3) << OSS) + 2) // 4
    X1 = (cal_data['AC3'] * B6) >> 13
    X2 = (cal_data['B1'] * (B6 * B6) // 2**12) >> 16
    X3 = ((X1 + X2) + 2) >> 2
    B4 = (cal_data['AC4'] * (X3 + 32768)) >> 15
    B7 = (up - B3) * (50000 >> OSS)

    if B7 < 0x80000000:
        p = (B7 * 2) // B4
    else:
        p = (B7 // B4) * 2

    X1 = (p >> 8) * (p >> 8)
    X1 = (X1 * 3038) >> 16
    X2 = (-7357 * p) >> 16

    p = p + ((X1 + X2 + 3791) >> 4)
    return p

# Main program
try:
    # Read the calibration data
    calibration_data = read_calibration_data()

    # Read the uncompensated temperature and pressure values
    ut = read_raw_temperature()
    up = read_raw_pressure()

    # Calculate the temperature
    temperature = calculate_temperature(ut, calibration_data)

    # Calculate B5 - Intermediate value required for pressure calculation
    x1 = ((ut - int(calibration_data['AC6'])) * int(calibration_data['AC5'])) >> 15
    x2 = (int(calibration_data['MC']) << 11) // (x1 + int(calibration_data['MD']))
    b5 = x1 + x2


    # Calculate the pressure
    pressure = calculate_pressure(up, calibration_data, b5)

    # Print the results
    print("Temperature: {:.2f} °C".format(temperature))
    print("Pressure: {:.2f} Pa".format(pressure))
    print("Pressure: {:.2f} hPa".format(pressure / 100))
    print("-" * 20)

    time.sleep(2)

except IOError as e:
    print("I/O error({0}): {1}".format(e.errno, e.strerror))
    print("Make sure I2C is enabled and the BMP180 is connected correctly.")
except KeyboardInterrupt:
    print("Program stopped by user.")
finally:
    print("Exiting.")