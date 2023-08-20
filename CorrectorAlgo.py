import random 
import numpy as np

random.seed(5)

def genArr(min_min = 1, max_min = 100, min_width = 1, max_width = 10, min_gap = 0.1, max_gap = 1.0):
    _MIN = random.randint(min_min, max_min)
    _WIDTH = random.randint(min_width, max_width)
    _MAX = _MIN + _WIDTH
    _GAP = random.randint(min_gap * 10, max_gap * 10) / 10

    _ARR = np.arange(_MIN, _MAX + _GAP, _GAP)

    return (_MIN, _WIDTH, _MAX, _GAP, _ARR)

def getWidth(arr): return ((np.max(arr) - np.min(arr)) / (arr[1] -arr[0])) + 1
def calculateEquivalentReference(_in, _ref_width, _ref_gap, _max_test, _min_ref, _min_test): return _min_ref + (_in - _min_test) / (_max_test - _min_test) * (_ref_width - 1) * _ref_gap

REFERENCE_MIN, REFERENCE_WIDTH, REFERENCE_MAX, REFERENCE_GAP, REFERENCE_ARR = genArr(min_gap=0.5, max_gap=0.5)
TESTING_MIN, TESTING_WIDTH, TESTING_MAX, TESTING_GAP, TESTING_ARR = genArr(min_gap=0.5, max_gap=0.5)

print("\nREFERENCE:")
print(f"MIN: {REFERENCE_MIN} \nMAX: {REFERENCE_MAX} \nGAP: {REFERENCE_GAP}")
#print(f"REFERENCE ARR:\n{REFERENCE_ARR}")

print("\nTESTING:")
print(f"MIN: {TESTING_MIN} \nMAX: {TESTING_MAX} \nGAP: {TESTING_GAP}")
#print(f"TESTING ARR:\n{TESTING_ARR}")

# Calculate Reference Width
Calculated_Reference_Width = getWidth(REFERENCE_ARR)
Calculated_Testing_Width = getWidth(TESTING_ARR)
Calculated_Min_Reference = np.min(REFERENCE_ARR)
Calculated_Min_Testing = np.min(TESTING_ARR)
Calculated_Max_Testing = np.max(TESTING_ARR)
Calculated_Reference_Gap = REFERENCE_ARR[1] - REFERENCE_ARR[0]
New_Input = 80
Result = calculateEquivalentReference(New_Input, Calculated_Reference_Width, Calculated_Reference_Gap, Calculated_Max_Testing, Calculated_Min_Reference, Calculated_Min_Testing)

print("\nCalculations:")
print(f"Reference Width: {Calculated_Reference_Width} | Actual : {len(REFERENCE_ARR)}")
print(f"Testing Width: {Calculated_Testing_Width} | Actual : {len(TESTING_ARR)}")
print(f"New Input: {New_Input} | Result Equiv Ref: {Result}")



    

