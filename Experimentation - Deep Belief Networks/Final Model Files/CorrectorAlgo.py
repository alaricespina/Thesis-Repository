import random 
import numpy as np

"HATDOG"

random.seed(5)

class Corrector():
    def __init__(self, generate=False) -> None:
        self.REFERENCE_MIN, self.REFERENCE_WIDTH, self.REFERENCE_MAX, self.REFERENCE_GAP, self.REFERENCE_ARR = self.genArr(min_gap=0.5, max_gap=0.5)
        self.TESTING_MIN, self.TESTING_WIDTH, self.TESTING_MAX, self.TESTING_GAP, self.TESTING_ARR = self.genArr(min_gap=0.5, max_gap=0.5)
        print("FIRST DEFINITION")
        
    def printTestingReferenceArrays(self):
        print("\nREFERENCE:")
        print(f"MIN: {self.REFERENCE_MIN} \nMAX: {self.REFERENCE_MAX} \nGAP: {self.REFERENCE_GAP}")

        print("\nTESTING:")
        print(f"MIN: {self.TESTING_MIN} \nMAX: {self.TESTING_MAX} \nGAP: {self.TESTING_GAP}")

    def genArr(self, min_min = 1, max_min = 100, min_width = 1, max_width = 10, min_gap = 0.1, max_gap = 1.0):
        _MIN = random.randint(min_min, max_min)
        _WIDTH = random.randint(min_width, max_width)
        _MAX = _MIN + _WIDTH
        _GAP = random.randint(min_gap * 10, max_gap * 10) / 10

        _ARR = np.arange(_MIN, _MAX + _GAP, _GAP)

        return (_MIN, _WIDTH, _MAX, _GAP, _ARR)

    def getWidth(self, arr): 
        return ((np.max(arr) - np.min(arr)) / (arr[1] -arr[0])) + 1
    
    def calculateEquivalentReference(self, _in, _ref_width, _ref_gap, _max_test, _min_ref, _min_test): 
        return _min_ref + (_in - _min_test) / (_max_test - _min_test) * (_ref_width - 1) * _ref_gap
    
    def interpolateFromData(filename, object, is_File=True, is_Object=False):
        # Used in Fit Function
        pass 

    def fit(self):
        Calculated_Reference_Width = self.getWidth(self.REFERENCE_ARR)
        Calculated_Testing_Width = self.getWidth(self.TESTING_ARR)
        Calculated_Min_Reference = np.min(self.REFERENCE_ARR)
        Calculated_Min_Testing = np.min(self.TESTING_ARR)
        Calculated_Max_Testing = np.max(self.TESTING_ARR)
        Calculated_Reference_Gap = self.REFERENCE_ARR[1] - self.REFERENCE_ARR[0]
        New_Input = 80
        Result = self.calculateEquivalentReference(New_Input, Calculated_Reference_Width, Calculated_Reference_Gap, Calculated_Max_Testing, Calculated_Min_Reference, Calculated_Min_Testing) 

        print("\nCalculations:")
        print(f"Reference Width: {Calculated_Reference_Width} | Actual : {len(self.REFERENCE_ARR)}")
        print(f"Testing Width: {Calculated_Testing_Width} | Actual : {len(self.TESTING_ARR)}")
        print(f"New Input: {New_Input} | Result Equiv Ref: {Result}")

if __name__ == "__main__":
    Test = Corrector()
    Test.printTestingReferenceArrays()
    Test.fit()




    

