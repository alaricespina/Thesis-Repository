import random 
import pandas as pd 
import numpy as np 
from pprint import pprint
from copy import deepcopy

# Recommended Range = 1 - 5
DIFFICULTY = 5

# Input Parameters
INPUT_MIN = 1
INPUT_MAX = 10

# Problem Type, 0 - Regression, 1 - Classification
PROBLEM_TYPE = 0

# Output Partition, Only for Classification
PARTITION_NUM = 10

def generateEq(diff, i_min, i_max):
    # Constants
    Constants = [random.randint(i_min, i_max) for _ in range(diff)]

    # Constants Operations, 0 - Add, 1 - Sub, 2 - Mult, 3 - Div
    Con_Op = [random.randint(0, 3) for _ in range(diff)]

    # Variable Operation, 0 - Mult, 1 - Div
    Var_Op = [random.randint(0, 1) for _ in range(diff)]

    # Variable Coefficients
    Var_Coeff = [random.randint(i_min, i_max) for _ in range(diff)]

    return (Constants, Con_Op, Var_Op, Var_Coeff)

def generateInputs(diff, i_min, i_max):
    # Input Vars
    Input_Vars = [random.randint(i_min, i_max) for _ in range(diff)]

    return Input_Vars

def computeGivenInputs(con, con_op, var_op, var_coeff, input_vars):
    running_sum = 0

    for Con, CO, VO, VC, IV in zip(con, con_op, var_op, var_coeff, input_vars):
        raw_input = IV
        
        if VO == 0:
            first_step = raw_input * VC
        elif VO == 1:
            first_step = raw_input / VC 
        
        if CO == 0:
            second_step = first_step + Con 
        elif CO == 1:
            second_step = first_step - Con
        elif CO == 2:
            second_step = first_step * Con 
        elif CO == 3:
            second_step = first_step / Con 

    running_sum += second_step 

    return running_sum 

def parseEquation(con, con_op, var_op, var_coeff):
    running_string = "0"

    for Con, CO, VO, VC in zip(con, con_op, var_op, var_coeff):
        raw_input = "X"
        
        if VO == 0:
            running_string += f" + ({VC} * ({raw_input})"
        elif VO == 1:
            running_string += f" + (({raw_input})/{VC}"
        
        if CO == 0:
            running_string += f" + {Con})"
        elif CO == 1:
            running_string += f" - {Con})" 
        elif CO == 2:
            running_string += f" * {Con})"
        elif CO == 3:
            running_string += f" / {Con})"

    return running_string

def computeMinMax(con, con_op, var_op, var_coeff, i_min, i_max):
    comp_diff = len(var_coeff)

    input_arrs = []

    if comp_diff == 5:
        for a in range(i_min, i_max+1): 
            for b in range(i_min, i_max+1): 
                for c in range(i_min, i_max+1): 
                    for d in range(i_min, i_max+1): 
                        for e in range(i_min, i_max+1): 
                            input_arrs.append([a, b, c, d, e])

    min_value = 1000000
    max_value = 0

    for input_comb in input_arrs:
        val = computeGivenInputs(con, con_op, var_op, var_coeff, input_comb)
        if val < min_value:
            min_value = deepcopy(val)
        
        if val > max_value:
            max_value = deepcopy(val)

    return min_value, max_value

def getPercentile(min_val, max_val, test_val):
    return int(((test_val - min_val) / (max_val - min_val)) * 10)



    

Constants, Con_Op, Var_Op, Var_Coeff = generateEq(DIFFICULTY, INPUT_MIN, INPUT_MAX)
Input_Vars = generateInputs(DIFFICULTY, INPUT_MIN, INPUT_MAX)
Eq = parseEquation(Constants, Con_Op, Var_Op, Var_Coeff)
Res = computeGivenInputs(Constants, Con_Op, Var_Op, Var_Coeff, Input_Vars)
Min_Val, Max_Val = computeMinMax(Constants, Con_Op, Var_Op, Var_Coeff, INPUT_MIN, INPUT_MAX)
res_percentile = getPercentile(Min_Val, Max_Val, Res)

t_s = f"""
    Equation: {Eq}
    Input Variables : {Input_Vars}
    Result from Input Variables : {Res}
    Percentile (Class) : {res_percentile}
    Minimum Value : {Min_Val}
    Maximum Value : {Max_Val}
"""
print(t_s)

Data_Length = 10000
New_Input_Vars = [generateInputs(DIFFICULTY, INPUT_MIN, INPUT_MAX) for _ in range(Data_Length)]
New_Results = [computeGivenInputs(Constants, Con_Op, Var_Op, Var_Coeff, New_Input_Vars[x]) for x in range(Data_Length)]
Min_Val, Max_Val = computeMinMax(Constants, Con_Op, Var_Op, Var_Coeff, INPUT_MIN, INPUT_MAX)
Res_Classes = [getPercentile(Min_Val, Max_Val, X) for X in New_Results]

d_df = {
    "Inputs" : New_Input_Vars,
    "Outputs" : New_Results,
    "Output Class" : Res_Classes
}

for x in range(DIFFICULTY):
    d_df[f"Inputs {x}"] = [i[x] for i in New_Input_Vars]

t_df = pd.DataFrame(d_df)

print(t_df.head())

t_df.to_csv("Generated Data.csv")






    
    