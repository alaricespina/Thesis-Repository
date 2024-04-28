import os 
import pandas as pd 

print(os.listdir("Data"))

files = []
for filename in os.listdir("Data"):
    current = pd.read_csv("Data/"+filename)
    files.append(current)

concatenated_data = pd.concat(files, ignore_index=True, sort=False)
concatenated_data.to_csv("Concatenated Data.csv")
