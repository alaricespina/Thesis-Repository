import pandas as pd 
import os 

listOfFiles = []
for file in os.scandir("Data Raw"):
    listOfFiles.append(pd.read_csv(file))

pd.concat(listOfFiles, ignore_index = True, sort = False ).to_csv("CombinedData.csv")
print("Done Concatenating")