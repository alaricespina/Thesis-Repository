import pandas as pd 
import os 

listOfFiles = []
folderName = "Raw Atmospheric Data"
for file in os.listdir(folderName):
    if "2024" not in file:
        print(file)
        listOfFiles.append(pd.read_csv(os.path.join(folderName, file)))
    

pd.concat(listOfFiles, ignore_index = True, sort = False ).to_csv("1998 to 2023 CombinedData.csv")

print("Done Concatenating")