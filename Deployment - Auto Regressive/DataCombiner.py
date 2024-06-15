import pandas as pd 
import os 

listOfFiles = []
folderName = "Raw Atmospheric Data"
initial_file = None
last_file = None
for file in os.listdir(folderName):
    if initial_file == None:
        initial_file = file.replace(".csv", "") 

    if "2024" not in file:
        # print(file)
        listOfFiles.append(pd.read_csv(os.path.join(folderName, file)))
        last_file = file.replace(".csv", "")

print("Initial File is:", initial_file, "Last File is:", last_file)

pd.concat(listOfFiles, ignore_index = True, sort = False ).to_csv(f"{initial_file} to {last_file} CombinedData.csv")

print("Done Concatenating")