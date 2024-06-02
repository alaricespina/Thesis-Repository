import pandas as pd 
import os 

listOfFiles = []
for file in os.listdir("Data Raw"):
    print(file)
    if "2024" not in file:
        listOfFiles.append(pd.read_csv(os.path.join("Data Raw", file)))
    

pd.concat(listOfFiles, ignore_index = True, sort = False ).to_csv("CombinedData.csv")
print("Done Concatenating")