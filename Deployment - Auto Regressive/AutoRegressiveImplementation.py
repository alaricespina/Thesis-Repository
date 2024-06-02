import numpy as np 
import pandas as pd

from pprint import pprint

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class AutoRegressiveClass:
    def __init__(self, model = LinearRegression(), order = 3) -> None:
        self.model = model 
        self.order = order 

    def splitData(self, rawX, train_size = 0.3, test_size = 0.7):
        # Given Raw X Data Split into testing and training data
        # Training Dataset[DateTime, Feature. . .], Testing Dataset[Datetime, Feature. . .] = splitData()
        train_length = 0
        
        if (train_size + test_size > 1.0):
            bigger_size = max(train_size, test_size)
            smaller_size = 1.0 - bigger_size
            train_length = int(smaller_size * len(rawX))           
        else:
            train_length = int(train_size * len(rawX))
        
        Training_Dataset = rawX.copy()[:train_length]
        Testing_Dataset = rawX.copy()[train_length:]
        
        return (Training_Dataset, Testing_Dataset)

    def removeMonthlySeasonality(self, XDataset, date_column = "Date", create_index = True):
        # Given X Dataset, Remove the Monthly Seasonality
        # Assume that ["Date"] Column Exists
        # Normalized_Dataset, {TempMonthlyMean, ...Mean} = removeMonthlySeasonality()

        current_dataset = XDataset.copy()
        dataset_features = current_dataset.columns.to_list()
        
        current_dataset["Month"] = pd.DatetimeIndex(current_dataset[date_column]).month
        
        filter_columns = ["index", date_column, "Month", "Year"]
        for f in filter_columns:
            if f in dataset_features : dataset_features.remove(f)

        existing_columns = [x + "_Mean" for x in dataset_features]
        for f in existing_columns:
            if f in dataset_features : dataset_features.remove(f)
        
        features_means = {}
        dataset_means = {}
        for feature in dataset_features:
            current_dataset[f"{feature}_Mean"] = current_dataset[feature]
            monthly_mean = current_dataset.groupby(by = "Month").mean(numeric_only = True)[feature]
            features_means[f"{feature}_monthly_mean"] = monthly_mean.to_dict()

            for month in current_dataset["Month"].unique():
                current_dataset.loc[current_dataset["Month"] == month, f"{feature}_Mean"] -= monthly_mean[month]
                

            dataset_means[f"{feature}_Mean"] = current_dataset[f"{feature}_Mean"].mean()
            current_dataset[f"{feature}_Mean"] -= dataset_means[f"{feature}_Mean"]
        
        if create_index and "index" not in current_dataset.columns:
            current_dataset = current_dataset.reset_index()

        return (current_dataset, features_means, dataset_means)

    def removeYearlySeasonality(self, XDataset, date_column = "Date", create_index = True):
        # Same as Monthly Seasonality, but this time year instead of Month is used
        current_dataset = XDataset.copy()
        dataset_features = current_dataset.columns.to_list()
        
        current_dataset["Year"] = pd.DatetimeIndex(current_dataset[date_column]).year
        
        filter_columns = ["index", date_column, "Month", "Year"]
        for f in filter_columns:
            if f in dataset_features : dataset_features.remove(f)

        existing_columns = [x + "_Mean" for x in dataset_features]
        for f in existing_columns:
            if f in dataset_features : dataset_features.remove(f)
        

        
        features_means = {}
        dataset_means = {}
        for feature in dataset_features:
            current_dataset[f"{feature}_Year_Mean"] = current_dataset[feature]
            yearly_mean = current_dataset.groupby(by = "Year").mean(numeric_only = True)[feature]
            features_means[f"{feature}_yearly_mean"] = yearly_mean.to_dict()

            for year in current_dataset["Year"].unique():
                current_dataset.loc[current_dataset["Year"] == year, f"{feature}_Mean"] -= yearly_mean[year]

            dataset_means[f"{feature}_Year_Mean"] = current_dataset[f"{feature}_Mean"].mean()
            current_dataset[f"{feature}_Year_Mean"] -= dataset_means[f"{feature}_Mean"]
        
        if create_index and "index" not in current_dataset.columns:
            current_dataset = current_dataset.reset_index()

        return (current_dataset, features_means, dataset_means)

    def prepareTrainingData():
        pass 

    def prepareTestingData():
        pass

    def modelPredict():
        pass 

    def modelFit():
        pass 
    
    def addMonthlySeasonality():
        pass 

    def addYearlySeasonality():
        pass 

    def showSeasonality():
        pass 

    def showPredictions():
        pass 

    def showTrainingData():
        pass 

    def showTestingData():
        pass

    def showSanityCheck():
        pass 

if __name__ == "__main__":
    ARC = AutoRegressiveClass()
    dataRaw = pd.read_csv("CombinedData.csv")
    dataRaw = dataRaw[["datetime", "temp"]]
    dataRaw.rename(columns={"datetime" : "Date", "temp" : "Temp"}, inplace=True)
    Training_Dataset, Testing_Dataset = ARC.splitData(dataRaw)
    # Normalized Training Dataset (NTD), Training Feature Monthly Mean (TFMM), Training Datset Mean (TDM)
    NMTD, TFMM, TDM = ARC.removeMonthlySeasonality(Training_Dataset)
    NYTD, TFYM, TYDM = ARC.removeYearlySeasonality(NMTD)
    NYTD.to_csv("PreviewMonthlySeasonality.csv")
    pprint(NYTD)


