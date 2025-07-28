import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from tqdm import tqdm
from sklearn.linear_model import LinearRegression

from datetime import datetime, timedelta

from IPython.display import clear_output

class AutoRegressiveProcess():
    def __init__(self, MODEL_ORDER = 2, MONTHLY_SCALE = 0, YEARLY_SCALE = 0, YEARLY_LENGTH = 7):
        self.MODEL_ORDER = MODEL_ORDER 
        self.MONTHLY_STD_SCALE = MONTHLY_SCALE
        self.YEARLY_STD_SCALE = YEARLY_SCALE 
        self.MONTHLY_STD = None 
        self.YEARLY_STD = None 
        self.YEAR_LENGTH = YEARLY_LENGTH
    
    
    def executeProcess(self, targetColumnName, rawDataFrame, monthlyResolution = False):
        # Add Month and Year
        self.RawDF = rawDataFrame.copy()
        self.RawDF["Month"] = pd.DatetimeIndex(self.RawDF["datetime"]).month 
        self.RawDF["Year"] = pd.DatetimeIndex(self.RawDF["datetime"]).year 

        self.targetColumn = targetColumnName

        pass 

    def removeYearlyMean(self):
        self.NormalizedYearlyDF = self.RawDF.copy()
        self.NormalizedYearlyDF["DataMod"] = self.NormalizedYearlyDF[self.targetColumn]

        availableYears = self.RawDF["Year"].unique().to_list()
        minimumYear = availableYears[0]

        normalizedYears = [(x - minimumYear) % self.YEAR_LENGTH for x in availableYears]

        self.NormalizedYearlyDF["YearAug"] = (self.NormalizedYearlyDF["datetime"].year - minimumYear) % self.YEAR_LENGTH

        YearlyMeans = self.NormalizedYearlyDF.groupby(by = "YearAug").mean(numeric_only = True)[self.targetColumn]

        moddedColumnName = "Mod" + self.targetColumn

        for year in self.NormalizedYearlyDF["YearAug"].unique():
            self.NormalizedYearlyDF.loc[self.NormalizedYearlyDF["YearAug"] == year, "Mod"] -= YearlyMeans[year]
