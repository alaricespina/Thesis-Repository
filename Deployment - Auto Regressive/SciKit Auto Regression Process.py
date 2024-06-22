import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression 
from datetime import datetime, timedelta 
from tqdm import tqdm 

class AutoRegressivePredictorModel():
    def __init__(self, model_order, year_std, month_std, year_std_scale, month_std_scale, predictor = LinearRegression() ,percentage = 70):
        self.ORDER = model_order
        self.MONTH_STD = month_std
        self.YEAR_STD = year_std 
        self.MONTH_STD_SCALE = month_std_scale
        self.YEAR_STD_SCALE = year_std_scale 
        self.PERCENT_TRAIN = percentage / 100
        self.PREDICTOR = predictor

    def window_data(self, data):
        n = len(data)
        x = []
        y = []
        for i in range(n - self.ORDER):
            x.append(data[i : i + MODEL_ORDER])
            y.append(data[i + MODEL_ORDER])

        return np.array(x), np.array(y)

    def construct_dataset(self, raw_data):
        self.split_point = int(len(raw_data) * self.PERCENT_TRAIN)
        self.train_x = self.window_data[:self.split_point]
        self.test_x = self.window_data[self.split_point:]
        self.train_y = self.window_data[:self.split_point]
        self.test_y = self.window_data[self.split_point:]

    def get_split_dataset(self):
        return (self.train_x, self.test_x, self.train_y, self.test_y)
    
    def fit_model(self, train_x, train_y):
        self.PREDICTOR.fit(train_x, train_y) 

    def predict(self, num_predict):
        r_queue = np.copy(self.train_x)[- self.ORDER]
        preds = np.array([])

        for i in tqdm(range(num_predictions)):

            tr = np.reshape(r_queue, (1, -1))
            month_resid = np.random.normal(loc = 0, scale = self.MONTH_STD * self.MONTH_STD_SCALE)
            year_resid = np.random.normal(loc = 0, scale = self.YEAR_STD * self.YEAR_STD_SCALE)

            next_val = self.PREDICTOR.predict(tr) + month_resid + year_resid

            preds = np.append(preds, next_val)
            r_queue = np.roll(r_queue, -1)
            r_queue[-1] = next_val


        return preds

class PlotHelpers():
    def plotSeasonality(self, df, col):
        fig, ax = plt.subplots(2, 3, figsize = (24, 14))

        df.boxplot(column = [col], by = "Month", ax = ax[0, 0])
        ax[0, 1].scatter(df["Month"], df[col])
        ax[0, 2].plot(df.groupby(by = "Month").mean(numeric_only = True)[col].values)
        
        
        df.boxplot(column = [col], by = "Year", ax = ax[1, 0])
        ax[1, 1].scatter(df["Year"], df[col])
        ax[1, 2].plot(df.groupby(by = "Year").mean(numeric_only = True)[col].values)
        
        ax[0, 0].set_title("Month - Box Plot")
        ax[0, 1].set_title("Month - Scatter Plot")
        ax[0, 2].set_title("Month - Average")
        ax[1, 0].set_title("Year - Box Plot")
        ax[1, 1].set_title("Year - Scatter Plot")
        ax[1, 2].set_title("Year - Average")
        
        plt.show()

class AutoRegressivePreprocessor():
    def __init__(self, year_cycle = 7, augment = True):
        self.YEAR_CYCLE = year_cycle
        self.AUGMENT = augment 

    # Assume that datetime is in cols
    def removeYearlyMean(raw_data, col):
        if "datetime" not in col:
            raise "No Date time column provided"

        years = pd.DatetimeIndex(raw_data["datetime"]).year.unique().to_list()
        min_val = years[0]
        years_normalized = [(x - min_val) % YEARLY_SEASONALITY_LENGTH for x in years]
        raw_data["YearAug"] = (pd.DatetimeIndex(raw_data["datetime"]).year - min_val) % YEARLY_SEASONALITY_LENGTH
        column_name = "Year" + col
        raw_data[column_name] = raw_data[col]

        yr_grp = raw_data.groupby(by = "YearAug").mean(numeric_only = True)[col]

        for year in raw_data["YearAug"].unique():
            raw_data.loc[raw_data["YearAug"] == year, column_name] -= yr_grp[col]
        
        yearly_mean = raw_data[col].mean()

        raw_data[col] = raw_data[col] - yearly_mean

        modelYearlySTD = np.std(raw_data[col])

        return raw_data, modelYearlySTD

    def removeMonthlyMean(norm_data, col):
        pass
        

    def addYearlyMean():
        pass 

    def addMonthlyMean():
        pass 