import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm 
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.base import clone

yearly_df = []

raw_data = pd.read_csv("1989 to 2023 CombinedData.csv")
df = raw_data.copy()
df["Month"] = pd.DatetimeIndex(df["datetime"]).month 
df["Year"] = pd.DatetimeIndex(df["datetime"]).year

MODEL_ORDER = 2 
RMSE_DICT = {}


# ====================== DATA LOADER HELPERS ======================
def loadHistoricalData(start_year=2001, end_year=2024):
    data = []

    for year in range(start_year, end_year + 1):
        c_d = os.path.dirname(__file__)
        p_d = os.path.abspath(os.path.join(c_d, os.pardir))
        file_path = os.path.join(p_d, "Data Source Files", f"{year}.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            data.append(df)
        else:
            print(f"File for year {year} does not exist: {file_path}")

    combined_df = pd.concat(data, ignore_index=True)
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
    combined_df['Year'] = combined_df['datetime'].dt.year
    combined_df['Month'] = combined_df['datetime'].dt.month

    return combined_df[["datetime", "temp", "humidity", "precip", "Year", "Month"]]

def loadTestData(year=2025):
    c_d = os.path.dirname(__file__)
    p_d = os.path.abspath(os.path.join(c_d, os.pardir))
    file_path = os.path.join(p_d, "Data Source Files", f"{year}.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['Year'] = df['datetime'].dt.year
        df['Month'] = df['datetime'].dt.month
        return df[["datetime", "temp", "humidity", "precip", "Year", "Month"]]
    else:
        print(f"Test data for year {year} does not exist: {file_path}")
        return None

# 1
def removeYearlyMean(df, col, YEAR_LENGTH = 7):
    MODEL_YEARLY_STD = None 
    YEARLY_SEASONALITY_LENGTH = YEAR_LENGTH

    normalized_yearly_df = df.copy()
    normalized_yearly_df["YearMod"] = normalized_yearly_df[col]

    years = pd.DatetimeIndex(normalized_yearly_df["datetime"]).year.unique().to_list()
    min_val = years[0]
    years_normalized = [(x - min_val) % YEARLY_SEASONALITY_LENGTH for x in years]
    normalized_yearly_df["YearAug"] = (pd.DatetimeIndex(normalized_yearly_df["datetime"]).year - min_val) % YEARLY_SEASONALITY_LENGTH

    yr_grp = normalized_yearly_df.groupby(by = "YearAug").mean(numeric_only = True)[col]

    for year in normalized_yearly_df["YearAug"].unique():
        normalized_yearly_df.loc[normalized_yearly_df["YearAug"] == year, "YearMod"] -= yr_grp[year]
        # normalized_yearly_df.loc[normalized_yearly_df["YearAug"] == year, "YearMod"] /= yr_grp[year]

    yearly_mean = normalized_yearly_df["YearMod"].mean()
    normalized_yearly_df["YearMod"] = normalized_yearly_df["YearMod"] - yearly_mean
    # normalized_yearly_df["YearMod"] = normalized_yearly_df["YearMod"] / yearly_mean

    MODEL_YEARLY_STD = np.std(normalized_yearly_df["YearMod"])

    return MODEL_YEARLY_STD, yearly_mean, min_val, yr_grp, normalized_yearly_df
# 2
def removeMonthlyMean(in_df):
    normalized_df = in_df.copy()
    normalized_df["MonthMod"] = normalized_df["YearMod"]

    month_grp = normalized_df.groupby(by = "Month").mean(numeric_only = True)["MonthMod"]

    for month in normalized_df['Month'].unique():
        # Look for df["Month"] == current unique month, Change TempMod Value to be (currentTemp of the Day) - (Monthly Mean)
        normalized_df.loc[normalized_df['Month'] == month, "MonthMod"] -= month_grp[month]

    monthly_mean = normalized_df['MonthMod'].mean()

    # Subtract the Normalized Temperature Mean by the Total Mean of all the other Normalized
    normalized_df['MonthMod'] = normalized_df['MonthMod'] - monthly_mean

    return normalized_df, month_grp, monthly_mean

# 3.1
def window_data(data):
    n = len(data)
    x = []
    y = []
    for i in range(n - MODEL_ORDER):
        x.append(data[i : i + MODEL_ORDER])
        y.append(data[i + MODEL_ORDER])
    
    return np.array(x), np.array(y)

# 3
def splitData(df, split_percent = 70):
    train_data = df["MonthMod"].copy()
    n = len(train_data)
    split_point = int(split_percent / 100 * n)

    MODEL_SIGMA = np.std(train_data[:split_point])
    window_x, window_y = window_data(train_data)

    train_x = window_x[:split_point]
    test_x = window_x[split_point:]
    train_y = window_y[:split_point]
    test_y = window_y[split_point:]

    return (train_x, test_x, train_y, test_y), MODEL_SIGMA

# 4
def trainModel(predictorClass, train_x, train_y):
    MODEL_PREDICTOR = clone(predictorClass)
    MODEL_PREDICTOR.fit(train_x, train_y)

    return MODEL_PREDICTOR


# 5
def calculatePredictionDate(normalized_df, train_y, YEAR, MONTH, DATE, MODEL_ORDER):
    target_date = datetime(YEAR, MONTH, DATE)
    
    StringFormat = "%Y-%M-%d"
    raw_start_date = normalized_df["datetime"][MODEL_ORDER]
    
    train_start_date = datetime.strptime(raw_start_date, StringFormat)
    train_end_date = train_start_date + timedelta(days = len(train_y)) # + 1 Since Range

    prediction_start_date = train_end_date # -1 + 1
    prediction_end_date = target_date + timedelta(days = 1)

    prediction_date_range = np.arange(prediction_start_date, prediction_end_date, timedelta(days = 1)).astype("datetime64[D]")
    prediction_length = len(prediction_date_range)

    return prediction_date_range, prediction_length

# 6
def calculateTrainingDateRange(df, train_y, MODEL_ORDER):
    StringFormat = "%Y-%M-%d"
    raw_start_date = df["datetime"][MODEL_ORDER]
    train_start_date = datetime.strptime(raw_start_date, StringFormat)
    train_end_date = train_start_date + timedelta(days = len(train_y)) # + 1 Since Range <

    training_date_range = np.arange(train_start_date, train_end_date, timedelta(days = 1)).astype("datetime64[D]")

    return training_date_range


# 7
def doPredictions(train_x, num_predictions, MODEL_ORDER, MONTH_STD, MONTH_STD_SCALE, YEAR_STD, YEAR_STD_SCALE, MODEL_PREDICTOR):
    preds = np.array([])
    r_queue = np.copy(train_x)[-MODEL_ORDER]
    
    for i in tqdm(range(num_predictions)):
        tr = np.reshape(r_queue, (1, -1))
        month_resid = np.random.normal(loc = 0, scale = MONTH_STD * MONTH_STD_SCALE)
        year_resid = np.random.normal(loc = 0, scale = YEAR_STD * YEAR_STD_SCALE)
        next_val = MODEL_PREDICTOR.predict(tr) + month_resid + year_resid 
        # next_val = MODEL_PREDICTOR.predict(tr) + month_resid * year_resid 
        preds = np.append(preds, next_val)
        r_queue = np.roll(r_queue, -1)
        r_queue[-1] = next_val # NOTE : Causes deprecation Warning, but works fine

    return preds 

# 8
def reconstructDF(prediction_date_range, training_date_range, preds, train_y, month_grp, year_grp, year_min_val, monthly_mean, yearly_mean, YEARLY_SEASONALITY_LENGTH):
    full_datetime_range = np.concatenate((training_date_range, prediction_date_range))
    concatenated_preds = np.concatenate((train_y, preds))

    constructed_df = pd.DataFrame({
        "datetime" : full_datetime_range,
        "PredictionFull" : concatenated_preds
    })

    constructed_df["Month"] = pd.DatetimeIndex(constructed_df["datetime"]).month
    constructed_df["Year"] = pd.DatetimeIndex(constructed_df["datetime"]).year 

    constructed_df["YearAug"] = (constructed_df["Year"] - year_min_val) % YEARLY_SEASONALITY_LENGTH

    constructed_df["ReconMonth"] = constructed_df["PredictionFull"]

    for month in constructed_df["Month"].unique():
        constructed_df.loc[constructed_df['Month'] == month, "ReconMonth"] += month_grp[month]

    constructed_df["ReconMonth"] += monthly_mean 

    constructed_df["ReconstructedFinal"] = constructed_df["ReconMonth"]

    for year in constructed_df["YearAug"].unique():
        constructed_df.loc[constructed_df["YearAug"] == year, "ReconstructedFinal"] += year_grp[year]
        # constructed_df.loc[constructed_df["YearAug"] == year, "ReconstructedFinal"] *= year_grp[year]

    # constructed_df["ReconstructedFinal"] *= yearly_mean
    constructed_df["ReconstructedFinal"] += yearly_mean

    return constructed_df

# 9
def evaluatePredictions(finalDF, normalized_df, col, train_percent = 70):
    final_train_idx = int(normalized_df.shape[0] * train_percent / 100)
    final_train_datetime_val = normalized_df.loc[final_train_idx, "datetime"]
    finalDF_train_last_idx = finalDF.loc[finalDF["datetime"] == final_train_datetime_val].index[0] 
    normalizedDF_train_last_idx = normalized_df.loc[normalized_df["datetime"] == final_train_datetime_val].index
    y_preds = finalDF[finalDF_train_last_idx + MODEL_ORDER : finalDF_train_last_idx + normalized_df.shape[0] - normalizedDF_train_last_idx[0]]["ReconstructedFinal"].values
    y_true = normalized_df[normalizedDF_train_last_idx[0] + MODEL_ORDER : normalized_df.shape[0]][col].values

    return root_mean_squared_error(y_true, y_preds)

# Single Pass on DF
def singlePass(col = "temp", myss = 0, ysl = 7, mmss = 1, sp = 70, mo = 2, py = 2050, pm = 12, pd = 31):
    temperature_df = df[["datetime", col, "Month", "Year"]].copy()
    MODEL_YEARLY_STD_SCALE = myss 
    YEARLY_SEASON_LENGTH = ysl
    MODEL_YEARLY_STD, yearly_mean, year_min_val, year_grp, normalized_yearly_df = removeYearlyMean(temperature_df, col, YEARLY_SEASON_LENGTH)
    normalized_df, month_grp, monthly_mean = removeMonthlyMean(normalized_yearly_df)
    MODEL_ORDER = mo
    MODEL_MONTHLY_STD_SCALE = mmss
    SPLIT_POINT = sp
    (train_x, test_x, train_y, test_y), MODEL_MONTHLY_STD = splitData(normalized_df, SPLIT_POINT)
    # PREDICTOR = trainModel(LinearRegression(), train_x, train_y)
    PREDICTOR = trainModel(RandomForestRegressor(), train_x, train_y)
    YEAR = py
    MONTH = pm
    DATE = pd

    prediction_date_range, prediction_length = calculatePredictionDate(normalized_df, train_y, YEAR, MONTH, DATE, MODEL_ORDER)
    training_date_range = calculateTrainingDateRange(normalized_df, train_y, MODEL_ORDER)
    
    predictions = doPredictions(train_x, prediction_length, MODEL_ORDER, MODEL_MONTHLY_STD, MODEL_MONTHLY_STD_SCALE, MODEL_YEARLY_STD, MODEL_YEARLY_STD_SCALE, PREDICTOR)
    finalDF = reconstructDF(prediction_date_range, training_date_range, predictions, train_y, month_grp, year_grp, year_min_val, monthly_mean, yearly_mean, YEARLY_SEASON_LENGTH)
    mae = evaluatePredictions(finalDF, normalized_df, col, SPLIT_POINT)

    return mae, finalDF

def singlePassWithGraph(col, YearScale, MonthScale, ModelOrder, Monthly_View = False):
    MODEL_ORDER = 3
    mae, finalDF = singlePass(myss = YearScale, mmss = MonthScale, col=col, sp = 1, mo = ModelOrder)

    ref_values = df[col].values[ModelOrder:]
    pred_values = finalDF["ReconstructedFinal"].values

    if Monthly_View:
        df["YearMonth"] = pd.DatetimeIndex(df["datetime"]).year * 1000 + pd.DatetimeIndex(df["datetime"]).month
        finalDF["YearMonth"] = pd.DatetimeIndex(finalDF["datetime"]).year * 1000 + pd.DatetimeIndex(finalDF["datetime"]).month

        ref_values = df.groupby(by = "YearMonth").mean(numeric_only = True)[col].values

        pred_values = finalDF.groupby(by = "YearMonth").mean(numeric_only = True)["ReconstructedFinal"].values
        

    plt.figure(figsize = (24, 7))
    if Monthly_View:
        plt.title(col.title() + " Predictions [Monthly Resolution]")
    else:
        plt.title(col.title() + " Predictions [Daily Resolution]")
        
    plt.plot(pred_values, label = "Predictions")
    plt.plot(ref_values, label = "Actual", alpha = 0.5)
    plt.legend()
    plt.show()
    print("RMSE: " ,mae)
    RMSE_DICT[col] = mae

singlePassWithGraph("temp", 0.5, 0.5, 3)
singlePassWithGraph("humidity", 0.2, 0.5, 3)
singlePassWithGraph("precip", 0.2, 0.5, 3)

print(RMSE_DICT)