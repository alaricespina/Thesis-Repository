import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm 
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.base import clone


# ====================== PLOTTING HELPERS ======================
def plotSeasonality(df, col):
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

def plotSeasonalDecomposition(decompose_result):
    fig, ax = plt.subplots(2, 2, figsize = (24, 14))
    observed_series = decompose_result.observed.dropna() 
    trend_series = decompose_result.trend.dropna()
    seasonal_series = decompose_result.seasonal.dropna()
    resid_series = decompose_result.resid.dropna()

    ax[0, 0].plot(observed_series)
    ax[0, 0].set_title("Observed")
    ax[0, 1].plot(trend_series)
    ax[0, 1].set_title("Trend")
    ax[1, 0].plot(seasonal_series)
    ax[1, 0].set_title("Seasonal")
    ax[1, 1].plot(resid_series)
    ax[1, 1].set_title("Residual")

    plt.show()

def multiPeriodSeasonalDecomposition(df, decomposition_mode):
    cols = 5
    rows = df.shape[0] // 2 // cols + 1

    # Trend
    fig, trend_ax = plt.subplots(rows, cols, figsize = (24, rows * 3))
    fig, seasonal_ax = plt.subplots(rows, cols, figsize = (24, rows * 3))
    fig, resid_ax = plt.subplots(rows, cols, figsize = (24, rows * 3))

    lowest_val = min(yearly_df["means"]) - 1
    highest_val = min(yearly_df["means"]) + 2

    for i in range(1, yr.shape[0] // 2 + 1):
        try:
            ysd = seasonal_decompose(yearly_df["means"], model = decomposition_mode, period = i)
            x = i % cols 
            y = i // cols 

            seasonal_ax[y, x].plot(ysd.seasonal)
            seasonal_ax[y, x].set_title(i)

            trend_ax[y, x].plot(ysd.trend)
            trend_ax[y, x].set_title(i)
            trend_ax[y, x].set_ylim([lowest_val, highest_val])

            resid_ax[y, x].plot(ysd.resid)
            resid_ax[y, x].hlines(ysd.resid.mean(), ysd.resid.index[0], ysd.resid.index[-1], color = "red", alpha = 0.3)
            resid_ax[y, x].set_title(i)

        except Exception as E:
            print("Error at i:", i, "Error:", E)

    # clear_output()
    plt.show()

def plotPredsTest(preds, test_y):
    plt.figure(figsize = (24, 7))
    plt.scatter(np.arange(0, len(preds)), preds, alpha = 0.5, label = "Predictions")
    plt.scatter(np.arange(0, len(test_y)), test_y, label="Actual")
    plt.plot(preds, alpha = 0.5, label = "Prediction Lines")
    plt.plot(test_y, label = "Actual Lines")
    plt.legend()
    plt.show()
