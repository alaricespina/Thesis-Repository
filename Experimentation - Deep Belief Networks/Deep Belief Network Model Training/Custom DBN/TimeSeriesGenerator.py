import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def generate_timeseries(length, filename, trend="linear", seasonality=False, noise=0.1):
    # Create base trend
    if trend == "linear":
        base_trend = np.linspace(0, 1, length)

    elif trend == "sinusoidal":
        base_trend = np.sin(np.linspace(0, 2*np.pi, length))

    else:
        base_trend = np.zeros(length)

    # Add seasonality (optional)
    if seasonality:
        seasonality_component = np.sin(np.linspace(0, 2*np.pi, length) * 8) * 0.2
        timeseries = base_trend + seasonality_component
    else:
        timeseries = base_trend

    # Add noise
    timeseries += np.random.normal(0, noise, length)

    # Create DataFrame and save
    data = pd.DataFrame({"value": timeseries})
    data.to_csv(filename, index=False)

    return timeseries

if __name__ == "__main__":
    # NN - No Noise, MXN - Max Noise, MNN - Min Noise, MDN - Medium Noise
    # NS - No Seasonality, WS - With Seasonality
    # S - Sinusoidal, L - Linear
    
    # data = generate_timeseries(1_000, "NNNS1K_TS.csv", trend="sinusoidal", seasonality=False, noise=0.0)
    # print(data)
    # plt.figure(figsize=(10, 5))
    # plt.plot(data)
    # plt.show()

    timeSeriesLength = 1_000
    # Very Clean Data
    generate_timeseries(timeSeriesLength, "NNNSS.csv", trend="sinusoidal", seasonality=False, noise=0.0)

    # Very Noisy Data
    generate_timeseries(timeSeriesLength, "MXNWSS.csv", trend="sinusoidal", seasonality=True, noise=0.9)
    

