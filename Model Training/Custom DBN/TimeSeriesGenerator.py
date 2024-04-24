import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def generate_timeseries(length, trend="linear", seasonality=False, noise=0.1):
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
    data.to_csv("timeseries.csv", index=False)

    return timeseries

if __name__ == "__main__":
    data = generate_timeseries(200, trend="sinusoidal", seasonality=True, noise=0.5)
    print(data)
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.show()

