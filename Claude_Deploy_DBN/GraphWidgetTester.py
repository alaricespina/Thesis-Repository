import sys
import pandas as pd
import numpy as np
import pyqtgraph as pg
import os 

from pprint import pprint

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets # Use pyqtgraph's Qt bindings for consistency

# --- Qt Imports (PyQt5 specific, but pyqtgraph.Qt often abstracts) ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QCheckBox, QLineEdit, QLabel, QSizePolicy)
from PyQt5.QtCore import Qt, QDateTime, QEvent

from GraphWidgetClass import GraphWidget


# --- Main Application Window (Example Usage) ---
class AppMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Self-Contained GraphWidget Demo")
        self.setGeometry(100, 100, 1000, 750)

        self.central_w = QWidget()
        self.setCentralWidget(self.central_w)
        self.app_layout = QVBoxLayout(self.central_w)

        # 1. Create an instance of the compound GraphWidget
        self.graph_display_widget = GraphWidget(self) # Pass parent if needed
        self.app_layout.addWidget(self.graph_display_widget)

        # 2. (Optional) Add a button in your main app to trigger data loading/plotting
        self.load_data_button = QPushButton("Load/Re-Plot Sample Data")
        self.load_data_button.clicked.connect(self.load_new_data_into_graph)
        # self.load_data_button.clicked.connect(self.loadHistoricalData)
        self.app_layout.addWidget(self.load_data_button)
        
        # self.load_new_data_into_graph() # Load initial data on startup
        self.loadHistoricalData()

    def loadHistoricalData(self):
        all_data = pd.DataFrame()
        for file in os.listdir("Data/Yearly"):
            df = pd.read_csv(os.path.join("Data/Yearly", file))
            all_data = pd.concat([all_data, df], axis=0)

        print("[Start Up] Getting Historical Data")

        # print(all_data.describe())
        # print(all_data.columns)

        self.historical_temperature_data = all_data['temp'].copy()
        self.historical_humidity_data = all_data['humidity'].copy()
        self.historical_pressure_data = all_data['sealevelpressure'].copy()
        self.historical_wind_data = all_data['windspeed'].copy()
        self.raw_df = all_data[["datetime","conditions", "tempmax", "tempmin", "temp", "humidity", "windspeed", "sealevelpressure"]].copy()

        data = {
            'Timestamp': all_data["datetime"].to_numpy(),
            'Temperature': all_data["temp"].to_numpy(),
            'Humidity': all_data["humidity"].to_numpy(),
            'Pressure': all_data["sealevelpressure"].to_numpy(),
            'Wind Speed': all_data["windspeed"].to_numpy(),
            'Max Temp': all_data["tempmax"].to_numpy(),
            'Min Temp': all_data["tempmin"].to_numpy()
        }

        print("[Finish] Historical Data Loaded")
        
        return pd.DataFrame(data)

    def _generate_sample_df(self, LENGTH_OF_DATA = 150):
        """Helper to generate sample DataFrame."""
        rng = pd.date_range('2024-01-15 00:00:00', periods=LENGTH_OF_DATA, freq='2h') # 2-hourly data
        data = {
            'Timestamp': rng,
            'MetricA': np.cumsum(np.random.randn(LENGTH_OF_DATA) * 0.5) + 50,
            'MetricB': np.sin(np.linspace(0, 8 * np.pi, LENGTH_OF_DATA)) * 20 + 70,
            'MetricC': np.random.rand(LENGTH_OF_DATA) * 10 + 30,
            'MetricD': np.cumsum(np.random.randn(LENGTH_OF_DATA) * 0.5) + 50,
            'MetricE': np.sin(np.linspace(0, 8 * np.pi, LENGTH_OF_DATA)) * 20 + 70,
            'MetricF': np.random.rand(LENGTH_OF_DATA) * 10 + 30
        }
        return pd.DataFrame(data)

    def load_new_data_into_graph(self):
        """Prepares data and plots it using the GraphWidget."""
        # sample_df = self._generate_sample_df(1_000)
        sample_df = self.loadHistoricalData()
        
        cols = sample_df.columns.to_list()

        plot_pens = [
            pg.mkPen(color=(50, 200, 255), width=2),   # Light Blue
            pg.mkPen(color=(255, 180, 50), width=2),   # Orange
            pg.mkPen(color=(150, 255, 150), width=2, style=Qt.DotLine), # Light Green Dotted
            pg.mkPen(color=(150, 200, 255), width=2),   # Light Blue
            pg.mkPen(color=(55, 180, 50), width=2),   # Orange
            pg.mkPen(color=(150, 55, 150), width=2, style=Qt.DotLine) # Light Green Dotted
        ]
        
        # columns_to_plot = ['MetricA', 'MetricB', 'MetricC']
        cols.remove('Timestamp')
        cols.remove('Pressure')
        columns_to_plot = cols 
        
        # Call the public method of GraphWidget to plot
        self.graph_display_widget.plot_dataframe(
            df=sample_df,
            x_col_datetime='Timestamp',
            y_cols_primary=columns_to_plot,
            y_cols_secondary=['Pressure'],
            title="Hourly Sensor Metrics",
            pens_primary=plot_pens,
            pens_secondary=plot_pens
        )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = AppMainWindow()
    main_window.show()
    sys.exit(app.exec_())