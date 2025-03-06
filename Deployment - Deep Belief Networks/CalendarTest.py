import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import pandas as pd
import calendar
from datetime import date

class CalendarWidget(QtWidgets.QWidget):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.df = df
        self.df["datetime"] = pd.to_datetime(self.df["datetime"])
        self.df["year"] = self.df["datetime"].dt.year
        self.df["conditions"] = self.df["conditions"].replace(
            {"Overcast" : "Cloudy", 
            "Rain, Overcast" : "Windy", 
            "Partially cloudy" : "Sunny", 
            "Rain, Partially cloudy" : "Rainy"})
    
        self.available_years = sorted(df["year"].unique().tolist())  # Get available years from DataFrame

        self.main_layout = QtWidgets.QHBoxLayout(self)  # Main layout is now horizontal

        # Right Panel for Weather Icons and Information (Swapped)
        self.right_panel = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.right_panel)

        # Weather Icons Grid
        self.weather_icon_grid = QtWidgets.QGridLayout()
        self.right_panel.addLayout(self.weather_icon_grid)

        self.cloudy_active = False
        self.windy_active = False
        self.sunny_active = False
        self.rainy_active = False

        # Create weather icons (replace with actual icon paths)
        self.sunny_icon_active = self.create_icon_label("WeatherIcons/SUNNY ACTIVE.png", "Sunny")
        self.cloudy_icon_active = self.create_icon_label("WeatherIcons/CLOUDY ACTIVE.png", "Cloudy")
        self.rainy_icon_active = self.create_icon_label("WeatherIcons/RAINY ACTIVE.png", "Rainy")
        self.windy_icon_active = self.create_icon_label("WeatherIcons/WINDY ACTIVE.png", "Windy")

        self.sunny_icon_inactive = self.create_icon_label("WeatherIcons/SUNNY INACTIVE.png", "Sunny")
        self.cloudy_icon_inactive = self.create_icon_label("WeatherIcons/CLOUDY INACTIVE.png", "Cloudy")
        self.rainy_icon_inactive = self.create_icon_label("WeatherIcons/RAINY INACTIVE.png", "Rainy")
        self.windy_icon_inactive = self.create_icon_label("WeatherIcons/WINDY INACTIVE.png", "Windy")

        # Add icons to the grid
        self.weather_icon_grid.addWidget(self.sunny_icon_active, 0, 0)
        self.weather_icon_grid.addWidget(self.cloudy_icon_active, 0, 1)
        self.weather_icon_grid.addWidget(self.rainy_icon_active, 1, 0)
        self.weather_icon_grid.addWidget(self.windy_icon_active, 1, 1)

        self.weather_icon_grid.addWidget(self.sunny_icon_inactive, 0, 0)
        self.weather_icon_grid.addWidget(self.cloudy_icon_inactive, 0, 1)
        self.weather_icon_grid.addWidget(self.rainy_icon_inactive, 1, 0)
        self.weather_icon_grid.addWidget(self.windy_icon_inactive, 1, 1)

        # Information Label
        self.info_label = QtWidgets.QLabel("Select a date to see weather information.")
        font = QtGui.QFont()
        font.setPointSize(6)

        self.info_label.setWordWrap(True)  # Allow text to wrap
        self.info_label.setFont(font)
        self.right_panel.addWidget(self.info_label)

        # Calendar Widget (Left Panel) - Swapped
        self.calendar_widget = QtWidgets.QWidget()
        self.calendar_layout = QtWidgets.QVBoxLayout(self.calendar_widget)
        self.main_layout.addWidget(self.calendar_widget)

        # Year Selection
        self.year_combo = QtWidgets.QComboBox()
        self.year_combo.addItems([str(year) for year in self.available_years])
        self.year_combo.currentIndexChanged.connect(self.year_changed)
        self.calendar_layout.addWidget(self.year_combo)

        # Month Selection
        self.month_layout = QtWidgets.QHBoxLayout()
        self.month_buttons = []
        for month in range(1, 13):
            month_button = QtWidgets.QPushButton(calendar.month_abbr[month])  # Use abbreviated month names
            month_button.setCheckable(True)  # Make the buttons checkable
            month_button.clicked.connect(lambda checked, m=month: self.month_clicked(m, checked))  # Pass month number
            self.month_layout.addWidget(month_button)
            self.month_buttons.append(month_button)
        self.calendar_layout.addLayout(self.month_layout)

        self.month_label = QtWidgets.QLabel("")
        self.month_label.setAlignment(QtCore.Qt.AlignCenter)
        self.calendar_layout.addWidget(self.month_label)

        self.grid_layout = QtWidgets.QGridLayout()
        self.grid_layout.setSpacing(0)
        self.calendar_layout.addLayout(self.grid_layout)

        days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        for col, day in enumerate(days):
            header_label = QtWidgets.QLabel(day)
            header_label.setAlignment(QtCore.Qt.AlignCenter)
            header_label.setStyleSheet("background-color: lightgray; border: 1px solid gray;")
            self.grid_layout.addWidget(header_label, 0, col)

        self.cell_buttons = []
        for row in range(6):
            row_buttons = []
            for col in range(7):
                cell_button = QtWidgets.QPushButton("")
                cell_button.setStyleSheet("border: 1px solid gray; padding: 5px;")
                cell_button.clicked.connect(self.cell_clicked)
                self.grid_layout.addWidget(cell_button, row + 1, col)
                row_buttons.append(cell_button)
            self.cell_buttons.append(row_buttons)

        self.current_year = self.available_years[0] if self.available_years else 2023  # Default to first available year or 2023
        self.current_month = 1
        self.fill_calendar(self.current_year, self.current_month)
        self.selected_date = None

    def create_icon_label(self, icon_path, weather_type):
        """Creates a QLabel with an icon."""
        label = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(icon_path)
        label.setPixmap(pixmap.scaled(50, 50, QtCore.Qt.KeepAspectRatio))  # Scale the image
        label.setToolTip(weather_type)
        return label

    def update_icon_states(self):
        """Updates the visibility of the weather icons based on their active state."""
        self.set_icon_visibility(self.sunny_icon_active, self.sunny_active)
        self.set_icon_visibility(self.cloudy_icon_active, self.cloudy_active)
        self.set_icon_visibility(self.rainy_icon_active, self.rainy_active)
        self.set_icon_visibility(self.windy_icon_active, self.windy_active)

        self.set_icon_visibility(self.sunny_icon_inactive, not self.sunny_active)
        self.set_icon_visibility(self.cloudy_icon_inactive, not self.cloudy_active)
        self.set_icon_visibility(self.rainy_icon_inactive, not self.rainy_active)
        self.set_icon_visibility(self.windy_icon_inactive, not self.windy_active)

    def set_icon_visibility(self, label, visible):
        """Sets the visibility of the icon label."""
        label.setVisible(visible)

    def year_changed(self, index):
        """Called when the year is changed in the combo box."""
        self.current_year = self.available_years[index]
        self.fill_calendar(self.current_year, self.current_month)

    def month_clicked(self, month, checked):
        """Called when a month button is clicked."""
        if checked:
            # Uncheck other month buttons
            for i, button in enumerate(self.month_buttons):
                if i + 1 != month:
                    button.setChecked(False)
            self.current_month = month
            self.fill_calendar(self.current_year, self.current_month)
        else:
            # If unchecking, revert to showing all months or do nothing
            pass  # Or implement a "no month selected" state

    def fill_calendar(self, year, month):
        """Fills the calendar with days for the given year and month."""
        self.month_label.setText(calendar.month_name[month] + " " + str(year))

        cal = calendar.Calendar()
        month_days = cal.monthdayscalendar(year, month)

        # Clear previous calendar
        for row_buttons in self.cell_buttons:
            for cell_button in row_buttons:
                cell_button.setText("")
                cell_button.setProperty("date", None)
                cell_button.setStyleSheet("border: 1px solid gray; padding: 5px;")

        # print(self.df["datetime"].tolist())
        for row, week in enumerate(month_days):
            for col, day in enumerate(week):
                cell_button = self.cell_buttons[row][col]
                if day != 0:
                    date = pd.to_datetime(f"{year}-{month}-{day}")
                    # print([date], date in self.df["datetime"].tolist())
                    cell_button.setText(str(day))
                    cell_button.setProperty("date", date)
                    self.apply_weather_color(cell_button, date)                    
                else:
                    cell_button.setText("")
                    cell_button.setProperty("date", None)

    def apply_weather_color(self, cell_button, date):
        weather_colors = {
            "Sunny": "lightyellow",
            "Rainy": "lightblue",
            "Cloudy": "lightgray",
            "Windy": "white"
        }

        if date in self.df["datetime"].tolist():
            weather_condition = self.df.loc[self.df["datetime"] == date, "conditions"].iloc[0]
            if weather_condition in weather_colors:
                cell_button.setStyleSheet(f"border: 1px solid gray; padding: 5px; background-color: {weather_colors[weather_condition]};")
            else:
                cell_button.setStyleSheet("border: 1px solid gray; padding: 5px;")
        else:
            cell_button.setStyleSheet("border: 1px solid gray; padding: 5px;")

    def cell_clicked(self):
        button = self.sender()
        date = button.property("date")
        if date:
            self.selected_date = date
            if date in self.df["datetime"].tolist():
                condition = self.df.loc[self.df["datetime"] == date, ["conditions", "tempmax", "tempmin", "temp", "humidity", "windspeed", "sealevelpressure"]].iloc[0]
                weather = condition["conditions"]
                tempmax = condition["tempmax"]
                tempmin = condition["tempmin"]
                temp = condition["temp"]
                humidity = condition["humidity"]
                windspeed = condition["windspeed"]
                sealevelpressure = condition["sealevelpressure"]
                if weather == "Windy":
                    self.windy_active = True
                    self.rainy_active = False
                    self.sunny_active = False
                    self.cloudy_active = False
                elif weather == "Rainy":
                    self.windy_active = False
                    self.rainy_active = True
                    self.sunny_active = False
                    self.cloudy_active = False
                elif weather == "Sunny":
                    self.windy_active = False
                    self.rainy_active = False
                    self.sunny_active = True
                    self.cloudy_active = False
                elif weather == "Cloudy":
                    self.windy_active = False
                    self.rainy_active = False
                    self.sunny_active = False
                    self.cloudy_active = True

                # Update the icon states
                self.update_icon_states()

                # Update the information label
                info_text = (
                    f"Date: {date.strftime('%Y-%m-%d')}\n"
                    f"Weather: {weather}\n"
                    f"Max Temp: {tempmax}\n"
                    f"Min Temp: {tempmin}\n"
                    f"Avg Temp: {temp}\n"
                    f"Humidity: {humidity}\n"
                    f"Wind Speed: {windspeed}\n"
                    f"Pressure: {sealevelpressure}"
                )
                self.info_label.setText(info_text)
            else:
                self.info_label.setText(f"Date: {date.strftime('%Y-%m-%d')}, No data available.")
                # Reset icon states when no data is available
                self.windy_active = False
                self.rainy_active = False
                self.sunny_active = False
                self.cloudy_active = False
                self.update_icon_states()
        else:
            self.info_label.setText("No date selected.")
            # Reset icon states when no date is selected
            self.windy_active = False
            self.rainy_active = False
            self.sunny_active = False
            self.cloudy_active = False
            self.update_icon_states()

# Example DataFrame
# data = {
#     "weather_condition": ["Sunny", "Rainy", "Cloudy", "Sunny", "Snowy", "Rainy", "Sunny", "Cloudy"],
# }
# dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-15", "2023-01-20", "2023-01-25", "2024-01-01", "2024-02-10"])
# df = pd.DataFrame(data, index=dates)

if __name__ == "__main__":
    df = pd.read_csv("Data/Concatenated Data.csv")
    print(df)
    app = QtWidgets.QApplication(sys.argv)
    window = CalendarWidget(df)
    window.show()
    sys.exit(app.exec_())