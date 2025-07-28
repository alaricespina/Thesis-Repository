import sys
import pandas as pd
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QCheckBox, QLineEdit, QLabel, QSizePolicy)
from PyQt5.QtCore import Qt, QDateTime, QEvent

pg.setConfigOptions(antialias=True)

class GraphWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0,0,0,0)

        self.controls_widget = QWidget()
        self.controls_layout = QHBoxLayout(self.controls_widget)
        self.controls_layout.setContentsMargins(5,5,5,5)
        self.goto_label = QLabel("Go to (YYYY-MM-DD HH:MM:SS):")
        self.controls_layout.addWidget(self.goto_label)
        self.goto_x_input = QLineEdit()
        self.goto_x_input.setPlaceholderText(QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"))
        self.controls_layout.addWidget(self.goto_x_input)
        self.goto_x_button = QPushButton("Go")
        self.goto_x_button.clicked.connect(self._handle_go_to_x_internal)
        self.controls_layout.addWidget(self.goto_x_button)
        self.controls_layout.addStretch(1)
        self.line_toggles_container_widget = QWidget()
        self.line_toggles_layout = QHBoxLayout(self.line_toggles_container_widget)
        self.line_toggles_layout.setContentsMargins(0,0,0,0)
        self.line_toggle_buttons = {}
        self.controls_layout.addWidget(self.line_toggles_container_widget)
        self.main_layout.addWidget(self.controls_widget)

        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.getPlotItem()
        self.primary_view_box = self.plot_item.getViewBox()
        self.main_layout.addWidget(self.plot_widget)

        self.secondary_view_box = pg.ViewBox()
        self.secondary_view_box.setMouseEnabled(y=False)
        self.right_axis = pg.AxisItem(orientation='right')
        self.right_axis.linkToView(self.secondary_view_box)
        self.plot_item.layout.addItem(self.right_axis, 2, 3)
        self.plot_item.scene().addItem(self.secondary_view_box)
        self.primary_view_box.sigXRangeChanged.connect(self._update_secondary_vb_x_range)
        self.secondary_view_box.sigXRangeChanged.connect(self._update_primary_vb_x_range)
        self.primary_view_box.sigResized.connect(self._update_secondary_vb_geometry)

        self.primary_view_box.setBackgroundColor((30, 31, 34))
        self.secondary_view_box.setBackgroundColor(None) # Transparent
        axis_pen = pg.mkPen(color=(180, 180, 180), width=1)
        axis_text_pen = pg.mkPen(color=(210, 210, 210))
        self.date_axis = pg.DateAxisItem(orientation='bottom', pen=axis_pen, textPen=axis_text_pen)
        self.plot_item.setAxisItems({'bottom': self.date_axis})
        left_axis = self.plot_item.getAxis('left')
        left_axis.setPen(axis_pen)
        left_axis.setTextPen(axis_text_pen)
        self.right_axis.setPen(axis_pen)
        self.right_axis.setTextPen(axis_text_pen)
        self.date_axis.setGrid(60)
        left_axis.setGrid(60)

        self.primary_view_box.setMouseEnabled(y=False)
        self.primary_view_box.setMouseMode(pg.ViewBox.PanMode)
        self.primary_view_box.installEventFilter(self)

        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=(255,255,100,150), width=1, style=Qt.DashLine))
        self.plot_item.addItem(self.v_line, ignoreBounds=True)
        self.v_line.hide()
        self.hover_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 150))
        self.plot_item.addItem(self.hover_scatter)
        self.hover_scatter.hide()
        self.hover_text = pg.TextItem(anchor=(0, 1), color=(220, 220, 220), fill=(0, 0, 0, 180), border=pg.mkPen(color=(100,100,100), width=1))
        self.plot_item.addItem(self.hover_text)
        self.hover_text.hide()
        self.hover_text.setZValue(100)
        self.plotted_curves = {}
        self.plot_widget.scene().sigMouseMoved.connect(self._mouse_moved_internal)
        self.plot_widget.installEventFilter(self)

        self.primary_view_box.setZValue(0)
        self.secondary_view_box.setZValue(10)
        self.legend = None # Initialize legend attribute

    def _update_secondary_vb_x_range(self):
        primary_x_range = self.primary_view_box.viewRange()[0]
        self.secondary_view_box.setXRange(primary_x_range[0], primary_x_range[1], padding=0, update=False)

    def _update_primary_vb_x_range(self):
        secondary_x_range = self.secondary_view_box.viewRange()[0]
        self.primary_view_box.setXRange(secondary_x_range[0], secondary_x_range[1], padding=0, update=False)

    def _update_secondary_vb_geometry(self):
        self.secondary_view_box.setGeometry(self.primary_view_box.sceneBoundingRect())

    def eventFilter(self, watched_object, event):
        if watched_object is self.plot_widget:
            if event.type() == QEvent.Leave:
                self._mouse_left_plot_area_internal()
                return True
        elif watched_object is self.primary_view_box:
            if event.type() == QEvent.Wheel:
                event.accept()
                return True
        return super().eventFilter(watched_object, event)

    def _mouse_left_plot_area_internal(self):
        self.v_line.hide()
        self.hover_scatter.hide()
        self.hover_text.hide()

    def _mouse_moved_internal(self, pos):
        if not self.plot_item.sceneBoundingRect().contains(pos) or not self.plotted_curves:
            self._mouse_left_plot_area_internal()
            return

        mouse_point_primary_vb = self.primary_view_box.mapSceneToView(pos)
        mouse_x = mouse_point_primary_vb.x()

        in_data_range = False
        for curve_name, data in self.plotted_curves.items():
            if data['visible'] and len(data['x_data']) > 0 and \
               data['x_data'].min() <= mouse_x <= data['x_data'].max():
                in_data_range = True
                break

        if not in_data_range:
            self._mouse_left_plot_area_internal()
            return

        self.v_line.setPos(mouse_x)
        self.v_line.show()

        hover_points_data = []
        hover_info_html = f"<div style='font-family: Arial, sans-serif; font-size: 9pt; background-color:rgba(0,0,0,0.75); color:white; padding:5px; border-radius:3px;'>"
        dt_object = QDateTime.fromSecsSinceEpoch(int(mouse_x))
        hover_info_html += f"<b>Time:</b> {dt_object.toString('yyyy-MM-dd HH:mm:ss')}</b><br/>"
        hover_info_html += "<hr style='border-color: #555; margin-top: 3px; margin-bottom: 3px;'>"

        any_valid_y = False
        for curve_name, data in self.plotted_curves.items():
            if not data['visible']:
                continue

            y_val = np.nan
            if len(data['x_data']) > 0:
                idx = np.searchsorted(data['x_data'], mouse_x)
                if 0 < idx < len(data['x_data']):
                     y_val = np.interp(mouse_x, data['x_data'], data['y_data'])
                     any_valid_y = True
                elif idx == 0 and mouse_x == data['x_data'][0]:
                    y_val = data['y_data'][0]
                    any_valid_y = True
                elif idx == len(data['x_data']) and mouse_x == data['x_data'][-1]:
                    y_val = data['y_data'][-1]
                    any_valid_y = True


                if not np.isnan(y_val):
                    scatter_y_on_primary_scale = y_val
                    hover_points_data.append({'pos': (mouse_x, scatter_y_on_primary_scale),
                                              'brush': data['curve'].opts['pen'].color()})

            if not np.isnan(y_val):
                hover_info_html += f"<span style='color:{data['pen'].color().name()};'>■</span> {curve_name}: {y_val:.2f}<br/>"
            else:
                hover_info_html += f"<span style='color:grey;'>■</span> {curve_name}: N/A<br/>"
        hover_info_html += "</div>"

        if any_valid_y:
            self.hover_scatter.setData(hover_points_data)
            self.hover_scatter.show()
            self.hover_text.setHtml(hover_info_html)
            self.hover_text.setPos(mouse_point_primary_vb.x(), mouse_point_primary_vb.y())
            self.hover_text.show()
        else:
            self._mouse_left_plot_area_internal()


    def plot_dataframe(self, df, x_col_datetime, y_cols_primary, y_cols_secondary=None,
                       title="DataFrame Plot", pens_primary=None, pens_secondary=None):

        for curve_name in list(self.plotted_curves.keys()):
            curve_item = self.plotted_curves[curve_name]['curve']
            view_box = curve_item.getViewBox()
            if view_box:
                view_box.removeItem(curve_item)
            elif curve_item in self.plot_item.items:
                 self.plot_item.removeItem(curve_item)
        self.plotted_curves.clear()

        # Clear existing legend thoroughly
        if self.plot_item.legend is not None:
            if self.plot_item.legend.scene():
                self.plot_item.legend.scene().removeItem(self.plot_item.legend)
            self.plot_item.legend = None
        self.legend = None # Clear our class member reference as well


        self._update_secondary_vb_geometry()

        if df is None or df.empty or x_col_datetime not in df.columns:
            self.plot_item.setTitle("No data or invalid X column", color=(230,230,230))
            self.date_axis.setLabel("")
            self.plot_item.getAxis('left').setLabel("")
            self.right_axis.setLabel("")
            self._mouse_left_plot_area_internal()
            self._update_line_toggle_buttons_internal([])
            return

        if not pd.api.types.is_datetime64_any_dtype(df[x_col_datetime]):
            try:
                df[x_col_datetime] = pd.to_datetime(df[x_col_datetime])
            except Exception as e:
                self.plot_item.setTitle(f"Error converting '{x_col_datetime}' to datetime: {e}", color=(230,230,230))
                self._mouse_left_plot_area_internal()
                self._update_line_toggle_buttons_internal([])
                return

        df_sorted = df.sort_values(by=x_col_datetime)
        x_data_timestamps = df_sorted[x_col_datetime].astype(np.int64) // 10**9

        def plot_on_viewbox_helper(y_cols_list, pens_list, target_view_box,
                                   existing_y_labels_for_legend_ref, existing_y_cols_plotted_ref):
            if not y_cols_list: return
            if isinstance(y_cols_list, str): y_cols_list = [y_cols_list]

            num_cols = len(y_cols_list)
            if pens_list is None:
                default_colors_set = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                if target_view_box == self.secondary_view_box:
                    default_colors_set = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                pens_list = [pg.mkPen(color=default_colors_set[i % len(default_colors_set)], width=2) for i in range(num_cols)]
            elif not isinstance(pens_list, list) or len(pens_list) < num_cols:
                if not isinstance(pens_list, list): pens_list = [pens_list]
                pens_list = [pg.mkPen(pens_list[i % len(pens_list)]) if not isinstance(pens_list[i % len(pens_list)], pg.Qt.QtGui.QPen) else pens_list[i % len(pens_list)] for i in range(num_cols)]


            for i, y_col_name in enumerate(y_cols_list):
                if y_col_name not in df_sorted.columns or not pd.api.types.is_numeric_dtype(df_sorted[y_col_name]):
                    print(f"Warning: Y-column '{y_col_name}' not found or not numeric. Skipping.")
                    continue

                y_data = df_sorted[y_col_name].values
                existing_y_labels_for_legend_ref.append(y_col_name)
                current_pen = pens_list[i % len(pens_list)]

                curve = pg.PlotDataItem(x_data_timestamps.values, y_data, pen=current_pen, name=y_col_name)
                target_view_box.addItem(curve)

                self.plotted_curves[y_col_name] = {
                    'curve': curve, 'x_data': x_data_timestamps.values, 'y_data': y_data,
                    'visible': True, 'pen': current_pen, 'view_box': target_view_box
                }
                existing_y_cols_plotted_ref.append(y_col_name)

        all_y_cols_plotted = []
        all_y_labels_for_legend = []
        plot_on_viewbox_helper(y_cols_primary, pens_primary, self.primary_view_box,
                               all_y_labels_for_legend, all_y_cols_plotted)
        if y_cols_secondary:
            plot_on_viewbox_helper(y_cols_secondary, pens_secondary, self.secondary_view_box,
                                   all_y_labels_for_legend, all_y_cols_plotted)

        # --- 4. Legend Handling (MODIFIED SECTION) ---
        if all_y_cols_plotted:
            # Create a new legend. self.plot_item.legend will be set by addLegend.
            self.legend = self.plot_item.addLegend(offset=(-10, 10))
            if self.legend: # Ensure legend was created
                self.legend.setBrush(pg.mkBrush(40, 40, 45, 180))
                self.legend.setPen(pg.mkPen(80, 80, 80))

                # Manually add all plotted curves to the legend
                # Iterate in the order they were specified (primary then secondary) for consistent legend order
                ordered_curve_names = (y_cols_primary or []) + (y_cols_secondary or [])
                
                for curve_name in ordered_curve_names:
                    if curve_name in self.plotted_curves:
                        data = self.plotted_curves[curve_name]
                        if data['curve'] is not None:
                             # Check if already added (e.g. if addLegend was unexpectedly smart for primary items)
                            already_present = False
                            for item_in_legend, _ in self.legend.items:
                                if item_in_legend.item is data['curve']:
                                    already_present = True
                                    break
                            if not already_present:
                                self.legend.addItem(data['curve'], name=curve_name)

                # Style all legend items
                for sample, label in self.legend.items: # sample is ItemSample, label is TextItem
                    actual_item = sample.item # This is the PlotDataItem
                    curve_name_for_style = actual_item.opts.get('name')
                    if curve_name_for_style and curve_name_for_style in self.plotted_curves:
                        is_visible = self.plotted_curves[curve_name_for_style]['visible']
                        text_color = pg.mkColor(200,200,200) if is_visible else pg.mkColor(100,100,100)
                        label.setText(curve_name_for_style, color=text_color)
                    else:
                        label.setText(label.text, color=pg.mkColor(200,200,200))
        # --- End of Legend Handling ---


        self.plot_item.setTitle(title, color=(230,230,230), size='11pt')
        self.date_axis.setLabel(text=x_col_datetime, units=None)

        primary_y_names_plotted = [name for name in (y_cols_primary or []) if name in all_y_cols_plotted]
        secondary_y_names_plotted = [name for name in (y_cols_secondary or []) if name in all_y_cols_plotted]

        self.plot_item.getAxis('left').setLabel(", ".join(primary_y_names_plotted) if primary_y_names_plotted else "Primary",
                                               color=(210,210,210))
        self.right_axis.setLabel(", ".join(secondary_y_names_plotted) if secondary_y_names_plotted else "Secondary",
                                 color=(210,210,210))

        self.plot_item.showGrid(x=True, y=True, alpha=0.15)
        self.primary_view_box.autoRange(padding=0.05)
        self.secondary_view_box.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.secondary_view_box.updateAutoRange()

        current_y_range_sec = self.secondary_view_box.viewRange()[1]
        padding_y_sec = (current_y_range_sec[1] - current_y_range_sec[0]) * 0.1
        if padding_y_sec == 0 and (current_y_range_sec[1] - current_y_range_sec[0] == 0) : padding_y_sec = 1
        elif padding_y_sec < 1e-9 and padding_y_sec !=0 : pass
        elif padding_y_sec == 0 : padding_y_sec = 0.1

        self.secondary_view_box.setYRange(current_y_range_sec[0] - padding_y_sec,
                                        current_y_range_sec[1] + padding_y_sec,
                                        padding=0)


        self._mouse_left_plot_area_internal()
        self._update_line_toggle_buttons_internal(all_y_cols_plotted)


    def _handle_go_to_x_internal(self):
        datetime_str = self.goto_x_input.text()
        try:
            dt_object = QDateTime.fromString(datetime_str, "yyyy-MM-dd HH:mm:ss")
            if not dt_object.isValid():
                dt_object = QDateTime.fromString(datetime_str, "yyyy-MM-dd")
                if dt_object.isValid():
                    dt_object.setTime(QtCore.QTime(0,0,0))
                else:
                    print(f"Invalid datetime format: {datetime_str}")
                    self.goto_x_input.setText("Invalid Format!")
                    return
            timestamp = dt_object.toSecsSinceEpoch()
            self._go_to_x_plot_internal(timestamp)
        except Exception as e:
            print(f"Error parsing datetime for 'Go to X': {e}")
            self.goto_x_input.setText("Error!")

    def _go_to_x_plot_internal(self, x_timestamp):
        if not self.plotted_curves:
            return
        current_x_range = self.primary_view_box.viewRange()[0]
        range_width = current_x_range[1] - current_x_range[0]
        if range_width <= 0: range_width = 3600 * 24

        new_x_min = x_timestamp - range_width / 2
        new_x_max = x_timestamp + range_width / 2

        self.primary_view_box.setXRange(new_x_min, new_x_max, padding=0)

    def _toggle_line_visibility_internal(self, line_name, visible):
        if line_name in self.plotted_curves:
            curve_data = self.plotted_curves[line_name]
            if visible:
                if not curve_data['visible']:
                    curve_data['curve'].show()
            else:
                curve_data['curve'].hide()
            curve_data['visible'] = visible

            if self.legend:
                for sample, label_item in self.legend.items:
                    if sample.item.opts.get('name') == line_name:
                        text_color = pg.mkColor(200,200,200) if visible else pg.mkColor(100,100,100)
                        label_item.setText(line_name, color=text_color)
                        break
            self._mouse_left_plot_area_internal()

    def _update_line_toggle_buttons_internal(self, line_names):
        for i in reversed(range(self.line_toggles_layout.count())):
            widget_to_remove = self.line_toggles_layout.itemAt(i).widget()
            if widget_to_remove:
                widget_to_remove.setParent(None)
                widget_to_remove.deleteLater()
        self.line_toggle_buttons.clear()
        if not line_names: return

        toggle_label = QLabel("Toggle Lines:")
        toggle_label.setStyleSheet("QLabel { color: rgb(210,210,210); }")
        self.line_toggles_layout.addWidget(toggle_label)

        for name in line_names:
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, line_name=name:
                                          self._toggle_line_visibility_internal(line_name, state == Qt.Checked))
            self.line_toggles_layout.addWidget(checkbox)
            self.line_toggle_buttons[name] = checkbox

class AppMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual Y-Axis GraphWidget Demo")
        self.setGeometry(100, 100, 1100, 800)

        self.central_w = QWidget()
        self.setCentralWidget(self.central_w)
        self.app_layout = QVBoxLayout(self.central_w)

        self.graph_display_widget = GraphWidget(self)
        self.app_layout.addWidget(self.graph_display_widget)

        self.load_data_button = QPushButton("Load/Re-Plot Sample Data with Dual Y-Axis")
        self.load_data_button.clicked.connect(self.load_new_data_into_graph)
        self.app_layout.addWidget(self.load_data_button)

        self.load_new_data_into_graph()

    def _generate_sample_df_dual_y(self):
        rng = pd.date_range('2024-01-15 00:00:00', periods=150, freq='H')
        data = {
            'Timestamp': rng,
            'Temperature': np.sin(np.linspace(0, 5 * np.pi, 150)) * 10 + 25 + np.random.randn(150) * 0.5,
            'Humidity': np.cos(np.linspace(0, 3 * np.pi, 150)) * 20 + 50 + np.random.randn(150) * 1,
            'Pressure': (np.sin(np.linspace(0, 1.5 * np.pi, 150)) * 10 + 1010) + np.random.randn(150) * 0.2
        }
        return pd.DataFrame(data)

    def load_new_data_into_graph(self):
        sample_df = self._generate_sample_df_dual_y()

        pens_pri = [
            pg.mkPen(color=(50, 200, 255), width=2),
            pg.mkPen(color=(150, 255, 150), width=2)
        ]
        pens_sec = [
            pg.mkPen(color=(255, 180, 50), width=2, style=Qt.SolidLine)
        ]

        self.graph_display_widget.plot_dataframe(
            df=sample_df,
            x_col_datetime='Timestamp',
            y_cols_primary=['Temperature', 'Humidity'],
            y_cols_secondary=['Pressure'],
            title="Sensor Data (Dual Y-Axis)",
            pens_primary=pens_pri,
            pens_secondary=pens_sec
        )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = AppMainWindow()
    main_window.show()
    sys.exit(app.exec_())