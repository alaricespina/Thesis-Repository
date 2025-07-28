import pyqtgraph as pg
from PyQt5 import QtWidgets
import sys
import numpy as np

app = QtWidgets.QApplication(sys.argv)

win = pg.GraphicsLayoutWidget(show=True, title="Scrollable Graph")
win.resize(800, 600)

plot = win.addPlot(title="Data Plot")

x = np.arange(0, 100, 0.1)
y = np.sin(x)

plot.plot(x, y, pen='b')

plot.setXRange(0, 20) #Setting an initial range, so the whole data is not initially shown.
plot.setMouseEnabled(x=True, y=False)
if __name__ == '__main__':
    sys.exit(app.exec_())