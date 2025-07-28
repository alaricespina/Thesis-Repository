import sys
from PyQt5 import QtWidgets, QtCore, QtGui

class TabbedWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tabbed Interface")
        self.setGeometry(100, 100, 800, 600)

        self.tab_widget = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Set tab position to the bottom
        self.tab_widget.setTabPosition(QtWidgets.QTabWidget.South)

        self.tab1 = QtWidgets.QWidget()
        self.tab2 = QtWidgets.QWidget()
        self.tab3 = QtWidgets.QWidget()

        self.tab_widget.addTab(self.tab1, "Tab 1")
        self.tab_widget.addTab(self.tab2, "Tab 2")
        self.tab_widget.addTab(self.tab3, "Tab 3")

        self.setup_tabs()

    def setup_tabs(self):
        # ... (tab content setup remains the same)
        pass

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TabbedWindow()
    window.show()
    sys.exit(app.exec_())