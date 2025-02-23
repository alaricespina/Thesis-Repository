import sys
from PyQt5 import QtWidgets

app = QtWidgets.QApplication(sys.argv)

window = QtWidgets.QMainWindow()
window.setWindowTitle("QFrame Example")

# Create a QFrame object
frame = QtWidgets.QFrame()
frame.setFrameShape(QtWidgets.QFrame.Box)
frame.setFrameShadow(QtWidgets.QFrame.Raised)
frame.setLineWidth(2)
frame.setStyleSheet("background-color: lightgray; border-color: black;")

# Create widgets and set the frame as their parent
label = QtWidgets.QLabel("This is a label", frame)
button = QtWidgets.QPushButton("Click me", frame)

# Use a layout manager to arrange the widgets within the frame
layout = QtWidgets.QVBoxLayout()
layout.addWidget(label)
layout.addWidget(button)
frame.setLayout(layout)

# Set the frame as the central widget of the main window
window.setCentralWidget(frame)

window.show()
sys.exit(app.exec_())