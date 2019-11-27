#!/usr/bin/python3
"""
Plots channels zero and one in two different windows. Requires pyqtgraph.
"""

import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from pyfirmata2 import Arduino
import scipy.signal as signal

fs = 100


PORT = Arduino.AUTODETECT

# create a global QT application object
app = QtGui.QApplication(sys.argv)

# signals to all threads in endless loops that we'd like to run these
running = True

class IIR_filter:
    def __init__(self, _a, _b):
        q_scalingFactor = 15
        self.a0 = _a[0]
        self.a1 = _a[1]
        self.a2 = _a[2]
        self.b0 = _b[0]
        self.b1 = _b[1]
        self.b2 = _b[2]
        # Delay Lines
        self.m_x1 = 0
        self.m_x2 = 0
        self.m_y1 = 0
        self.m_y2 = 0

    def filter(self, x):

        output = 0
        output = self.b0*x + self.b1*self.m_x1+self.b2 * self.m_x2 - self.a1 * self.m_y1 - self.a2 * self.m_y2;
        # Update delay lines
        self.m_x2 = self.m_x1
        self.m_x1 = x
        self.m_y2 = self.m_y1
        self.m_y1 = output
        return output

class QtPanningPlot:

    def __init__(self, title):
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle(title)
        self.plt = self.win.addPlot()
        self.plt.setYRange(-1, 1)
        self.plt.setXRange(0, 500)
        self.curve = self.plt.plot()
        self.data = []
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)
        self.layout = QtGui.QGridLayout()
        self.win.setLayout(self.layout)
        self.win.show()

    def update(self):
        self.data = self.data[-500:]
        if self.data:
            self.curve.setData(np.hstack(self.data))

    def addData(self, d):
        self.data.append(d)


# Butterworth Filters
# order of filter
n = 2

# cutoff frequency
fc = 15
fn = (fc / fs)*2

# Coefficients
b, a = signal.butter(n, fn, 'low')
myFilter = IIR_filter(a, b)

# Let's create two instances of plot windows
qtPanningPlot1 = QtPanningPlot("Arduino 1st channel")
qtPanningPlot2 = QtPanningPlot("Arduino 2nd channel")

# sampling rate: 100Hz
samplingRate = 100


# called for every new sample which has arrived from the Arduino
def callBack(data):
    # send the sample to the plotwindow
    qtPanningPlot1.addData(data)
    ch1 = board.analog[1].read()
    # 1st sample of 2nd channel might arrive later so need to check
    if ch1:
        qtPanningPlot2.addData(myFilter.filter(ch1))


# Get the Ardunio board.
board = Arduino(PORT)

# Set the sampling rate in the Arduino
board.samplingOn(1000 / samplingRate)

# Register the callback which adds the data to the animated plot
board.analog[0].register_callback(callBack)

# Enable the callback
board.analog[0].enable_reporting()
board.analog[1].enable_reporting()

# showing all the windows
app.exec_()

# needs to be called to close the serial port
board.exit()

print("Finished")