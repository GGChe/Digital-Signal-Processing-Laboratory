#!/usr/bin/python3
"""
Plots channels zero and one in two different windows. Requires pyqtgraph.
"""

import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import numpy as np

from pyfirmata2 import Arduino

PORT = Arduino.AUTODETECT

# create a global QT application object
app = QtGui.QApplication(sys.argv)

# signals to all threads in endless loops that we'd like to run these
running = True
class IIR_filter:
    def __init__(self, _b0, _b1, _a1, _a2):
        self.a1 = _a1
        self.a2 = _a2
        self.b0 = _b0
        self.b1 = _b1
        self.buffer1 = 0
        self.buffer2 = 0

    def filter(self, x):
        acc_input = x - self.buffer1 * self.a1 - self.buffer2 * self.a2
        acc_output = acc_input * self.b0 + self.buffer1 * self.b1 + + self.buffer2*self.b2
        self.buffer2 = self.buffer1
        self.buffer1 = acc_input
        return acc_output

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


# Let's create two instances of plot windows
qtPanningPlot1 = QtPanningPlot("Arduino 1st channel")
qtPanningPlot2 = QtPanningPlot("Arduino 2nd channel")

# sampling rate: 100Hz
samplingRate = 100

# Normalised frequency 0.1
# T = 1
f = 0.1

# Q factor
q = 10

# s infinite as defined for a 2nd order resonator (see impulse invar)
si = np.complex(-np.pi * f / q, np.pi * f * np.sqrt(1 / (q ** 2)))

# Coefficients
b0 = 1
b1 = -1
a1 = np.real(-(np.exp(si)+np.exp(np.conjugate(si))))
a2 = np.exp(2*np.real(si))

f = IIR_filter(b0, b1, a1, a2)

x = np.zeros(100)
x[10] = 1
y = np.zeros(100)

# called for every new sample which has arrived from the Arduino
def callBack(data):
    # send the sample to the plotwindow
    qtPanningPlot1.addData(data)
    ch1 = board.analog[0].read()
    # 1st sample of 2nd channel might arrive later so need to check
    out = f.filter(ch1)
    if ch1:
        qtPanningPlot2.addData(out)


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