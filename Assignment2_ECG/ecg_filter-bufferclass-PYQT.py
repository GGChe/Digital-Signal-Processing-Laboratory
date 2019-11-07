#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia


"""
The input signal is a .dat file with four columns -> [time, Channel1, Channel2, Channel3]
The channels 2 and 3 were recorded at x5 amplification so, the amplitude must be divided by 5.
The total time of recording is was: 47 seconds
"""
import pyqtgraph as pg
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtGui, QtCore

data = np.loadtxt("Gabriel_Brea_norm.dat")
t = data[:, 0]
ECG = data[:, 1]

# 1000 Hz sampling rate
fs = 1000

# Plot of Channel 1
plt.figure(1)
plt.plot(t, ECG)
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')


class Plot2D():
    def __init__(self):
        self.traces = dict()

        # QtGui.QApplication.setGraphicsSystem('raster')
        self.app = QtGui.QApplication([])
        # mw = QtGui.QMainWindow()
        # mw.resize(800,800)

        self.win = pg.GraphicsWindow(title="Basic plotting examples")
        self.win.resize(1000, 600)
        self.win.setWindowTitle('pyqtgraph example: Plotting')

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

        self.canvas = self.win.addPlot(title="Pytelemetry")

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def trace(self, name, dataset_x, dataset_y):
        if name in self.traces:
            self.traces[name].setData(dataset_x, dataset_y)
        else:
            self.traces[name] = self.canvas.plot(pen='y')


class RingBuffer:
    def __init__(self, size):
        self.data = [0 for i in range(size)]

    def append(self, x):
        self.data.pop(0)
        self.data.append(x)

    def get(self):
        return self.data


class FIR_filter(object):
    buffer = RingBuffer(200)
    P: int
    FIRfilter = []

    def __init__(self, ntaps, f0, f1, f2):
        self.P = 0
        self.FIRfilter = np.zeros(ntaps)
        f_resp = np.ones(ntaps)
        # Limits for the filtering
        k0 = int((f0 / fs) * ntaps)
        k1 = int((f1 / fs) * ntaps)
        k2 = int((f2 / fs) * ntaps)
        f_resp[k1:k2 + 1] = 0
        f_resp[ntaps - k2:ntaps - k1 + 1] = 0
        f_resp[0:k0 + 1] = 0
        f_resp[ntaps - k0:ntaps] = 0
        hc = np.fft.ifft(f_resp)
        h = np.real(hc)
        h_shift = np.zeros(ntaps)
        h_shift[0:int(ntaps / 2)] = h[int(ntaps / 2):ntaps]
        h_shift[int(ntaps / 2):ntaps] = h[0:int(ntaps / 2)]
        w = np.blackman(ntaps)
        self.FIRfilter = h_shift * w
        plt.figure(5)
        plt.plot(self.FIRfilter)

    def dofilter(self, v):
        self.buffer.append(v)
        currentBuffer = self.buffer.get()
        output = np.sum(currentBuffer[:] * self.FIRfilter[:])
        if self.P == 200 - 1:
            self.P = 0
        if self.P < 200 - 1:
            self.P = self.P + 1
        print(output)
        # print("Buffer:", self.buffer, "-- Output: ", output)
        return output


# ----------------- MAIN -----------------
# Create an instance of an animated scrolling window
# To plot more channels just create more instances and add callback handlers below
p = Plot2D()
f = FIR_filter(200, 1, 45, 55)  # Create the object FIR_filter initialising the filter


def update():
    y = np.empty(len(ECG))
    for i in range(len(ECG)):
        c = f.dofilter(ECG[i])
        p.trace("cos", t, c)


timer = QtCore.QTimer()
# timer.timeout.connect(update)
timer.start(50)
p.start()
plt.show()

print('finished')
