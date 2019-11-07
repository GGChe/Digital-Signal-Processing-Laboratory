from pyqtgraph import PlotWidget, plot

from PyQt5 import QtWidgets, uic
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import numpy as np
import os

data = np.loadtxt("Gabriel_Brea_norm.dat")
t = data[:, 0]
ECG = data[:, 1]

# 1000 Hz sampling rate
fs = 1000


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

    def dofilter(self, v):
        self.buffer.append(v)
        currentBuffer = self.buffer.get()
        output = np.sum(currentBuffer[:] * self.FIRfilter[:])
        if self.P == 200 - 1:
            self.P = 0
        if self.P < 200 - 1:
            self.P = self.P + 1
        # print("Buffer:", self.buffer, "-- Output: ", output)
        return output


f = FIR_filter(200, 1, 45, 55)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        y = np.zeros(len(t))
        hour = t
        for i in range(len(t)):
            y[i] = f.dofilter(ECG[i])
            temperature = y

        self.graphWidget.setBackground('w')
        self.graphWidget.plot(hour, temperature)
        pen = pg.mkPen(color=(255, 0, 0))
        self.graphWidget.plot(hour, temperature, pen=pen)

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
