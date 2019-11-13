# python_live_plot.py

import random
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

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
        return output


filter = FIR_filter(200, 1, 45, 55)

plt.style.use('fast')

x_values = []
y_values = []

index = count()
fig, ax = plt.subplots()

def animate(i):
    x_values.append(next(index))
    y_values.append(filter.dofilter(ECG[i]))
    plt.cla()
    plt.plot(x_values, y_values)


ani = FuncAnimation(fig, animate, interval = 1)

plt.tight_layout()
plt.show()