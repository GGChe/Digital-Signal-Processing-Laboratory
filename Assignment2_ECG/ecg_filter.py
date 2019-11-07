#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia


"""
The input signal is a .dat file with four columns -> [time, Channel1, Channel2, Channel3]
The channels 2 and 3 were recorded at x5 amplification so, the amplitude must be divided by 5.
The total time of recording is was: 47 seconds
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# Fourier transform Calculation
ecg_fft = np.fft.fft(ECG)
f = np.linspace(0, fs, len(ECG))  # Full spectrum frequency range
plt.figure(2)
plt.plot(f, 20 * np.log10(abs(ecg_fft)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.xscale('log')


class FIR_filter(object):
    buffer = []
    P: int
    FIRfilter = []

    def __init__(self, ntaps, f0, f1, f2):
        self.buffer = np.zeros(ntaps)
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
        self.FIRfilter = h_shift

    def dofilter(self, v):
        self.buffer[self.P] = v
        output = np.sum(self.buffer[:] * self.FIRfilter[:])
        if self.P == len(self.buffer) - 1:
            self.P = 0
        if self.P < len(self.buffer) - 1:
            self.P = self.P + 1
        return output


"""class RealtimePlotWindow:

    def __init__(self):
        # create a plot window
        self.fig, self.ax = plt.subplots()
        # that's our plotbuffer
        self.plotbuffer = np.zeros(500)
        # create an empty line
        self.line, = self.ax.plot(self.plotbuffer)
        # axis
        self.ax.set_ylim(0, 1)
        # That's our ringbuffer which accumluates the samples
        # It's emptied every time when the plot window below
        # does a repaint
        self.ringbuffer = []
        # start the animation
        plt.figure(4)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=100)

    # updates the plot
    def update(self, data):
        # add new data to the buffer
        self.plotbuffer = np.append(self.plotbuffer, self.ringbuffer)
        # only keep the 500 newest ones and discard the old ones
        self.plotbuffer = self.plotbuffer[-500:]
        self.ringbuffer = []
        # set the new 500 points of channel 9
        self.line.set_ydata(self.plotbuffer)
        return self.line,

    # appends data to the ringbuffer
    def addData(self, v):
        self.ringbuffer.append(v)
"""

# Create an instance of an animated scrolling window
# To plot more channels just create more instances and add callback handlers below
# realtimePlotWindow = RealtimePlotWindow()

classfilter = FIR_filter(200, 1, 45, 500)
for i in range(0, len(ECG)):
    ecgin = ECG[i]
    y = classfilter.dofilter(ecgin)
    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, y, np.arange(1, 20), interval=25)
plt.show()

print('finished')
