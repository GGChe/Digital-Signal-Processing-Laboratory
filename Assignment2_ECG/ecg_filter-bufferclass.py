#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia


"""
The input signal is a .dat file with four columns -> [time, Channel1, Channel2, Channel3]
The channels 2 and 3 were recorded at x5 amplification so, the amplitude must be divided by 5.
The total time of recording is was: 47 seconds
"""
import time
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
"""ecg_fft = np.fft.fft(ECG)
f = np.linspace(0, fs, len(ECG))  # Full spectrum frequency range
plt.figure(2)
plt.plot(f, 20 * np.log10(abs(ecg_fft)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.xscale('log')"""

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
        output = np.sum( currentBuffer[:] * self.FIRfilter[:])
        if self.P == 200 - 1:
            self.P = 0
        if self.P < 200 - 1:
            self.P = self.P + 1
        print(output)
        # print("Buffer:", self.buffer, "-- Output: ", output)
        return output


# Create an instance of an animated scrolling window
# To plot more channels just create more instances and add callback handlers below

f = FIR_filter(200, 1, 45, 55)
y= np.empty(len(ECG))
for i in range(len(ECG)):
    y[i] = f.dofilter(ECG[i])

plt.figure(2)
plt.plot(y)
plt.show()


print('finished')
