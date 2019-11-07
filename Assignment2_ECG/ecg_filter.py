#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia


"""
The input signal is a .dat file with four columns -> [time, Channel1, Channel2, Channel3]
The channels 2 and 3 were recorded at x5 amplification so, the amplitude must be divided by 5.
The total time of recording is was: 47 seconds
"""

import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, coeff):
        self.P = 0
        self.FIRfilter = coeff

    def dofilter(self, v):
        self.buffer.append(v)
        currentBuffer = self.buffer.get()
        y = 0
        for j in range(len(coeff)):
            y += currentBuffer[j] * self.FIRfilter[j]
        return y


# Calculate the FIR filter
f0 = 3
f1 = 45
f2 = 55
ntaps = 200
f_resp = np.ones(ntaps)

# Conversion to samples in the FIR filter
k0 = int((f0 / fs) * ntaps)
k1 = int((f1 / fs) * ntaps)
k2 = int((f2 / fs) * ntaps)

# Calculate filter type
f_resp[k1:k2 + 1] = 0
f_resp[ntaps - k2:ntaps - k1 + 1] = 0
f_resp[0:k0 + 1] = 0
f_resp[ntaps - k0:ntaps] = 0
hc = np.fft.ifft(f_resp)
h = np.real(hc)
coeff = np.zeros(ntaps)
coeff[0:int(ntaps / 2)] = h[int(ntaps / 2):ntaps]
coeff[int(ntaps / 2):ntaps] = h[0:int(ntaps / 2)]
w = np.blackman(ntaps)
coeff = coeff * w

f = FIR_filter(coeff)

y = np.empty(len(ECG))
for i in range(len(ECG)):
    y[i] = f.dofilter(ECG[i])


plt.figure(2)
plt.plot(y)
plt.show()

print('finished')
