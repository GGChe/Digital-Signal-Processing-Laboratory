#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia


"""
The input signal is a .dat file with four columns -> [time, Channel1, Channel2, Channel3]
The channels 2 and 3 were recorded at x5 amplification so, the amplitude must be divided by 5.
The total time of recording is was: 47 seconds
"""

import numpy as np
import matplotlib.pylab as plt
import scipy.signal as signal

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
        P = 0
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
        self.Buffer[self.P] = v
        output = np.sum(self.Buffer[:] * self.FIRfilter[:])
        if self.P == len(self.buffer):
            self.P = 0
        if self.P < len(self.buffer):
            self.P = self.P + 1
        print(output)
        return output

classfilter = FIR_filter(200, 1, 45, 500)
for i in range(0, len(ECG)):
    ecgin = ECG[i]
    y = classfilter.dofilter(ecgin)

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,




plt.show()
