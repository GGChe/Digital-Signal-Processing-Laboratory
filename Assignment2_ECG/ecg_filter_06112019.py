"""
#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia

The input signal is a .dat file with four columns -> [time, Channel1, Channel2, Channel3]
The channels 2 and 3 were recorded at x5 amplification so, the amplitude must be divided by 5.
The total time of recording is was: 47 seconds

"""

import numpy as np
import matplotlib.pylab as plt
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


# -----------

class FIR_filter(object):
    buffer = []
    P: int
    h = []

    def __init__(self, ntaps, f0, f1, f2):
        self.buffer = np.zeros(ntaps)
        self.P = 0
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
        hreal = np.real(hc)
        h_shift = np.zeros(ntaps)
        h_shift[0:int(ntaps / 2)] = hreal[int(ntaps / 2):ntaps]
        h_shift[int(ntaps / 2):ntaps] = hreal[0:int(ntaps / 2)]
        self.h = h_shift

    def dofilter(self, v):
        self.buffer[self.P] = v
        result = np.sum(self.buffer[:] * self.h[:])
        if self.P == len(self.buffer):
            self.P = 0
        if self.P < len(self.buffer):
            self.P = self.P + 1

        return v


for i in range(0, len(ECG)):
    ecgin = ECG[i]
    FIR = FIR_filter(200, 1, 45, 500)
    y = FIR.dofilter(ecgin)

line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l), interval=50, blit=True)
plt.show()
