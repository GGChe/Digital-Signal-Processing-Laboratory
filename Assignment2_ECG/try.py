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


# -----------

class FIR_filter(object):
    def __init__(self, ntaps, f0, f1, f2):
        self.ntaps = ntaps
        self.f0 = f0
        self.f1 = f1
        self.f2 = f2

    def dofilter(self):
        f_resp = np.ones(self.ntaps)
        # Limits for the filtering
        k0 = int((self.f0 / fs) * self.ntaps)
        k1 = int((self.f1 / fs) * self.ntaps)
        k2 = int((self.f2 / fs) * self.ntaps)
        f_resp[k1:k2+1] = 0
        f_resp[self.ntaps-k2:self.ntaps-k1+1] = 0
        f_resp[0:k0+1] = 0
        f_resp[self.ntaps-k0:self.ntaps] = 0
        hc = np.fft.ifft(f_resp)
        h = np.real(hc)
        h_shift = np.zeros(self.ntaps)
        h_shift[0:int(self.ntaps / 2)] = h[int(self.ntaps / 2):self.ntaps]
        h_shift[int(self.ntaps / 2):self.ntaps] = h[0:int(self.ntaps / 2)]
        # h_wind = h_shift * np.hamming(self.ntaps)
        y = signal.lfilter(h_shift, 1, ECG)
        plt.figure(3)
        plt.plot(y)
        return y


classfilter = FIR_filter(200, 1, 45, 500)
y = classfilter.dofilter()

plt.figure(4)
plt.plot(20 * np.log10(np.fft.fft(y, fs)))
plt.xscale('log')
plt.show()