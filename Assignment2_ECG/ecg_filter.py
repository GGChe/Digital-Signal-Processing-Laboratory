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

ecg = np.loadtxt("Gabriel_Brea_norm.dat")
t = ecg[:, 0]
ch1 = ecg[:, 1]

# 1000 Hz sampling rate
fs = 1000

# Plot of Channel 1
plt.figure(1)
plt.plot(t, ch1)
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

# Fourier transform Calculation
ecg_fft = np.fft.fft(ch1)
f = np.linspace(0, fs, len(ch1)) # Full spectrum frequency range
plt.figure(2)
plt.plot(f, 20*np.log10(abs(ecg_fft)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.xscale('log')

# filter
M = 200
k1 = int(45/fs * M)
k2 = int(55/fs * M)

x = np.ones(M)

x[k1:k2+1] = 0
x[M-k2:M-k1+1] = 0
x = np.fft.ifft(x)

x = np.real(x)

h = np.zeros(M)

h[0:int(M/2)] = x[int(M/2):M]
h[int(M/2):M] = x[0:int(M/2)]

h = h * np.hamming(M)

y2 = signal.lfilter(h, 1, ch1)

plt.figure(3)
plt.plot(y2)

plt.show()


class FIR_filter:
    def __init__(self, _coefficients):
        self._coefficients=_coefficients
        self.buffer=np.zeros(len(_coefficients))

    def dofilter(self, v):

        return result

"""
fs = 1000
f1 = 45
f2 = 55
scale = 2 ** 12

b = signal.firwin(999, [f1 / fs * 2, f2 / fs * 2])
b = b * scale
b = b.astype(int)
np.savetxt("coeff12bit.dat", b)
"""

