#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia


"""
The input signal is a .dat file with four columns -> [time, Channel1, Channel2, Channel3]
The channels 2 and 3 were recorded at x5 amplification so, the amplitude must be divided by 5.
The total time of recording is was: 47 seconds
"""

import numpy as np
import matplotlib.pylab as plt
ecg = np.loadtxt("Gabriel_Brea_norm.dat")
t = ecg[:, 0]
ch1 = ecg[:, 1]
ch2 = ecg[:, 2]
ch2 = ecg[:, 3]

fs = 1000

# Plot of Channel 1
plt.plot(t, ch1)
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

# Fourier transform Calculation

ecg_fft = np.fft.fft(ch1)
f = np.linspace(0, fs, len(ch1)) # Full spectrum frequency range


plt.figure(2)
plt.plot(f, 20*np.log10(abs(ch1)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
# plt.xscale('log')



plt.show()

class FIR_filter:
    def __init__(self, _coefficients):
        self._coefficients=_coefficients
        self.buffer=np.zeros(len(_coefficients))

    def dofilter(self, v):
        M = 200
        k1 = int(45/fs * M)
        k2 = int(55 / fs * M)
        x = np.ones(M)
        x[k1:k2+1] = 0
        x[M-k2:M-k1+1] = 0
        x = np.fft.ifft(x)
        x = np.real()

        return result



fs = 1000
f1 = 45
f2 = 55
scale = 2 ** 12

b = signal.firwin(999, [f1 / fs * 2, f2 / fs * 2])
b = b * scale
b = b.astype(int)
np.savetxt("coeff12bit.dat", b)


