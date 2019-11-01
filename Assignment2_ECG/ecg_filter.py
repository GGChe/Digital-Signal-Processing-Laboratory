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
# plt.xscale('log')

M = 100
f_resp=np.ones(100)
# note we need to add "+1" to the end because the end of
# the range is not included.
f_resp[4:6+1]=0
f_resp[94:96+1]=0
hc=np.fft.ifft(f_resp)
h=np.real(hc)
# this is from index 0 to index 49 on the left
# and on the right hand side from index 50 to index 99
h_shift = np.zeros(100)
h_shift[0:50]=h[50:100]
h_shift[50:100]=h[0:50]
h_wind=h_shift * np.hamming(100)

f_resp2=np.ones(100)
# note we need to add "+1" to the end because the end of
# the range is not included.
f_resp2[2:50]=0
f_resp2[50:98]=0
hc2=np.fft.ifft(f_resp2)
h2=np.real(hc2)
# this is from index 0 to index 49 on the left
# and on the right hand side from index 50 to index 99
h_shift2 = np.zeros(100)
h_shift2[0:50] = h2[50:100]
h_shift2[50:100] = h2[0:50]
h_wind2=h_shift2 * np.hamming(100)

y2 = signal.lfilter(h_wind2, 1, ch1)
plt.figure(3)
plt.plot(y2)
plt.show()

"""
# Code from lectures
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
"""

"""


class FIR_filter:
    def __init__(self,_coefficients):
        self.ntaps = len(_coefficients)
        self.coefficients = _coefficients
        self.buffer = np.zeros(self.ntaps)
    def dofilter(self,v):
        buffer = np.roll(buffer,1)
        self.buffer[0] = v
        return np.inner(self.buffer,self.coefficients)
"""
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

