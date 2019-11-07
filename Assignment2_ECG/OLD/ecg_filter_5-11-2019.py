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
f = np.linspace(0, fs, len(ch1))  # Full spectrum frequency range
plt.figure(2)
plt.plot(f, 20 * np.log10(abs(ecg_fft)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
# plt.xscale('log')


"""
M = 200  # Number of Taps
f_resp = np.ones(M)
# Limits for the filtering
f1 = int((45 / fs) * M)
f2 = int((55 / fs) * M)
f0 = int((5 / fs) * M)
f_resp[f1:f2+1] = 0
f_resp[M-f2:M-f1+1] = 0
f_resp[0:f0+1] = 0
f_resp[M-f0:M] = 0
hc = np.fft.ifft(f_resp)
h = np.real(hc)
h_shift = np.zeros(M)
h_shift[0:int(M/2)] = h[int(M/2):M]
h_shift[int(M/2):M] = h[0:int(M/2)]
h_wind=h_shift * np.hamming(M)
y = signal.lfilter(h_wind, 1, ch1)

plt.figure(3)
plt.plot(y)

"""

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
        y = signal.lfilter(h_shift, 1, ch1)
        plt.figure(3)
        plt.plot(y)
        return y


classfilter = FIR_filter(200, 5, 45, 100)
y = classfilter.dofilter()

plt.figure(4)
plt.plot(20 * np.log10(np.fft.fft(y, fs)))
plt.xscale('log')
plt.show()

"""
class FIR_filter:
    def __init__(self, ntaps, f0, f1, f2):
        self.M = ntaps
        self.f0 = f0
        self.f1 = f1
        self.f2 = f2
        
    def dofilter(self, ntaps, f0, f1, f2):
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
        h_wind = h_shift * np.hamming(ntaps)
        y = signal.lfilter(h_wind, 1, ch1)
        return y
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
