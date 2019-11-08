#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia


"""
The input signal is a .dat file with four columns -> [time, Channel1, Channel2, Channel3]
The channels 2 and 3 were recorded at x5 amplification so, the amplitude must be divided by 5.
The total time of recording is was: 47 seconds
"""

import numpy as np
import matplotlib.pylab as plt


data = np.loadtxt("Gabriel_Brea_norm.dat")
t = data[:, 0]
ECG = data[:, 1]

# 1000 Hz sampling rate
fs = 1000

# Plot of Channel 1
plt.figure(1)
plt.subplot(211)
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
    def __init__(self, h):
        self.Filter = h
        self.Buffer = np.zeros(len(h))

    def dofilter(self, v):
        resultFIR = 0
        for j in range(len(self.Buffer) - 1, 0, -1):
            self.Buffer[j] = self.Buffer[j - 1]  # buffer is used here to push input
        self.Buffer[0] = v
        for j in range(len(self.Buffer)):
            resultFIR += self.Filter[j] * self.Buffer[j]
        return resultFIR


f0 = 1
f1 = 45
f2 = 55
ntaps = 200

FIRfrequencyResponse = np.ones(ntaps)
# Limits for the filtering
k0 = int((f0 / fs) * ntaps)
k1 = int((f1 / fs) * ntaps)
k2 = int((f2 / fs) * ntaps)

FIRfrequencyResponse[k1:k2 + 1] = 0
FIRfrequencyResponse[ntaps - k2:ntaps - k1 + 1] = 0
FIRfrequencyResponse[0:k0 + 1] = 0
FIRfrequencyResponse[ntaps - k0:ntaps] = 0

timeFIR = np.fft.ifft(FIRfrequencyResponse)
h_real = np.real(timeFIR)
FIR_shifted = np.zeros(ntaps)
FIR_shifted[0:int(ntaps / 2)] = h_real[int(ntaps / 2):ntaps]
FIR_shifted[int(ntaps / 2):ntaps] = h_real[0:int(ntaps / 2)]

myFIR = FIR_filter(FIR_shifted)
y = np.zeros(len(ECG))
for i in range(len(ECG)):
    y[i] = myFIR.dofilter(ECG[i])


plt.figure(1)
plt.subplot(212)
plt.plot(y)

print("HELLO FROM ECG_FILTER!!!")
plt.show()