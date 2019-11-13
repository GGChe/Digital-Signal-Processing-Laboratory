"""
---------------------------------------------------------
Code for the Assignment 1 of Digital Signal Processing
Authors: Gabriel Galeote-Checa & Anton Saikia
---------------------------------------------------------
The input signal is a .dat file with four columns -> [time, Channel1, Channel2, Channel3]
The channels 2 and 3 were recorded at x5 amplification so, the amplitude must be divided by 5.
The total time of recording is was: 47 seconds
"""

import numpy as np
import matplotlib.pylab as plt

# Initialise the script
dataFromECG = np.loadtxt("Gabriel_Brea_norm.dat")
time = dataFromECG[:, 0]
myECG = dataFromECG[:, 1]
fs = 1000  # 1000 Hz sampling rate

# Plot the ECG prefiltered
plt.figure(1)
plt.subplot(211)
plt.plot(time, myECG)
plt.title("ECG Prefiltered")
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')

# Fourier transform of the ECG to analyse the frequency spectrum
# However, only the first half of the frequency spectrum is shown as the other half is mirrored.
ecg_fft = np.fft.fft(myECG)
f_axis = np.linspace(0, fs, len(myECG))  # Full spectrum frequency range
plt.figure(2)
plt.subplot(211)
plt.plot(f_axis[:int(len(f_axis) / 2)], 20 * np.log10(abs(ecg_fft[:int(len(ecg_fft) / 2)])))
plt.title("Frequency Spectrum of the ECG signal")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

"""
This class calculate the result of the FIR filter for a given value. The class function dofilter(input) 
introduces the given value of the signal in the buffer in the current position after a proper management of the 
buffer shifting. Then, it is calculated the mathematical result of FIR filter of the buffer storaged that was 
previously shifted to put in the first position the current input value. 
"""


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


# Define the frequencies for the FIR filter
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
y = np.zeros(len(myECG))
for i in range(len(myECG)):
    y[i] = myFIR.dofilter(myECG[i])

plt.figure(1)
plt.subplot(212)
plt.plot(y)

yfft = np.fft.fft(y)
plt.figure(2)
plt.subplot(212)
plt.plot(f_axis[:int(len(f_axis) / 2)], 20 * np.log10(abs(yfft[:int(len(ecg_fft) / 2)])))

plt.show()
