#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia


import matplotlib.pylab as plt
import numpy as np
import scipy.io.wavfile as wav
from scipy.io.wavfile import write

fs, soundwave = wav.read ('original2_mono.wav')

# plot time domain
t = np.linspace (0, len(soundwave)/fs, len(soundwave))
plt.figure(1)
plt.plot(t, soundwave)
plt.xlabel('time (s)')
plt.ylabel('amplitude (samples)')


# Plot Frequency Domain
                                          
fftSoundWave = np.fft.fft(soundwave)
fftSoundwaveHalf = fftSoundWave[:len(fftSoundWave) // 2]  # REMOVE  the half of the spectrum
fHalf = np.linspace(0, fs / 2, len(fftSoundwaveHalf))

# Plot in linear and logarithmic axis of the signal in frequency domain
plt.figure(2)
ax = plt.subplot(211)
ax.plot(fHalf, abs(fftSoundwaveHalf))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Samples')
ax = plt.subplot(212)
ax.loglog(fHalf, abs(fftSoundwaveHalf))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')

# Plot of the spectrogram
plt.figure(3)
plt.subplot(211)
plt.plot(t, soundwave)
plt.xlabel('time (s)')
plt.ylabel('amplitude (samples)')
plt.subplot(212)
powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(soundwave, Fs=fs)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.yscale('log')
plt.ylim([10, 1e4])

# Signal Processing

fLow=3000
fHigh=8000
fLowSample = int(np.round((fLow/(fs/2))*len(fftSoundwaveHalf)))
fHighSample = int(np.round((fHigh/(fs/2))*len(fftSoundwaveHalf)))
filteredSoundWave = fftSoundwaveHalf
filteredSoundWave[fLowSample:fHighSample] = filteredSoundWave[fLowSample:fHighSample] * 0.01
plt.figure(4)
plt.plot(fHalf, abs(filteredSoundWave))
timeFilteredSoundWave = np.fft.ifft(filteredSoundWave)
plt.figure(5)
plt.plot(t, )


write('SignalRectified.wav', fs, timeFilteredSoundWave)
plt.show()




