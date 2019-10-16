#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia


import matplotlib.pylab as plt
import numpy as np
import scipy.io.wavfile as wav
from scipy.io.wavfile import write
import wave

fs, soundwave = wav.read ('CalibrationSentence.wav') # Read sound signal

# Representation of Time Domain
t = np.linspace (0, len(soundwave)/fs, len(soundwave))
plt.figure(1)
plt.plot(t, soundwave)
plt.xlabel('time (s)')
plt.ylabel('amplitude (samples)')

# Plot Frequency Domain
fftSoundWave = np.fft.fft(soundwave)
fftSoundwaveHalf = fftSoundWave[:len(fftSoundWave) // 2]  # REMOVE  the half of the spectrum
f = np.linspace(0, fs / 2, len(fftSoundWave))
fHalf = np.linspace(0, fs / 2, len(fftSoundwaveHalf))

print(len(f), len(fHalf))

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
#plt.yscale('log') # activate for logarithm scale
#plt.ylim([10, 1e4]) # set logarithm scale within that limits

# Signal Processing

fLow=2500
fHigh=8000
fAmplifyL = 85
fAmplifyH = 250
fL = int(np.round((fLow / (fs / 2)) * len(fftSoundWave))) # CHECK
fH = int(np.round((fHigh / (fs / 2)) * len(fftSoundWave))) # CHECk
fAL = int(np.round((fAmplifyL / (fs / 2)) * len(fftSoundWave)))
fAH = int(np.round((fAmplifyH / (fs / 2)) * len(fftSoundWave)))
FSW = fftSoundWave # Processing Sound Wave
FSW[fAL:fAH] = FSW[fAL:fAH] *2
FSW[fL:fH] = FSW[fL:fH] * 0.01
FSW[len(FSW) - fH:len(FSW) - fL] = FSW[len(FSW) - fH:len(FSW) - fL] * 0.01
plt.figure(4)
plt.subplot(211)
plt.plot(f, abs(FSW))
timeFilteredSoundWave = np.fft.ifft(FSW)
timeFilteredSoundWave = np.real(timeFilteredSoundWave)
plt.subplot(212)
plt.plot(f, abs(FSW))
plt.xscale('log')




# open and write a .wav file
f = wave.open(r"Processed.wav", "wb")

# set up the channels to 1、sample width to 2 and frame rate to 2*fs
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(2*fs)
# put x_clean into new audio
timeFilteredSoundWave=timeFilteredSoundWave.astype(int)
f.writeframes(timeFilteredSoundWave.tostring())



plt.show()




