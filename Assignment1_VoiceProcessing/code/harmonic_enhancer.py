#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia

import matplotlib.pylab as plt
import numpy as np
import scipy.io.wavfile as wav
import wave
plt.rcParams.update({'font.size': 20})

fs, soundwave = wav.read ('original.wav') # Read sound signal

# Representation of Time Domain
t = np.linspace(0, len(soundwave)/fs, len(soundwave))  # time vector

# ---- Figure 1 ------
plt.figure(1)
plt.plot(t, soundwave)
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')

# ---- Frequency Domain Calculations ----
fftSoundWave = np.fft.fft(soundwave)
f = np.linspace(0, fs, len(fftSoundWave)) # Full spectrum frequency range

fftSoundwaveHalf = fftSoundWave[:len(fftSoundWave) // 2]  # REMOVE  the half of the mirrored spectrum
fHalf = np.linspace(0, fs / 2, len(fftSoundwaveHalf)) # Half spectrum frequency range

# ---- Figure 2 ----
# FFT signal in linear and logarithmic axis
plt.figure(2)
plt.plot(fHalf, 20*np.log10(abs(fftSoundwaveHalf)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.xscale('log')

# ---- Voice Processing ----
"""
Description:

The processing consists of two parts: amplification of the harmonic between 300 and 355 Hz and attenuation of the 
harmonic between 400 and 460 Hz. The reason is to give same importance (magnitude range) to the first 4 harmonics 
of the signal

We would need to convert from frequency to the exact sample of the array where we have to start operating. Therefore, 
the conversion needs to be parsed to int obtaining the position of the fft signal where we apply the operations.

"""

fLow = 7000  # Hz
fHigh = fs/2  # Hz
fAmplifyL1 = 1700  # Hz
fAmplifyH1 = 2400  # Hz
fAmplifyL2 = 3400  # Hz
fAmplifyH2 = 3800  # Hz
a1 = 8  # amplification 1st harmonic
a2 = 8  # amplification 2nd harmonic

# Conversion from frequency to sample
fL = int((fLow / fs) * len(fftSoundWave))
fH = int((fHigh / fs) * len(fftSoundWave))

fAL1 = int((fAmplifyL1 / fs) * len(fftSoundWave))
fAH1 = int((fAmplifyH1 / fs) * len(fftSoundWave))

fAL2 = int((fAmplifyL2 / fs) * len(fftSoundWave))
fAH2 = int((fAmplifyH2 / fs) * len(fftSoundWave))

FSW = fftSoundWave  # pass fftSoundWave to other variable to not overlap the original signal

# Operations

FSW[fL:fH] = FSW[fL:fH] * 0
FSW[len(FSW) - fH:len(FSW) - fL] = FSW[len(FSW) - fH:len(FSW) - fL] * 0

FSW[fAL1:fAH1] = FSW[fAL1:fAH1] * a1
FSW[len(FSW) - fAH1:len(FSW) - fAL1] = FSW[len(FSW) - fAH1:len(FSW) - fAL1] * a1

FSW[fAL2:fAH2] = FSW[fAL2:fAH2] * a2
FSW[len(FSW) - fAH2:len(FSW) - fAL2] = FSW[len(FSW) - fAH2:len(FSW) - fAL2] * a2

# Reconstruction of the signal
timeFilteredSoundWave = np.fft.ifft(FSW)
timeFilteredSoundWave = np.real(timeFilteredSoundWave)

# ---- Figure 3 ----
# Represents the result signal in linear and logarithmic axis.
plt.figure(3)
plt.plot(fHalf, 20*np.log10(abs(FSW[:len(FSW) // 2])))
plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

# ---- Save Processed Sound Recording ----
f = wave.open(r"improved.wav", "wb")
# 1 channel with a sample width of 2 and frame rate of 2*fs which is standard
f.setnchannels(1)
f.setsampwidth(2)
f.setframerate(2*fs)
timeFilteredSoundWave=timeFilteredSoundWave.astype(int)
f.writeframes(timeFilteredSoundWave.tostring())

# --- EXTRA ----
# Here, the spectogram is presented to see the variation of frequencies with respect to the time domain signal.
# This representation is very useful to analyse the profile of the sound and the frequencies distribution
# along the time. Thus, frequency over time is represented.
plt.figure(4)
plt.subplot(311)
plt.plot(t, soundwave)
plt.xlabel('Time (s)')
plt.xlim([0, len(t)/fs])
plt.ylabel(' Magnitude')
plt.subplot(312)
powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(soundwave, Fs=fs)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.subplot(313)
powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(timeFilteredSoundWave, Fs=fs)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')

plt.show()
