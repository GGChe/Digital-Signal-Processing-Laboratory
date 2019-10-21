#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton Saikia


import matplotlib.pylab as plt
import numpy as np
import scipy.io.wavfile as wav
from scipy.io.wavfile import write
import wave

fs, soundwave = wav.read ('VoiceRecording.wav') # Read sound signal

# Representation of Time Domain
t = np.linspace (0, len(soundwave)/fs, len(soundwave))
plt.figure(1)
plt.plot(t, soundwave)
plt.xlabel('time (s)')
plt.ylabel('amplitude (samples)')

# Plot Frequency Domain
fftSoundWave = np.fft.fft(soundwave)
fftSoundwaveHalf = fftSoundWave[:len(fftSoundWave) // 2]  # REMOVE  the half of the spectrum
f = np.linspace(0, fs, len(fftSoundWave))
fHalf = np.linspace(0, fs / 2, len(fftSoundwaveHalf))

# Plot in linear and logarithmic axis of the signal in frequency domain
plt.figure(2)
ax = plt.subplot(211)
ax.plot(fHalf, abs(fftSoundwaveHalf))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Samples')
ax = plt.subplot(212)
ax.plot(fHalf, 20*np.log10(abs(fftSoundwaveHalf)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.xscale('log')
# Plot of the spectrogram
plt.figure(3)
plt.subplot(211)
plt.plot(t, soundwave)
plt.xlabel('time (s)')
plt.xlim([0, len(t)/fs])
plt.ylabel('amplitude (samples)')
plt.subplot(212)
powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(soundwave, Fs=fs)
plt.xlabel('Time')
plt.ylabel('Frequency')
#plt.yscale('log') # activate for logarithm scale
#plt.ylim([10, 1e4]) # set logarithm scale within that limits

# Signal Processing9
fLow=0
fHigh=150
fAmplifyL = 305
fAmplifyH = 325
acoef=2

fL = int(np.round((fLow / fs ) * len(fftSoundWave))) # CHECK
fH = int(np.round((fHigh / fs ) * len(fftSoundWave))) # CHECk

fAL = int(np.round((fAmplifyL / fs ) * len(fftSoundWave)))
fAH = int(np.round((fAmplifyH / fs ) * len(fftSoundWave)))

FSW = fftSoundWave # Processing Sound Wave

FSW[fAL:fAH] = FSW[fAL:fAH] * acoef
FSW[len(FSW) - fAH:len(FSW) - fAL] = FSW[len(FSW) - fAH:len(FSW) - fAL] * acoef

FSW[fL:fH] = FSW[fL:fH] * 0.3
FSW[len(FSW) - fH:len(FSW) - fL] = FSW[len(FSW) - fH:len(FSW) - fL] * 0.3

'''
# ----------
fLow=3000
fHigh=4000
fAmplifyL = 5000
fAmplifyH = 6000
acoef=2

fL = int(np.round((fLow / fs ) * len(fftSoundWave))) # CHECK
fH = int(np.round((fHigh / fs ) * len(fftSoundWave))) # CHECk

fAL = int(np.round((fAmplifyL / fs ) * len(fftSoundWave)))
fAH = int(np.round((fAmplifyH / fs ) * len(fftSoundWave)))

FSW = fftSoundWave # Processing Sound Wave

FSW[fAL:fAH] = FSW[fAL:fAH] * acoef
FSW[len(FSW) - fAH:len(FSW) - fAL] = FSW[len(FSW) - fAH:len(FSW) - fAL] * acoef

FSW[fL:fH] = FSW[fL:fH] * 0.05
FSW[len(FSW) - fH:len(FSW) - fL] = FSW[len(FSW) - fH:len(FSW) - fL] * 0.05


# ------------
'''


plt.figure(4)
plt.subplot(211)
plt.plot(fHalf, abs(FSW[:len(FSW) // 2]))
timeFilteredSoundWave = np.fft.ifft(FSW)
timeFilteredSoundWave = np.real(timeFilteredSoundWave)
plt.subplot(212)
plt.plot(fHalf, 20*np.log10(abs(FSW[:len(FSW) // 2])))
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




