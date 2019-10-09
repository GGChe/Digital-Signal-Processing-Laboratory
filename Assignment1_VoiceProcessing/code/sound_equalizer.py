#   Code for the Assignment 1 of Digital Signal Processing
#   Authors: Gabriel Galeote-Checa & Anton


import matplotlib.pylab as plt
import numpy as np
import scipy.io.wavfile as wav

fs, soundwave = wav.read ('original_mono.wav')

print(max(soundwave))

# plot time domain
t = np.linspace (0, len(soundwave)/fs, len(soundwave))
plt.plot(t, soundwave)
plt.xlabel('time (s)')
plt.ylabel('amplitude ') # ask in laboratory
# plt.yscale('log')
plt.grid('true')
plt.show()


# Plot Frequency Domain
fftsoundwave = np.fft.fft(soundwave)
f = np.linspace(0, fs, len(fftsoundwave))
plt.plot(f, abs(fftsoundwave))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.grid('TRUE')
plt.show()



















#---------------------------
"""

fm=10
wc=70
wm=10
sampleRate = 44100
duration = 5
x_begin = 0
x_final = 15



def frequencyModulator(fm, wc, wm):
    ""
    :param fm: is the base signal frequency
    :param wc: carrier angular frequency
    :return: an array of n
    ""
    fc = wc / (2 * np.pi)
    df = (2 * np.pi * fc) - fm
    t = np.linspace(x_begin, x_final, duration * sampleRate)  # Creates a vector of time for that periodS
    return np.cos(wc * t + ((df * np.sin(wm * t)) / (fm)))




def rectifier(signal):
    ""
    In order to get a pure function, we need to insulate this function from the inputs and rest of the code.
    A pure function is a function where the return value is only determined by its input values,
    without observable side effects.
    :param signal: The signal that will be rectified
    :return: returns the rectified signal
    ""
    for i in range(len(signal)):
        if signal[i] < 0:
            signal[i] = 0
    return signal


def plot_save(signal1, signal2):
    ""
    :param signal1: input signal
    :param signal2: input signal
    ""
    t = np.linspace(x_begin, x_final, duration * sampleRate)  # Creates a vector of time for that period
    fig = plt.figure()
    ax = plt.subplot(211)
    ax.plot(t, signal1)
    plt.title('Frequency Modulated Signal')
    ax = plt.subplot(212)
    ax.plot(t, signal2)
    plt.title('Frequency Modulated Squared Signal ')
    plt.show()
    fig.savefig('plot.png')
    write('Signal.wav', sampleRate, signal1)
    write('SignalRectified.wav', sampleRate, signal2)


x = frequencyModulator(fm, wc, wm)
y = frequencyModulator(fm, wc, wm)
x_rect = rectifier(y)
plot_save(x, x_rect)


"""














