import numpy as np
import matplotlib.pylab as plt
import scipy.signal as signal

# Initialization
data = np.loadtxt('IIR_pulse_read1.dat')
fs = 100

# Time domain plot
plt.figure(1)
plt.subplot(211)
plt.plot(data)
plt.title("Time domain representation of the signal")

fft_data = np.fft.fft(data)
f_axis = np.linspace(0, fs, len(data))

plt.figure(2)
plt.subplot(211)
plt.plot(f_axis[:int(len(f_axis) / 2)], 20 * np.log10(np.abs(fft_data[:int(len(fft_data) / 2)])))

# cutoffs
f = (10 / fs) * 2

# scaling factor in bits
q = 14
# scaling factor as factor...
scaling_factor = 2 ** q

# let's generate a sequence of 2nd order IIR filters
sos = signal.butter(2, f, 'low', output='sos')
sos = np.round(sos * scaling_factor)

# plot the frequency response
b, a = signal.sos2tf(sos)
w, h = signal.freqz(b, a)
plt.figure(3)
plt.plot(w / np.pi / 2 * fs, 20 * np.log(np.abs(h)))
plt.xlabel('frequency/Hz')
plt.ylabel('gain/dB')


# ------------ REAL TIME FILTERING

class IIR_filter:
    def __init__(self, _a, _b):
        self.a0 = _a[0]
        self.a1 = _a[1]
        self.a2 = _a[2]

        self.b0 = _b[0]
        self.b1 = _b[1]
        self.b2 = _b[2]

    def filter(self, x):
        buffer1_in = 0
        buffer2_in = 0
        buffer1_out = 0
        buffer2_out = 0
        acc = 0

        """FIR Part"""
        acc = x * self.b0 + buffer1_in * self.b1 + buffer2_in * self.b2

        """IIR Part"""
        acc = acc - buffer1_out * self.a1 - buffer2_out * self.a2
        buffer2_in = buffer1_in
        buffer1_in = x
        buffer2_out = buffer1_out
        buffer1_out = acc

        return acc

# ----------------------------------
# y = signal.lfilter(b, a, data)
y = np.zeros(len(data))
h = IIR_filter(a, b)
for i in range(len(data)):
    y[i] = h.filter(data[i])

y = np.real(y)

plt.figure(1)
plt.subplot(212)
plt.plot(y)

plt.figure(2)
plt.subplot(212)
fft_y = np.fft.fft(y)
plt.plot(f_axis[:int(len(f_axis) / 2)], 20 * np.log10(np.abs(fft_y[:int(len(fft_y) / 2)])))

plt.show()
