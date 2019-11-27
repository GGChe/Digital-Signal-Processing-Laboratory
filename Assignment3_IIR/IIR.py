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


# ------------ REAL TIME FILTERING
class IIR_filter:
    def __init__(self, _a, _b):
        q_scalingFactor = 15
        self.a0 = _a[0]
        self.a1 = _a[1]
        self.a2 = _a[2]
        self.b0 = _b[0]
        self.b1 = _b[1]
        self.b2 = _b[2]
        # Delay Lines
        self.m_x1 = 0
        self.m_x2 = 0
        self.m_y1 = 0
        self.m_y2 = 0

    def filter(self, x):

        output = 0
        output = self.b0*x + self.b1*self.m_x1+self.b2 * self.m_x2 - self.a1 * self.m_y1 - self.a2 * self.m_y2;
        # Update delay lines
        self.m_x2 = self.m_x1
        self.m_x1 = x
        self.m_y2 = self.m_y1
        self.m_y1 = output
        return output


# Butterworth Filters
# order of filter
n = 2

# cutoff frequency
fc = 15
fn = (fc / fs)*2

# Coefficients
b, a = signal.butter(n, fn, 'low')

# Frequency response
w, h = signal.freqz(b, a)
h1 = 20 * np.log10(np.abs(h))
wc = w / np.pi / 2

# Plotting Frequency
plt.figure(4)
plt.plot(wc, h1, 'g')
plt.xscale('log')
plt.title('Butterworth Lowpass Filter')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Amplitude (dB)')

# Calling the filter class
myFilter = IIR_filter(a, b)

# Array of Zeros
y = np.zeros(len(data))

for i in range(len(data)):
    y[i] = myFilter.filter(data[i])

y = np.real(y)


plt.figure(1)
plt.subplot(212)
plt.plot(y)

plt.figure(2)
plt.subplot(212)
fft_y = np.fft.fft(y)
plt.plot(f_axis[:int(len(f_axis) / 2)], 20 * np.log10(np.abs(fft_y[:int(len(fft_y) / 2)])))




plt.show()
