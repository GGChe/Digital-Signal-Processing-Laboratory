import numpy as np
import matplotlib.pylab as plt
import scipy.signal as signal

data = np.loadtxt('IIR_pulse_read1.dat')

fs = 100
data_fft = np.fft.fft(data)
f_axis = np.linspace(0, fs, len(data))  # Full spectrum frequency range
plt.figure(1)
plt.subplot(211)
plt.plot(f_axis[:int(len(f_axis) / 2)], 20 * np.log10(abs(data_fft[:int(len(data_fft) / 2)])))

# 2nd order IIR filter
class IIR_filter:
    def __init__(self, _b0, _b1, _a1, _a2):
        self.a1 = _a1
        self.a2 = _a2
        self.b0 = _b0
        self.b1 = _b1
        self.b2 = 0
        self.buffer1 = 0
        self.buffer2 = 0

    def filter(self, x):
        acc_input = x - self.buffer1 * self.a1 - self.buffer2 * self.a2
        acc_output = acc_input * self.b0 + self.buffer1 * self.b1 + + self.buffer2*self.b2
        self.buffer2 = self.buffer1
        self.buffer1 = acc_input
        return acc_output


# Normalised frequency 0.1
# T = 1
f = 20

# Q factor
q = 10

# s infinite as defined for a 2nd order resonator (see impulse invar)
si = np.complex(-np.pi * f / q, np.pi * f * np.sqrt(1 / (q ** 2)))

# Calculate Coefficients
b, a = signal.butter(2, 0.1*2)
b0 = b[0]
b1 = [1]
a1 = a[1]
a2 = a[2]
f = IIR_filter(b0, b1, a1, a2)

plt.plot(f)


"""double b2 = 0;
for(int i=0;i<1000000;i++) 
{
	float a=0;
	if (i==10) a = 1;
	b2 = b;
    b = f.filter(a);
	assert_print(!isnan(b),"Lowpass output is NAN\n");
	if ((i>20) && (i<100))
		assert_print((b != 0) || (b2 != 0),
			     "Lowpass output is zero\n");
}
""" 
x = np.zeros(100)
x[10] = 1
y = []

for i in range(len(x)):
    y[i] = f.filter(x[i])


y_fft = np.fft.fft(y)

print(len(y_fft))
print(len(f_axis))

plt.figure(1)
plt.subplot(212)
plt.plot(f_axis[:int(len(f_axis) / 2)], 20 * np.log10(abs(y_fft[:int(len(y_fft) / 2)])))


plt.figure(3)

# unfiltered
plt.subplot(211)
plt.plot(data)
plt.xlabel('samples')
plt.ylabel('unfiltered/raw ADC units')
# filtered
plt.subplot(212)
plt.plot(y)
plt.xlabel('samples')
plt.ylabel('filtered/raw ADC units')
plt.show()



















