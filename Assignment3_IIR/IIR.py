import numpy as np
import pylab as pl


# 2nd order IIR filter
class IIR_filter:
    def __init__(self, _b0, _b1, _a1, _a2):
        self.a1 = _a1
        self.a2 = _a2
        self.b0 = _b0
        self.b1 = _b1
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
f = 0.1

# Q factor
q = 10

# s infinite as defined for a 2nd order resonator (see impulse invar)
si = np.complex(-np.pi * f / q, np.pi * f * np.sqrt(1 / (q ** 2)))

# Coefficients
b0 = 1
b1 = -1
a1 = np.real(-(np.exp(si)+np.exp(np.conjugate(si))))
a2 = np.exp(2*np.real(si))

f = IIR_filter(b0, b1, a1, a2)

x = np.zeros(100)
x[10] = 1
y = np.zeros(100)

for i in range(len(x)):
    y[i] = f.filter(x[i])

pl.plot(y)

pl.show()


