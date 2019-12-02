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
plt.plot(f_axis[:int(len(f_axis) / 2)], 20*np.log10(np.abs(fft_data[:int(len(fft_data)/2)])))

# cutoffs   d
f = (10/fs) * 2

# ------------ FILTER SECTION -----------------
class IIR2Filter(object):

    def createCoeffs(self, order, cutoff, filterType, design='butter', rp=1, rs=1, fs=0):

        self.coefficients = [0]
        # cutoff frequencies need to be normalised to Nyquist
        for i in range(len(cutoff)):
            cutoff[i] = cutoff[i] / fs * 2
        if design == 'butter':
            self.coefficients = signal.butter(order, cutoff, filterType, output='sos')
        elif design == 'cheby1':
            self.coefficients = signal.cheby1(order, rp, cutoff, filterType, output='sos')
        elif design == 'cheby2':
            self.coefficients = signal.cheby2(order, rs, cutoff, filterType, output='sos')

        return self.coefficients

    def __init__(self, order, cutoff, filterType, design='butter', rp=1, rs=1):
        self.coefficients = self.createCoeffs(order, cutoff, filterType, design, rp, rs, fs)
        self.acc_input = np.zeros(len(self.coefficients))
        self.acc_output = np.zeros(len(self.coefficients))
        self.buffer1 = np.zeros(len(self.coefficients))
        self.buffer2 = np.zeros(len(self.coefficients))
        self.input = 0
        self.output = 0

    def filter(self, input):
        self.input = input
        self.output = 0

        # In case that we need a filter with order more than 3, the filter is calculatd as
        # product of successive filters.
        for i in range(len(self.coefficients)):
            self.FIRcoeff = self.coefficients[i][0:3]
            self.IIRcoeff = self.coefficients[i][3:6]

            # Calculate accumulated input from the input and  the values coming as IIR Coeffs
            self.acc_input[i] = (self.input + self.buffer1[i]
                                * -self.IIRcoeff[1] + self.buffer2[i] * -self.IIRcoeff[2])

            # Calculate accumulated input from the input and  the values coming as FIR Coeffs
            self.acc_output[i] = (self.acc_input[i] * self.FIRcoeff[0]
                                    + self.buffer1[i] * self.FIRcoeff[1] + self.buffer2[i]
                                  * self.FIRcoeff[2])

            # Shifting the values on the delay line: acc_input->buffer1->buffer2
            self.buffer2[i] = self.buffer1[i]
            self.buffer1[i] = self.acc_input[i]
            self.input = self.acc_output[i]

            self.output = self.acc_output[i]

            return self.output


myFilter = IIR2Filter(2, [1, 8], 'bandpass', design='butter')
coeff = myFilter.coefficients
w, h = signal.freqz(coeff[1], coeff[0])

plt.figure(3)
plt.plot(w/np.pi/2*fs, 20*np.log(np.abs(h)))
plt.xlabel('frequency/Hz')
plt.ylabel('gain/dB')

y = np.zeros(len(data))
for i in range(len(data)):
    y[i] = myFilter.filter(data[i])

plt.figure(1)
plt.subplot(212)
plt.plot(y)

plt.figure(2)
plt.subplot(212)
fft_y = np.fft.fft(y)
plt.plot(f_axis[:int(len(f_axis) / 2)], 20*np.log10(np.abs(fft_y[:int(len(fft_y)/2)])))

plt.show()
