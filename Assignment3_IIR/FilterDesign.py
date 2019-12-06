import numpy as np
import matplotlib.pylab as plt
import scipy.signal as signal
plt.rcParams.update({'font.size': 18})

# Initialization
data = np.loadtxt('IIR_pulse_read1.dat')
fs = 100

# Time domain plot
plt.figure(1)
plt.title("Time Domain")
plt.subplot(211)
plt.plot(data)
plt.xlabel("Time (s)")
plt.ylabel("Magnitude")
plt.title("Time Domain Representation of the Signal")

fft_data = np.fft.fft(data)
f_axis = np.linspace(0, fs, len(data))

plt.figure(2)
plt.subplot(211)
plt.title("Frequency Spectrum of the Signal")
plt.plot(f_axis[:int(len(f_axis) / 2)], 20*np.log10(np.abs(fft_data[:int(len(fft_data)/2)])))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

# cutoffs   d
f = (10/fs) * 2

# ------------ FILTER SECTION -----------------
class IIR(object):
    """
    Given the proper parameters, this class calculates a filter (Butterworth, Chebyshev1 or Chebyshev2) and process an
    input value from the reading.

    Attributes:
        @:param order: Can be odd or even order as this class creates an IIR filter through the chain of second order filters and
        an extra first order at the end if odd order is required.
        @:param cutoff: For Lowpass and Highpass filters only one cutoff frequency is required while for Bandpass and Bandstop
        it is required an array of frequencies. The input values must be float or integer and the class will
        normalise them to the Nyquist frequency.
        @:param filterType: lowpass, highpass, bandpass, bandstop
        @:param design: butter, cheby1, cheby2.
        @:param rp: Only for cheby1, it defines the maximum allowed passband ripples in decibels.
        @:param rs: Only for cheby2, it defines the minimum required stopband attenuation in decibels.
    """
    def __init__(self, order, cutoff, filterType, design='butter', rp=1, rs=1):
        for i in range(len(cutoff)):
            cutoff[i] = cutoff[i] / fs * 2
        if design == 'butter':
            self.coefficients = signal.butter(order, cutoff, filterType, output='sos')
        elif design == 'cheby1':
            self.coefficients = signal.cheby1(order, rp, cutoff, filterType, output='sos')
        elif design == 'cheby2':
            self.coefficients = signal.cheby2(order, rs, cutoff, filterType, output='sos')
        self.acc_input = np.zeros(len(self.coefficients))
        self.acc_output = np.zeros(len(self.coefficients))
        self.buffer1 = np.zeros(len(self.coefficients))
        self.buffer2 = np.zeros(len(self.coefficients))
        self.input = 0
        self.output = 0

    def filter(self, input):
        """
        From the coefficients calculated in the constructor of the class, the filter is created as chains of IIR filters
        to obtain any order IIR filter. This is important as if order 8 IIR filter is required, it can be calculated as
        a chain of 4 2nd order IIR filters.
        :param input: input value from the reading in real time to be processed.
        :return: processed value.
        """
        self.input = input
        self.output = 0

        """ 
        This loop creates  any order filter by concatenating second order filters.
        If it is needed a 8th order filter, the loop will be executed 4 times obtaining
        a chain of 4 2nd order filters.
        """
        for i in range(len(self.coefficients)):
            self.FIRcoeff = self.coefficients[i][0:3]
            self.IIRcoeff = self.coefficients[i][3:6]

            """
            IIR Part of the filter:
            The accumulated input are the values of the IIR coefficients multiplied
            by the variables of the filter: the input and the delay lines.
            """
            self.acc_input[i] = (self.input + self.buffer1[i]
                                 * -self.IIRcoeff[1] + self.buffer2[i] * -self.IIRcoeff[2])

            """
            FIR Part of the filter:
            The accumulated output are the values of the FIR coefficients multiplied
            by the variables of the filter: the input and the delay lines.
            """
            self.acc_output[i] = (self.acc_input[i] * self.FIRcoeff[0]
                                  + self.buffer1[i] * self.FIRcoeff[1] + self.buffer2[i]
                                  * self.FIRcoeff[2])

            # Shifting the values on the delay line: acc_input->buffer1->buffer2
            self.buffer2[i] = self.buffer1[i]
            self.buffer1[i] = self.acc_input[i]
            self.input = self.acc_output[i]

        self.output = self.acc_output[i]
        return self.output


n = 2  # Order of the filter
cutoffFreq = [0.8, 4]
norm_cutoff = np.zeros(len(cutoffFreq))
for i in range(len(cutoffFreq)):
    norm_cutoff[i] = cutoffFreq[i] / fs * 2

myFilter = IIR(n, [0.8, 4], 'bandpass', design='butter')

coeff = signal.butter(n, norm_cutoff, 'bandpass', output='sos')
w, h = signal.freqz(coeff[1], coeff[0])

plt.figure(3)
plt.plot(w/np.pi/2*fs, 20*np.log(np.abs(h)))
plt.xlabel('frequency/Hz')
plt.ylabel('gain/dB')
plt.savefig('IIRDesign_Filter.eps', format='eps')

y = np.zeros(len(data))
for i in range(len(data)):
    y[i] = myFilter.filter(data[i])

plt.figure(1)
plt.subplot(212)
plt.plot(y)
plt.savefig('IIRDesign_Fig_1.eps', format='eps')

plt.figure(2)
plt.subplot(212)
fft_y = np.fft.fft(y)
plt.plot(f_axis[:int(len(f_axis) / 2)], 20*np.log10(np.abs(fft_y[:int(len(fft_y)/2)])))
plt.savefig('IIRDesign_Fig_2.eps', format='eps')

plt.figure(5)
plt.plot(f_axis[:int(len(f_axis) / 2)], 20*np.log10(np.abs(fft_data[:int(len(fft_data)/2)])))
plt.title("Frequency Spectrum of the signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.savefig('IIRDesign_Fig_5.eps', format='eps')


plt.show()
