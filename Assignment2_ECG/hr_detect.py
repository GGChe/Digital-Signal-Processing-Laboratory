import numpy as np
import matplotlib.pylab as plt

# Initialise the script
dataFromECG = np.loadtxt("Gabriel_Brea_norm.dat")
time = dataFromECG[10000:20000, 0]
myECG = dataFromECG[10000:20000, 1]
fs = 1000  # 1000 Hz sampling rate

plt.figure(1)
plt.plot(time, myECG)


class FIR_filter(object):
    def __init__(self, h):
        self.Filter = h
        self.Buffer = np.zeros(len(h))

    def dofilter(self, v):
        resultFIR = 0
        for j in range(len(self.Buffer) - 1, 0, -1):
            self.Buffer[j] = self.Buffer[j - 1]  # buffer is used here to push input
        self.Buffer[0] = v
        for j in range(len(self.Buffer)):
            resultFIR += self.Filter[j] * self.Buffer[j]
        return resultFIR


"""
The class HR_detect inheritates the class FIR_filter. Thus, we can use the functions and class attributes for the 
inherited class as HR_detect is an extension of FIR_filter.
"""
class matched_filter(FIR_filter):
    """
    :parameter coefffir: coefficients of the FIR filter that is going to be applied to the signal.
    """
    def __init__(self, coeff_fir):
        h = FIR_filter(coeff_fir)
        self.ECG_processed = np.zeros(len(myECG))
        for j in range(len(myECG)):
            self.ECG_processed[j] = h.dofilter(myECG[j])

        """
        :parameter template: is the template which is desired to be calculated from the signal. 
        """
    def detection(self, mytemplate):
        fir_coeff = mytemplate[::-1]
        detected_array = np.zeros(len(myECG))
        templateFIR = FIR_filter(fir_coeff)
        for j in range(len(time)):
            detected_array[j] = templateFIR.dofilter(self.ECG_processed[j])
        detected_output = detected_array * detected_array
        return detected_output

class TemplateMaker(object):
    def __init__(self):
        self

    def mexicanhat(self):
        t = np.linspace(-250, 250, 500)
        mytemplate = (2/np.sqrt(3*35)*np.pi**(1/4)) * (1 - (t**2/35**2)) * np.exp((-t**2)/(2*35**2))
        return mytemplate
    def gaussian(self):
        t = np.linspace(-250, 250, 500)
        myTemplate = -t * np.exp((-t**2)/50) / (125 * np.sqrt(2*np.pi))
        return myTemplate



# Define the frequencies for the FIR filter
f0 = 1
f1 = 45
f2 = 55
ntaps = 200

FIRfrequencyResponse = np.ones(ntaps)
# Limits for the filtering
k0 = int((f0 / fs) * ntaps)
k1 = int((f1 / fs) * ntaps)
k2 = int((f2 / fs) * ntaps)

FIRfrequencyResponse[k1:k2 + 1] = 0
FIRfrequencyResponse[ntaps - k2:ntaps - k1 + 1] = 0
FIRfrequencyResponse[0:k0 + 1] = 0
FIRfrequencyResponse[ntaps - k0:ntaps] = 0

timeFIR = np.fft.ifft(FIRfrequencyResponse)
h_real = np.real(timeFIR)
FIR_shifted = np.zeros(ntaps)
FIR_shifted[0:int(ntaps / 2)] = h_real[int(ntaps / 2):ntaps]
FIR_shifted[int(ntaps / 2):ntaps] = h_real[0:int(ntaps / 2)]


myTemplate = TemplateMaker()
template = myTemplate.gaussian()
plt.plot(template)
detectionOfHeartBeat = matched_filter(FIR_shifted)
det2 = detectionOfHeartBeat.detection(template)

plt.figure(2)
plt.plot(time, det2)
plt.show()









