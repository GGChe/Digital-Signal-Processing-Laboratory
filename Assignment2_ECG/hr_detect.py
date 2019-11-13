import numpy as np
import matplotlib.pylab as plt

# Initialise the script
dataFromECG = np.loadtxt("Gabriel_Brea_norm.dat")
time = dataFromECG[10000:20000, 0]
myECG = dataFromECG[10000:20000, 1]
fs = 1000  # 1000 Hz sampling rate

plt.figure(1)
plt.subplot(211)
plt.plot(myECG)


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

    def __init__(self, inputecg):
        self.inputECG = inputecg

    def detection(self, mytemplate):
        fir_coeff = mytemplate[::-1]
        detected_array = np.zeros(len(myECG))
        templateFIR = FIR_filter(fir_coeff)
        for j in range(len(time)):
            detected_array[j] = templateFIR.dofilter(self.inputECG[j])
        detected_output = detected_array * detected_array
        return detected_output


class TemplateMaker:
    def __init__(self):
        self.t = np.linspace(-250, 250, 500)

    def mexicanhat(self):
        mytemplate = (2 / np.sqrt(3 * 35) * np.pi ** (1 / 4)) * (1 - (self.t ** 2 / 35 ** 2)) * np.exp(
            (-self.t ** 2) / (2 * 35 ** 2))
        return mytemplate

    def gaussian(self):
        mytemplate = -self.t * np.exp((-self.t ** 2) / 50) / (125 * np.sqrt(2 * np.pi))
        return mytemplate


"""
This class calulates the momentary heart rate of an ECG signal provided in the constructor
:parameter inputList: is the input ECG signal preprocessed
"""


class MomentaryHeartRateDetector:

    def __init__(self, inputList):
        self.myList = inputList

    """
    :return It returns the list of the momentary heart rate calculations in the ECG signal.
    """

    def MHRdetect(self):
        listOfBeats = self.myList  # Output from Matched filter
        BPM = []  # It will be the array of Peaks
        aux = 0  # auxiliary counter
        threshold = max(self.myList)*0.1
        for i in range(len(listOfBeats)):
            if listOfBeats[i] > threshold:
                differenceTime = (i - aux)  # difference in time in second
                aux = i
                bpm = 1 / differenceTime * (60 * fs)  # msec to min = BPM
                if 200 > bpm > 30:  # 200bpm to 30bpm as filter
                    BPM.append(bpm)  # append means plus
        BPM = np.delete(BPM, 0)
        return BPM


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

ECG_processed = np.zeros(len(myECG))
FIR = FIR_filter(FIR_shifted)
for j in range(len(myECG)):
    ECG_processed[j] = FIR.dofilter(myECG[j])

# ----- MATCHED FILTER ----- #
myTemplate = TemplateMaker()
template = myTemplate.gaussian()
detectionOfHeartBeat = matched_filter(myECG)
det2 = detectionOfHeartBeat.detection(template)

MomentaryHeartRate = MomentaryHeartRateDetector(det2)
MHR = MomentaryHeartRate.MHRdetect()

plt.figure(1)
plt.subplot(212)
plt.plot(time, det2)

plt.figure(4)
plt.plot(MHR)

plt.show()
