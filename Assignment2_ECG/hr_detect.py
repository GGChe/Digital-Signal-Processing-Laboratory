import numpy as np
import matplotlib.pylab as plt
from scipy.signal import find_peaks

# Initialise the script
dataFromECG = np.loadtxt("Gabriel_Brea_norm.dat")
time = dataFromECG[:, 0]
myECG = dataFromECG[:, 1]
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


class TemplateMaker:
    def __init__(self):
        self

    def mexicanhat(self):
        t = np.linspace(-250, 250, 500)
        mytemplate = (2 / np.sqrt(3 * 35) * np.pi ** (1 / 4)) * (1 - (t ** 2 / 35 ** 2)) * np.exp(
            (-t ** 2) / (2 * 35 ** 2))
        return mytemplate

    def gaussian(self):
        t = np.linspace(-250, 250, 500)
        mytemplate = -t * np.exp((-t ** 2) / 50) / (125 * np.sqrt(2 * np.pi))
        return mytemplate


class detectBeats:
    def __init__(self, inputECG):
        self.ecg = inputECG
        self.threshold = np.max(inputECG) * 0.01

        """
        :parameter No input parameters
        :return this function returns: 1) An array of zeros with the length of the inputECG with ones on the peaks and
        2) An integer with the number of beats in the ECG provided.
        """

    def dodetectBeats(self):
        peaksList = np.zeros(len(self.ecg))
        for i in range(len(self.ecg)):
            if self.ecg[i] > self.threshold:
                peaksList[i] = 1
        peaks, _ = find_peaks(peaksList, distance=150)
        numberOfPeaks = len(peaks)
        return peaksList, numberOfPeaks


class MomentaryHeartRateDetector:

    def __init__(self, inputList):
        self.myList = inputList

    def MHRdetect(self):
        listOfBeats = self.myList  # Output from Matched filter
        BPM = []  # It will be the array of Peaks
        aux = 0  # auxiliary counter
        threshold = 0.5
        for i in range(len(listOfBeats)):
            if listOfBeats[i] > threshold:
                dt = (i - aux)  # difference in time in second
                aux = i
                bpm = 1 / dt * 60000  # BPS to BPM
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

myTemplate = TemplateMaker()
template = myTemplate.gaussian()
plt.plot(template)
detectionOfHeartBeat = matched_filter(FIR_shifted)
det2 = detectionOfHeartBeat.detection(template)

beatDetect = detectBeats(det2)
listOfDetectedBeats, peaksOfHeartBeats = beatDetect.dodetectBeats()

MomentaryHeartRate = MomentaryHeartRateDetector(listOfDetectedBeats)
MHR = MomentaryHeartRate.MHRdetect()

plt.figure(1)
plt.subplot(212)
plt.plot(time, det2)

plt.figure(3)
plt.plot(listOfDetectedBeats)

plt.figure(4)
plt.plot(MHR)

plt.show()
