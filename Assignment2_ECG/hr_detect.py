"""
---------------------------------------------------------
Code for the Assignment 1 of Digital Signal Processing
Authors: Gabriel Galeote-Checa & Anton Saikia
---------------------------------------------------------
Second part of the ECG filtering that consists of Heart Rate Detection by using matched filters.
"""
import numpy as np
import matplotlib.pylab as plt

# Initialise the script
dataFromECG = np.loadtxt("ecg2.dat")
time = dataFromECG[10000:40000, 0]
myECG = dataFromECG[10000:40000, 1]
fs = 1000  # 1000 Hz sampling rate
ntaps = 500

# -- Plot ECG in time domain -- #
plt.figure(1)
plt.subplot(211)
plt.plot(myECG)
plt.title("(TOP) ECG Pre-filtered; (BOTTOM) ECG Filtered with Match Filter")
plt.ylabel('Amplitude')

"""
---------- FIR FILTER ----------
 
FIR_Filter class that applies the FIR filter to an input signal.

This class calculate the result of the FIR filter for a given value. The class function dofilter(input) 
introduces the given value of the signal in the buffer in the current position after a proper management of the 
buffer shifting. Then, it is calculated the mathematical result of FIR filter of the buffer storaged that was 
previously shifted to put in the first position the current input value. 
"""


class FIR_filter:
    def __init__(self, inpurFIR):
        self.offset = 0
        self.P = 0
        self.coeval = 0
        self.Buffer = np.zeros(ntaps)
        self.myFIR = inpurFIR

    def dofilter(self, v):

        ResultFIR = 0
        self.CurrentBufferValue = self.P + self.offset
        self.Buffer[self.CurrentBufferValue] = v

        while self.CurrentBufferValue >= self.P:
            ResultFIR += self.Buffer[self.CurrentBufferValue] * self.myFIR[self.coeval]
            self.CurrentBufferValue -= 1
            self.coeval += 1

        self.CurrentBufferValue = self.P + ntaps - 1

        while self.coeval < ntaps:
            ResultFIR += self.Buffer[self.CurrentBufferValue] * self.myFIR[self.coeval]
            self.CurrentBufferValue -= 1
            self.coeval += 1

        self.offset += 1

        if self.offset >= ntaps:
            self.offset = 0

        self.coeval = 0
        return ResultFIR


"""
---------- MATCH FILTER ----------

Inherited from FIR_filter, it receives the class attributes, functions and methods from the superclass.
This class calculates match filtering of an input signal.
"""


class matched_filter(FIR_filter):
    def __init__(self, inputecg):
        self.inputECG = inputecg

    def detection(self, mytemplate):
        fir_coeff = mytemplate[::-1]
        y = np.zeros(len(myECG))
        templateFIR = FIR_filter(fir_coeff)
        for i in range(len(myECG)):
            y[i] = templateFIR.dofilter(self.inputECG[i])
        detected_output = y * y  # The signal is squared to improve the output
        return detected_output


"""
---------- TEMPLATE MAKER ----------

This class has different methods for calculating different templates for the match filter.
"""


class TemplateMaker:
    def __init__(self):
        self

    def mexicanhat(self):
        t = np.linspace(-250, 250, 500)
        mytemplate = 0.9 * (2 / np.sqrt(3 * 17) * np.pi ** (1 / 4)) * (1 - (t ** 2 / 17 ** 2)) \
                     * np.exp((-t ** 2) / (2 * 17 ** 2))
        return mytemplate

    def gaussian1OD(self):
        t = np.linspace(-250, 250, 500)
        mytemplate = -3.25 * t * np.exp((-t ** 2) / 50) / (125 * np.sqrt(2 * np.pi))
        return mytemplate

    def gaussian(self):
        t = np.linspace(-250, 250, 500)
        mytemplate = 3.75 * np.exp((-t ** 2) / 50) / (5 * np.sqrt(2 * np.pi))
        return mytemplate

    def shannon(self):
        t = np.linspace(-250, 250, 500)
        mytemplate = 15 * np.sqrt(100) * np.sinc(100 * t) * np.exp(2 * 1j * t * np.pi * 4)
        return mytemplate


"""
---------- MOMENTARY HEART RATE DETECTOR ----------

MomentaryHeartRateDetector class: From an input ECG signal, the method MRHdetect calculates the momentary heart rate  
of the ECG signal. 
"""
class MomentaryHeartRateDetector:
    def __init__(self, inputlist):
        self.myList = inputlist

    def MHRdetect(self):
        listOfBeats = self.myList  # Output from Matched filter
        BPM = []  # It will be the array of Peaks
        aux = 0  # auxiliary counter
        threshold = max(self.myList) * 0.05
        for i in range(len(listOfBeats)):
            if listOfBeats[i] > threshold:
                differenceTime = (i - aux)  # difference of time in second
                aux = i
                bpm = 1 / differenceTime * (60 * fs)
                if 200 > bpm > 40:  # Limits for the detection of momentary heart rate
                    BPM.append(bpm)  # Add this peak to the BPM array
        BPM = np.delete(BPM, 0)
        return BPM


# Define the frequencies for the FIR filter
f0 = 1
f1 = 45
f2 = 55

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

""" 
----- MATCHED FILTER ----- 
For the matching filter 4 templates will be created and tested: 
1) Gaussian
2) Gaussian Derivative  1st Order
3) Shannon
4) Mexican Hat
"""
myTemplate = TemplateMaker()  # Create the class TemplateMaker

# Create the templates for every kind of template
gaussian = myTemplate.gaussian()
devgaussian = myTemplate.gaussian1OD()
shannon = myTemplate.shannon()
mexicanHat = myTemplate.mexicanhat()

# Matching Filtering of the signal
detectionOfHeartBeat = matched_filter(ECG_processed)
detgaussian = detectionOfHeartBeat.detection(gaussian)
det1ODgaussian = detectionOfHeartBeat.detection(devgaussian)
detshannon = detectionOfHeartBeat.detection(shannon)
detmexicanHat = detectionOfHeartBeat.detection(mexicanHat)

# Create the classes for the momentary heart beat processing for every template
MomentaryHeartRateGaussian = MomentaryHeartRateDetector(detgaussian)
MomentaryHeartRateGaussian1OD = MomentaryHeartRateDetector(det1ODgaussian)
MomentaryHeartRateShannon = MomentaryHeartRateDetector(detshannon)
MomentaryHeartRateMexicanHat = MomentaryHeartRateDetector(detmexicanHat)

# Calculate Momentary Heart Beat
MHRGaussian = MomentaryHeartRateGaussian.MHRdetect()
MHRGaussian1OD = MomentaryHeartRateGaussian1OD.MHRdetect()
MHRShannon = MomentaryHeartRateShannon.MHRdetect()
MHRMexicanHat = MomentaryHeartRateMexicanHat.MHRdetect()

plt.figure(1)
plt.subplot(212)
plt.plot(det1ODgaussian)
plt.xlabel("Time (s)")
plt.ylim("Amplitude")
plt.savefig('HR_detect_Fig_1.eps', format='eps')

plt.figure(2)
plt.subplot(221)
plt.plot(detgaussian)
plt.title("Gaussian")

plt.subplot(222)
plt.plot(det1ODgaussian)
plt.title("Gaussian 1st Order Derivative")

plt.subplot(223)
plt.plot(detshannon)
plt.title("Shannon")

plt.subplot(224)
plt.plot(detmexicanHat)
plt.title("Mexican Hat")
plt.savefig('HR_detect_Fig_2.eps', format='eps')

plt.figure(3)
plt.subplot(221)
plt.plot(MHRGaussian)
plt.title("Gaussian")
plt.ylim(80, 110)

plt.subplot(222)
plt.plot(MHRGaussian1OD)
plt.title("Gaussian 1st Order Derivative")
plt.ylim(80, 110)

plt.subplot(223)
plt.plot(MHRShannon)
plt.title("Shannon")
plt.ylim(80, 110)

plt.subplot(224)
plt.plot(MHRMexicanHat)
plt.title("Mexican Hat")
plt.ylim(80, 110)
plt.savefig('HR_detect_Fig_3.eps', format='eps')

print("Execution finished!")

plt.tight_layout()
plt.show()
