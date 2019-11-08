


import numpy as np
import ecg_filter as fir

fs = 1000
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

myFIR = fir.FIR_filter(FIR_shifted)
print(myFIR.dofilter(1))





