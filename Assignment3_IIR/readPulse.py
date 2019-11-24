import numpy as np
import matplotlib.pylab as plt


data = np.loadtxt("IIR_pulse_read1.dat")
print(data)
plt.plot(data)
plt.show()