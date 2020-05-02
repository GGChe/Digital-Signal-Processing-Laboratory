# !/usr/bin/python3
"""
Authors: Gabriel Galeote-Checa & Anton Saikia

Documentation:

This script is for the digital signal processing of a light-based pulsemeter device consisting of a photoresistor and an LED.
A finger is placed between the LED and the photoresistor and by light attenuation due to different blood pressures
related to the heartbeat, we can detect a periodic signal.

The process of the script is simple:
Read sensor -> Filter -> Plot

For the filtering of the signal, an IIR filter was implemented in the class IIR:

- IIR2Filter(coefficients) --> Creates a 2nd order IIR filter from a given coefficients.

- IIRFilter(coefficients) --> Creates an IIR filter of any order as a chain of 2nd order IIR filters using the
                            class IIR2filter.
"""

import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from pyfirmata2 import Arduino
import scipy.signal as signal
import ug8lib as lcd

"""
# -------------  Initialization of the Script -------------

    - fs : sampling frequency, defined for every application and limited by the system specifications.         
    - PORT : communication port, detected automatically so we don't have to care about the specific COM port.   
    - app : global QT application object for plotting.                                                        
    - running = signals to all threads in endless loops that we'd like to run these                            

"""
U8GLIB_ST7920_128X64 u8g(13, 11, 12, U8G_PIN_NONE);
fs = 100
PORT = Arduino.AUTODETECT
app = QtGui.QApplication(sys.argv)
running = True
board = Arduino(PORT)       # Get the Arduino board.
board.samplingOn(1000 / fs) # Set the sampling rate in the Arduino

class IIR2Filter(object):
    """
    given a set of coefficients for the IIR filter this class creates an object that keeps the variables needed for the
    IIR filtering as well as creating the function "filter(x)" with filters the input data.

    Attributes:
        @:param coefficients: input coefficients of the IIR filter as an array of 6 elements where the first three
        coefficients are for the FIR filter part of the IIR filter and the last three coefficients are for the IIR part
        of the IIR filter.
    """

    def __init__(self, coefficients):
        self.myCoefficients = coefficients
        self.IIRcoeff = self.myCoefficients[3:6]
        self.FIRcoeff = self.myCoefficients[0:3]
        self.acc_input = 0
        self.acc_output = 0
        self.buffer1 = 0
        self.buffer2 = 0
        self.input = 0
        self.output = 0

    def filter(self, input):
        """
        :param input: input value to be processed.
        :return: processed value.
        """

        self.input = input
        self.output = 0

        """
        IIR Part of the filter:
            The accumulated input are the values of the IIR coefficients multiplied by the variables of the filter: 
            the input and the delay lines.
        """
        self.acc_input = (self.input + self.buffer1
                          * -self.IIRcoeff[1] + self.buffer2 * -self.IIRcoeff[2])

        """
        FIR Part of the filter:     
            The accumulated output are the values of the FIR coefficients multiplied by the variables of the filter: 
            the input and the delay lines. 
        
        """

        self.acc_output = (self.acc_input * self.FIRcoeff[0]
                           + self.buffer1 * self.FIRcoeff[1] + self.buffer2
                           * self.FIRcoeff[2])

        # Shifting the values on the delay line: acc_input->buffer1->buffer2
        self.buffer2 = self.buffer1
        self.buffer1 = self.acc_input
        self.input = self.acc_output
        self.output = self.acc_output
        return self.output


class IIRFilter(object):
    """
    given a set of coefficients for the IIR filter this class creates an object that keeps the variables needed for the
    IIR filtering as well as creating the function "filter(x)" with filters the input data.

    Attributes:
        @:param coefficients: input coefficients of the IIR filter as an array of n arrays where n is the order of the
        filter. The array of coefficients is organised in blocks of 6 coefficients where the first three coefficients
        are for the FIR filter part of the IIR filter and the last three coefficients are for the IIR part of the IIR
        filter.
    """

    def __init__(self, mycoeff):
        self.myCoefficients = mycoeff
        self.acc_input = np.zeros(len(self.myCoefficients))
        self.acc_output = np.zeros(len(self.myCoefficients))
        self.buffer1 = np.zeros(len(self.myCoefficients))
        self.buffer2 = np.zeros(len(self.myCoefficients))
        self.input = 0
        self.output = 0
        self.myIIRs = []

        """
        An IIR filter can be calculated as a chain of 2nd order IIR filters. For that, the array myIIRs contains
        a list of IIR2Filter classes initialised with the coefficients given.
        """
        for i in range(len(self.myCoefficients)):
            self.myIIRs.append(IIR2Filter(self.myCoefficients[i]))

    def filter(self, input):
        """
        Filter and input value with the IIR filter structure.
        :param input: input value from the reading in real time to be processed.
        :return: processed value.
        """

        self.input = input
        self.output = 0
        self.output = self.myIIRs[0].filter(input)
        for i in range(1, len(self.myCoefficients)):
            self.output = self.myIIRs[i].filter(self.output)
        return self.output


class QtPanningPlot:

    def __init__(self, title):
        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle(title)
        self.plt = self.win.addPlot()
        self.plt.setYRange(-1, 1)
        self.plt.setXRange(0, 600)
        self.curve = self.plt.plot()
        self.data = []
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)
        self.layout = QtGui.QGridLayout()
        self.win.setLayout(self.layout)
        self.win.show()

    def update(self):
        self.data = self.data[-500:]
        if self.data:
            self.curve.setData(np.hstack(self.data))

    def addData(self, d):
        self.data.append(d)


"""
|  ----------------------------------------------------------------------  |
|  ----------------------------     MAIN     ----------------------------  |
|  ----------------------------------------------------------------------  |
|  Cutoff frequencies:                                                     |
|          a) wc1 = 0.8 Hz to remove DC components                         |
|          b) wc2 = 4 Hz because the maximum heartrate is                    |
|                220 bpm = 220/60 = 3.67 Hz                                |
|                                                                          |
| Order of the filter:                                                     |    
|           n = 2 for a chain of two 2nd order IIR filter as we            |
|            are using 'sos' for second-order sections                     |  
|                                                                          |  
----------------------------------------------------------------------------
"""

cutoff = [0.8, 4]
order = 1

for i in range(len(cutoff)):
    cutoff[i] = cutoff[i] / fs * 2

coeff = signal.butter(order, cutoff, 'bandpass', output='sos')

# If the order of the filter is 1, is one IIR 2nd order filter otherwise, it is a chain of IIR filters.
if order > 1:
    myFilter = IIRFilter(coeff)
else:
    myFilter = IIR2Filter(coeff[0])

# Create two instances of Qt plots
qtPlot1 = QtPanningPlot("Arduino 1st channel")
qtPlot2 = QtPanningPlot("Arduino 2nd channel")

# This function is called for every new sample which has arrived from the Arduino
def callBack(data):
    qtPlot1.addData(data)
    ch1 = board.analog[1].read()
    if ch1:
        filteredData = myFilter.filter(ch1)
        qtPlot2.addData(filteredData * 10)


# Register the callback which adds the data to the animated plot
board.analog[0].register_callback(callBack)

# Enable the callback
board.analog[0].enable_reporting()
board.analog[1].enable_reporting()

# Show all windows
app.exec_()

# Close the serial port
board.exit()

print("Execution Finished")
