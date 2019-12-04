#!/usr/bin/python3
"""

Authors: Gabriel Galeote-Checa & Anton Saikia


---------------------------------- Documentation ----------------------------------

This script is a Digital Signal Processing of light-based pulsemeter device consisting of a photoresistor and a LED
One finger is placed between the LED and the photoresistor and by light attenuation due to different blood pressures
related to the heartbeat, we can read a periodic signal. There are some noises in the signal due to finger moving,
noise due to the amplification, and others.

The process of the script is simple:
Read sensor -> Filter -> Plot

For the filtering of the signal, an IIR filter was implemented in the class IIR:
IIR(order,cutoff,filterType,design='butter',rp=1,rs=1,fs=0)

This class calculate the IIR filter coefficients for a given filter type ( Butter, Cheby1 or Cheby2)
and then, the class function "filter" does the filtering to a given value.
---------------------------------------------------------------------------
"""

import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from pyfirmata2 import Arduino
import scipy.signal as signal

""" ---- Initialization of the script ------
|
|   fs : sampling frequency, defined for every application and limited by the system specifications.
|   PORT : communication port, detected automatically so we don't have to care about the specific COM port.
|   app : global QT application object for plotting. 
|   running = signals to all threads in endless loops that we'd like to run these
|
"""
fs = 100
PORT = Arduino.AUTODETECT
app = QtGui.QApplication(sys.argv)
running = True



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


myFilter = IIR(2, [0.8, 4], 'bandpass', design='butter')

# Let's create two instances of plot windows
qtPlot1 = QtPanningPlot("Arduino 1st channel")
qtPlot2 = QtPanningPlot("Arduino 2nd channel")

# sampling rate: 100Hz
samplingRate = 100

# called for every new sample which has arrived from the Arduino
def callBack(data):
    # send the sample to the plotwindow
    qtPlot1.addData(data)
    ch1 = board.analog[1].read()
    # 1st sample of 2nd channel might arrive later so need to check
    if ch1:
        filteredData = myFilter.filter(ch1)
        qtPlot2.addData(filteredData*10)


# Get the Arduino board.
board = Arduino(PORT)

# Set the sampling rate in the Arduino
board.samplingOn(1000 / samplingRate)

# Register the callback which adds the data to the animated plot
board.analog[0].register_callback(callBack)

# Enable the callback
board.analog[0].enable_reporting()
board.analog[1].enable_reporting()

# showing all the windows
app.exec_()

# needs to be called to close the serial port
board.exit()

print("Finished")