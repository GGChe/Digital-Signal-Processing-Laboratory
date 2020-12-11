# Digital Signal Processing
This is the laboratory repository for the projects of Digital Signal Processing carried out at the University of Glasgow, Scotland, 2019.

Check the Youtube video here! 
https://www.youtube.com/watch?v=2GbPQE1FDxk&feature=youtu.be

## First Project: Audio processing

In this first project, a sentence is recorded with an standard microphone and slightly improved by just removing parts of the frequency spectrum. 

![Figure 1](https://github.com/GGChe/DigitalSignalProcessing/blob/master/Pictures/Figure1.png)


## Second Project: Electrocardiogram (ECG) processing and heartbeat detection

In this project an ECG signal was recorded and filtered by removing DC component and 50 Hz noise usinf an FIR filter. Besides, the R waves are detected using 4 different templantes in match filtering. Gaussians (zero and one order derivative), mexican wavelet analytically tuned waves were used as templates for detection.


![Figure 2](https://github.com/GGChe/DigitalSignalProcessing/blob/master/Pictures/Figure2.png)
![Figure 3](https://github.com/GGChe/DigitalSignalProcessing/blob/master/Pictures/Figure3.png)
![Figure 4](https://github.com/GGChe/DigitalSignalProcessing/blob/master/Pictures/Figure4.png)




## Third Project: Real time filtering of heart pulse using IIR filter

A physical system was fabricated using a photoresistor to record the blood volume variations in a finger (any finger), also called plethismography. The signal is amplified with an OpAmp 101 times and sent to arduino where the signal is processed with a sampling frequency of 100 Hz. 

![Figure5](https://github.com/GGChe/DigitalSignalProcessing/blob/master/Pictures/IIRDesign_Fig_1.png)
![Figure6](https://github.com/GGChe/DigitalSignalProcessing/blob/master/Pictures/IIRDesign_Fig_2.png)


