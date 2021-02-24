import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np 

#If using termux
import subprocess
import shlex

#read .wav file
x,fs = sf.read('Sound_Noise.wav') #x is input signal and fs is sampling frequency
order = 4
fc = 4000.0  #cutoff frequency
Wn = 2*fc/fs
nx = len(x)

def H(z,num,den):
	num1 = np.polyval(num,z**(-1))
	den1 = np.polyval(den,z**(-1))
	H = num1/den1
	return H

num,den = signal.butter(order,Wn,'low')
k = np.arange(nx)
w = 2*np.pi*k/nx
z = np.exp(1j * w)	
Hz = H(z,num,den)


X = np.fft.fft(x)
Y = np.multiply(Hz,X)
y = np.fft.ifft(Y).real #output signal


sf.write('Sound_With_ReducedNoise_7.1.wav',y,fs)


#verification
y = signal.filtfilt(num,den,x)
sf.write('Sound_With_ReducedNoise_ver.wav',y,fs)

#plots
plt.figure(1)
plt.figure(figsize=(8,7))
plt.subplot(2,1,1)
plt.plot(y,'c')
plt.title('output with own routine')
plt.grid()

plt.subplot(2,1,2)
plt.plot(y,'m')
plt.title('output with built-in ')
plt.grid()

plt.savefig('../figs/ee18btech11033_1.eps')

plt.figure(2)
plt.figure(figsize=(8,7))
plt.subplot(2,1,1)
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(y))),'c')
plt.title('output with own routine')
plt.grid()

plt.subplot(2,1,2)
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(y))),'m')
plt.title('output with built-in ')
plt.grid()

plt.savefig('../figs/ee18btech11033_2.eps')
