import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

fig_path = '/home/tosi/Pictures/ThesisPics/'



t = np.linspace(0, 1120, 1120)
f0 = 10

x = np.exp(1j*2*np.pi*f0*t)

plt.figure("continous signal")
plt.plot(t,x)

n_samples = len(t)
Tc = 1/n_samples

n_support = 2048
fsupport = np.arange(n_support) - n_support/2

#gen periodic rectangular function of normalized period 0.8

rect = np.ones_like(t)
chunk_size = 140
zeros = 36
for i in range(0, len(rect), chunk_size):
    rect[i + chunk_size - zeros:i+chunk_size] = 0

plt.figure("rect function")
plt.plot(t, rect)
plt.title("TDD pattern windowing function", fontsize=16)
plt.xlabel("OFDM symbol", fontsize=16)
plt.savefig(fig_path+'rectFunct.eps', format='eps')
plt.show()

plt.figure("transform of continous signal")
plt.plot(fsupport,np.abs(sp.fft.fftshift(sp.fft.fft(rect, n_support))))
plt.show()

plt.figure("transform of windowed signal")
plt.plot(fsupport,sp.fft.fftshift(sp.fft.fft(np.multiply(x,rect
                                                         ), n_support)))
plt.show()


print()