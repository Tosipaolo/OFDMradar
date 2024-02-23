import numpy as np
import  scipy as sp
import matplotlib.pyplot as plt

fft_size = 4096

x = np.linspace(0, 1120, 1120)

funct = np.ones_like(x)


pattern = 140
DL_bins = 104
UL_bins = pattern - DL_bins
num_patterns = 8

rect = np.concatenate((np.ones(DL_bins), np.zeros(UL_bins)))
n_sym_rect = (fft_size - len(rect)) // 2
rect = np.concatenate((np.zeros(n_sym_rect), rect, np.zeros(n_sym_rect)))

for i in range(num_patterns):
    funct[(i * pattern) + DL_bins: (i+1)*(pattern)] = 0

funct = funct[:(1120-UL_bins)]
print(f"Funct len; {len(funct)}")

num_zero_syms = int(2** np.ceil(np.log2(len(funct))) - len(funct)) // 2

plt.figure("DL/UL pattern")
plt.title("DL/UL pattern")
plt.plot(np.linspace(0, len(funct), len(funct)), funct)

funct = np.concatenate((np.zeros(num_zero_syms), funct, np.zeros(num_zero_syms)))

transform = sp.fft.fftshift(sp.fft.fft(sp.fft.fftshift(funct), fft_size))

x = np.linspace(0, len(transform), len(transform)) - len(transform)//2


plt.figure("PS of pattern rectangle function")
plt.title("Power spectrum of Windowing function - dB")
plt.plot(x, 10*np.log10(np.square(np.abs(transform))))

plt.figure("Transform pattern function - no shift")
plt.title("Transform of DL/UL windowing function - absolute value", fontsize=16)
plt.plot(x, np.abs(transform))


plt.figure("Transform pattern function - shifted")
plt.plot(x, sp.fft.fftshift(sp.fft.fft(funct, fft_size)))
# DFT

# dft = sp.linalg.dft(fft_size)
# transform_dft = dft@funct
#
# plt.figure("DFT Transform")
# plt.plot(x, sp.fft.fftshift(transform_dft))

plt.figure("Rect transform")
plt.plot(x, sp.fft.fftshift(sp.fft.fft(sp.fft.fftshift(rect),  fft_size)))

transformed_rectangle = np.roll(rect, 16)
plt.figure("rect shift transform")
plt.plot(x, sp.fft.fftshift(sp.fft.fft(sp.fft.fftshift(transformed_rectangle),  fft_size)))


plt.show()




plt.show()