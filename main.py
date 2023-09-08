import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from OFDMradar import OFDMradar
from Target import Target
from TargetDetection import TargetDetector
from plotStatistics import *

lightspeed = 3e8
carrier_freq = 28e9
lambda_wv = lightspeed / carrier_freq

N_subcarriers = 512
M_symbols = 512

expansion_factor = 1
N_per = N_subcarriers * expansion_factor
M_per = M_symbols * expansion_factor

N_guard = 0  # approximation to remove CP -> need to change the channel matrix generator
# N_guard = int(N_subcarriers / 8)

QAM_size = 4

delta_f = 120e3
T = 1 / delta_f  # sym_duration
Tcp = T / 8
Ts = T + Tcp  # total sym_duration

SNR_F_dB = 5
SNR = 10 ** (SNR_F_dB / 10)
noise_variance = 1 / SNR

# CFAR Params
prob_false_alarm = 1e-4

max_unambiguous_range = lightspeed / (2 * delta_f)
range_resolution = max_unambiguous_range / N_per
max_unambiguous_velocity = lightspeed / (2 * carrier_freq * Ts)
velocity_resolution = lightspeed / (2 * M_symbols * Ts * carrier_freq)

max_target_range = Tcp * lightspeed
max_target_velocity = (delta_f / 10) * (lightspeed / carrier_freq)

print("RADAR PARAMETERS:")
print(
    f"\t delta_f: {delta_f}, \n\t max unambiguous range: {max_unambiguous_range}, \n\t range resolution: {range_resolution}")
print("\t Symbol duration: ", T, "\n\t Total sym_duration: ", Ts, "\n", "max unambiguous velocity: ",
      max_unambiguous_velocity, "\n", f"velocity resolution: ", velocity_resolution, )
print("\t Carrier frequency: ", carrier_freq, "\n\t lambda_wv: ", lambda_wv, "\n")
print(f"Cyclic prefix: {Tcp}")
print("\t N_subcarriers: ", N_subcarriers, "\n\t N_guard: ", N_guard, "\n\t M_symbols: ", M_symbols, "\n")

print(
    f"\t Detection Ranges: (according to OFDM assumptions):\n\t Max_target_range: {max_target_range} \n\t Max_target_velocity: {max_target_velocity} \n")

# Target definition
H = 3
target_list = []

for i in range(H):
    target_rcs = 20  # m^2
    target_distance = np.random.randint(0, max_target_range)  # m
    target_delay = 2 * target_distance / lightspeed
    target_speed = np.random.randint(-max_target_velocity / 2, max_target_velocity / 2)  # m/s
    target_doppler = 2 * target_speed / lambda_wv

    target1 = Target(target_rcs, target_distance, target_speed)
    print("Target #", i, " parameters: ", "\n distance: ", target_distance, "\n speed: ",
          target_speed,
          "\n", target_rcs, "target cross section\n")
    target_list.append(target1)

targetDetector = TargetDetector(N_per, M_per, expansion_factor, delta_f, Ts, carrier_freq)

# print(target_list)
# b0 = np.sqrt((lightspeed * target_rcs) / ((4 * np.pi) ** 2 * target_distance ** 2 * carrier_freq ** 2))

# initialize OFDM radar
ofdm = OFDMradar(delta_f, N_subcarriers, N_guard, M_symbols, expansion_factor, QAM_size, SNR_F_dB, H, target_list)

transmitted_frame = ofdm.gen_random_data()

noise_matrix = ofdm.gen_noise_matrix()

noise_periodgram, x = ofdm.periodgram(noise_matrix)

# plt.figure()
# plt.hist(np.reshape(noise_periodgram, -1), bins=256)

# print(f"Max bin Noise periodgram: {np.max(noise_periodgram)}")

channel_matrix = ofdm.gen_channel_matrix()

received_frame = np.multiply(transmitted_frame, channel_matrix)
received_frame = np.add(received_frame, noise_matrix)

signal_shape = np.shape(transmitted_frame)

print("qam_signal is: ", transmitted_frame, "\n its size is: \n" + "N (size 0) = ", signal_shape[0],
      "\n" + "M (size 1) = ", signal_shape[1], "\n")

# normalize Frx by Ftx:
frx_norm = np.divide(received_frame, transmitted_frame)

periodogram, main_lobes_normalized = ofdm.periodgram(frx_norm, windowing=True)

ofdm.plot_periodgram(periodogram)
targetlist, binary_map = targetDetector.cfar_estimation(periodogram, prob_false_alarm, max_target_range,
                                                        main_lobes_normalized)

CA_CFAR = targetDetector.ca_cfar_estimation(periodogram, expansion_factor, prob_false_alarm, main_lobes_normalized)

# Plotting
ofdm.plot_periodgram(binary_map)
plt.figure()
plt.hist(np.reshape(np.square(np.abs(noise_matrix)), -1), bins=256)
plt.hist(np.reshape(np.square(np.abs(periodogram)), -1), bins=256)

plt.show()
