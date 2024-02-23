import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from OFDMradar import OFDMradar
from Target import Target
from TargetDetection import TargetDetector
from plotStatistics import *
import time

lightspeed = 3e8
carrier_freq = 27.4e9
lambda_wv = lightspeed / carrier_freq

N_subcarriers = 1584
M_symbols = 1120
dl_symbols = 104
ul_symbols = 36

expansion_factor = 1
N_per = N_subcarriers * expansion_factor
M_per = M_symbols * expansion_factor

N_guard = 0  # approximation to remove CP -> need to change the channel matrix generator
# N_guard = int(N_subcarriers / 8)

QAM_size = 4

delta_f = 120e3
T = 8.33e-6  # sym_duration
Tcp = 0.59e-06
Ts = T + Tcp  # total sym_duration

SNR_F_dB = 10
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
H = 1
target_list = []



# for i in range(H):
#     target_rcs = 20  # m^2
#     # target_distance = np.random.randint(0, max_target_range)  # m
#     target_distance = 15
#     target_delay = 2 * target_distance / lightspeed
#     # target_speed = np.random.randint(-max_target_velocity / 2, max_target_velocity / 2)  # m/s
#     target_speed = 2
#     target_doppler = 2 * target_speed / lambda_wv
#
#     target_list.append(Target(target_rcs, target_distance, target_speed))
#     print("Target #", i, " parameters: ", "\n distance: ", target_distance, "\n speed: ",
#           target_speed,
#           "\n", target_rcs, "target cross section\n")

target_list_rs = [[15, 2.3],[10,0.85]]
target_rcs = 1
for i,target in enumerate(target_list_rs):
    target_delay = 2 * target[0] / lightspeed
    target_doppler = 2 * target[1]/ 3e8 * 27.4e9
    target_list.append(Target(target_rcs, target[0], target[1]))
    print("Target #", i, " parameters: ", "\n distance: ", target[0], "\n speed: ",  target[1], "\n", target_rcs, "target cross section\n")

targetDetector = TargetDetector(N_per, M_per, expansion_factor, delta_f, Ts, carrier_freq)


# print(target_list)
# b0 = np.sqrt((lightspeed * target_rcs) / ((4 * np.pi) ** 2 * target_distance ** 2 * carrier_freq ** 2))

# initialize OFDM radar
ofdm = OFDMradar(delta_f, N_subcarriers, N_guard, M_symbols, expansion_factor, QAM_size, SNR_F_dB, H, target_list, decimation=1)

transmitted_frame = ofdm.gen_random_data()

noise_matrix = ofdm.gen_noise_matrix()

# noise_periodgram, x = ofdm.periodgram(noise_matrix)

# plt.figure()
# plt.hist(np.reshape(noise_periodgram, -1), bins=256)

# print(f"Max bin Noise periodgram: {np.max(noise_periodgram)}")

channel_matrix = ofdm.gen_channel_matrix()

received_frame = np.multiply(channel_matrix, transmitted_frame)
received_frame = np.add(received_frame, noise_matrix)

signal_shape = np.shape(transmitted_frame)

# print("qam_signal is: ", transmitted_frame, "\n its size is: \n" + "N (size 0) = ", signal_shape[0],
#       "\n" + "M (size 1) = ", signal_shape[1], "\n")

# normalize Frx by Ftx:
H = np.divide(received_frame, transmitted_frame)

# num_groups = H.shape[1] // (dl_symbols + ul_symbols)
# # Reshape the H matrix to have shape (num_groups, dl_symbols + ul_symbols, -1)
# H_reshaped = H.reshape(H.shape[0], dl_symbols + ul_symbols, num_groups, order='F')
# # Set the last 36 symbols of each group to zero
# H_reshaped[:, -ul_symbols:, :] = 0j
# # Reshape the H matrix back to its original shape
# H = H_reshaped.reshape(H.shape, order='F')


#periodogram, main_lobes_normalized = ofdm.periodgram(H, windowing=True)


# targetlist, binary_map = targetDetector.cfar_estimation(periodogram, prob_false_alarm, max_target_range,main_lobes_normalized)

# t = time.time()
# CA_CFAR = targetDetector.ca_cfar_estimation(periodogram, expansion_factor, prob_false_alarm, main_lobes_normalized)
# print("Elapsed time try 1: ",time.time() - t)
# t = time.time()
# CA_CFAR = targetDetector.ca_cfar_estimation(periodogram, expansion_factor, prob_false_alarm, main_lobes_normalized, smart=1)
# print("Elapsed time try SMART: ",time.time() - t)
# Plotting
# periodogram, main_lobes_normalized = ofdm.periodgram(H, windowing=True)
# ofdm.plot_periodgram(periodogram)

plt.figure()
plt.imshow(np.real(H))
plt.ylabel('Subcarrier n', fontsize=16)
plt.xlabel('Symbol m', fontsize=16)
plt.title(r'$\mathfrak{Re}(CSI)$', fontsize=16)
plt.show()

periodogram1, _ = ofdm.create_periodogram(H, window='Hann')
ofdm.plot_periodogram1(periodogram1, scale='db')
# ofdm.plot_periodogram1(periodogram1, scale='lin', per_save_path='/home/tosi/Pictures/ThesisPics/TDD_win/periodogram_ideal_scenario')

# plt.figure()
# plt.hist(np.reshape(np.square(np.abs(noise_matrix)), -1), bins=256)
# plt.hist(np.reshape(np.square(np.abs(periodogram)), -1), bins=256)

