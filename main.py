import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from OFDMradar import OFDMradar
from Target import Target
from TargetDetection import TargetDetector


lightspeed = 3e8
carrier_freq = 28e9
lambda_wv = lightspeed / carrier_freq

N_subcarriers = 512
M_symbols = 512

expansion_factor = 8
N_per = N_subcarriers * expansion_factor
M_per = M_symbols * expansion_factor

N_guard = 0  # approximation to remove CP -> need to change the channel matrix generator
# N_guard = int(N_subcarriers / 8)

QAM_size = 4

delta_f = 120e3
T = 1 / delta_f  # sym_duration
Tcp = T / 8
Ts = T + Tcp # total sym_duration

SNR_F_dB = 15
SNR = 10 ** (SNR_F_dB / 10)

# CFAR Params
prob_false_alarm = 0.001

max_unambiguous_range = lightspeed / (2 * delta_f)
range_resolution = max_unambiguous_range/N_per
max_unambiguous_velocity = lightspeed / (2 * carrier_freq * Ts)
velocity_resolution = lightspeed / (2 * M_symbols * Ts * carrier_freq)

max_target_range = Tcp * lightspeed
max_target_velocity = (delta_f/10)*(lightspeed/carrier_freq)


print("RADAR PARAMETERS:")
print(f"\t delta_f: {delta_f}, \n\t max unambiguous range: {max_unambiguous_range}, \n\t range resolution: {range_resolution}")
print("\t Symbol duration: ", T, "\n\t Total sym_duration: ", Ts, "\n", "max unambiguous velocity: ", max_unambiguous_velocity, "\n", f"velocity resolution: ", velocity_resolution,)
print("\t Carrier frequency: ", carrier_freq, "\n\t lambda_wv: ", lambda_wv, "\n")
print(f"Cyclic prefix: {Tcp}")
print("\t N_subcarriers: ", N_subcarriers, "\n\t N_guard: ", N_guard, "\n\t M_symbols: ", M_symbols, "\n")

print(f"\t Detection Ranges: (according to OFDM assumptions):\n\t Max_target_range: {max_target_range} \n\t Max_target_velocity: {max_target_velocity} \n")

# Target definition
H = 1
target_list = []

for i in range(H):
    target_rcs = 20  # m^2
    target_distance = np.random.randint(0, max_target_range)  # m
    target_delay = 2 * target_distance / lightspeed
    target_speed = np.random.randint(-max_target_velocity/2, max_target_velocity/2)  # m/s
    target_doppler = 2 * target_speed / lambda_wv

    target1 = Target(target_rcs, target_distance, target_speed)
    print("Target #", i, " parameters: ", "\n distance: ", target_distance, "\n speed: ",
          target_speed,
          "\n", target_rcs, "target cross section\n")
    target_list.append(target1)

targetDetector = TargetDetector(N_per, M_per, delta_f, Ts, carrier_freq)

# print(target_list)
# b0 = np.sqrt((lightspeed * target_rcs) / ((4 * np.pi) ** 2 * target_distance ** 2 * carrier_freq ** 2))

# initialize OFDM radar
ofdm = OFDMradar(delta_f, N_subcarriers, N_guard, M_symbols, expansion_factor, QAM_size, SNR_F_dB, H, target_list)

transmitted_frame = ofdm.gen_random_data()

noise_matrix = ofdm.gen_noise_matrix()

channel_matrix = ofdm.gen_channel_matrix()

received_frame = np.multiply(transmitted_frame, channel_matrix)
received_frame = np.add(received_frame, noise_matrix)

signal_shape = np.shape(transmitted_frame)

print("qam_signal is: ", transmitted_frame, "\n its size is: \n" + "N (size 0) = ", signal_shape[0],
     "\n" + "M (size 1) = ", signal_shape[1], "\n")

# normalize Frx by Ftx:
frx_norm = np.divide(received_frame, transmitted_frame)


#periodgram_windowed = ofdm.periodgram(frx_norm, windowing=True)
periodgram, main_lobes_normalized = ofdm.periodgram(frx_norm, windowing=False)

# periodgram_shape = np.shape(periodgram)

# periodgram_difference = np.subtract(periodgram, periodgram_windowed)

# TODO add window main lobe as return value
# periodgram_power_windowed = np.sum(periodgram_windowed) # * (1/(N_per*M_per))
# print(f"periodgram_power_windowed is: {periodgram_power_windowed}")
# periodgram_power = np.sum(periodgram)
# print(f"periodgram_power is: {periodgram_power}")

# print(f"periodgram shape: {periodgram_shape}\n")

#max_position_naive = targetDetector.get_naive_maximum(periodgram_windowed, interpolation=False)

max_position_naive, target_estimates = targetDetector.get_naive_maximum(periodgram, interpolation=False, print_estimate=False)

targetlist = targetDetector.estimation_successive_canc(periodgram, prob_false_alarm, max_target_range, main_lobes_normalized)

''' CHECK FOR SUCCESSIVE CANCELLATION, TO BE REMOVED
Nwin = main_lobes_normalized[0]*N_per
Mwin = main_lobes_normalized[1]*M_per

ellipsis_function = np.zeros(periodgram.shape)

for i in range(ellipsis_function.shape[0]):
    for j in range(ellipsis_function.shape[1]):
        ellipsis_function[i, j] = 1 if ((i-max_position_naive[0])**2/(Nwin/2)**2) + ((j-max_position_naive[1])**2/(Mwin/2)**2) <= 1 else 0
        
ofdm.plot_periodgram(ellipsis_function)
'''

# ofdm.plot_periodgram(periodgram, interpolation=True)
ofdm.plot_periodgram(periodgram)
# ofdm.plot_periodgram(periodgram_difference)

plt.show()
