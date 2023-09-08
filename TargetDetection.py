import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def successive_cancellation(periodogram, main_lobes, threshold):
    print("--------------------------------")
    print("Starting SUCCESSIVE CANCELLATION:")

    (n_per, m_per) = periodogram.shape

    target_list = []

    submatrix_size = main_lobes * 2

    binary_map = np.copy(periodogram)
    plot_map = np.ones_like(periodogram)

    max_bin = np.max(periodogram)

    while max_bin > threshold:
        max_bin_position = np.unravel_index(binary_map.argmax(), binary_map.shape)

        # Calculate the indices for the submatrix
        start_row = int(max(0, max_bin_position[0] - submatrix_size[0] // 2))
        end_row = int(min(n_per, max_bin_position[0] + submatrix_size[0] // 2 + 1))
        start_col = int(max(0, max_bin_position[1] - submatrix_size[1] // 2))
        end_col = int(min(m_per, max_bin_position[1] + submatrix_size[1] // 2 + 1))

        # Extract the submatrix from the periodogram
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                if ((i - max_bin_position[0]) ** 2 / (main_lobes[0] / 2) ** 2) + (
                        (j - max_bin_position[1]) ** 2 / (main_lobes[1] / 2) ** 2) <= 1:
                    binary_map[i, j] = 0
                    plot_map[i, j] = 0

        target_list.append(max_bin_position)

        # find the successive max value, given that it's corresponding bin in the binary matrix is set to 1
        max_bin = np.max(binary_map)

    print(f"FINISH SUCCESSIVE CANCELLATION, TARGET FOUND: {len(target_list)}")

    return target_list, plot_map


def is_odd(num):
    if num % 2:
        return True
    else:
        return False


class TargetDetector(object):
    def __init__(self, n_per, m_per, upscale_factor, deltaf, Tsym, carrier_freq):
        self.n_per = n_per
        self.m_per = m_per
        self.deltaf = deltaf
        self.Tsym = Tsym
        self.carrier_freq = carrier_freq
        self.upscale_factor = upscale_factor

    def get_naive_maximum(self, periodgram, interpolation: bool = False, print_estimate: bool = False):
        lightspeed = 3e8
        carrier_freq: float = 28e9
        delta_f: float = self.deltaf
        ts: float = self.Tsym

        max_position = np.unravel_index(periodgram.argmax(), periodgram.shape)
        print(f"max position is: {max_position}")

        if interpolation:
            max_position = self.interpolate_local_maximum(periodgram, max_position, interp_type='linear')

        max_position = self.interpolate_local_maximum(periodgram, max_position, interp_type='nearest')
        estimated_distance = max_position[0] * lightspeed / (2 * delta_f * self.n_per)
        estimated_speed = (max_position[1] - self.m_per / 2) * lightspeed / (2 * carrier_freq * ts * self.m_per)

        if print_estimate:
            print("max position is: ", max_position)
            print("Estimated distance is: ", estimated_distance, "\n")
            print("Estimated speed is: ", estimated_speed, "\n")

        return max_position, [estimated_distance, estimated_speed]

    def interpolate_local_maximum(self, periodgram, max_coords, interp_type='linear'):

        if interp_type == 'linear':
            interpolated_n = (max_coords[0] - 1) * periodgram[max_coords[0] - 1, max_coords[1]]
            interpolated_n += max_coords[0] * periodgram[max_coords] + (max_coords[0] + 1) * periodgram[
                max_coords[0] + 1, max_coords[1]]
            interpolated_n /= periodgram[max_coords] + periodgram[max_coords[0] + 1, max_coords[1]] + periodgram[
                max_coords[0] - 1, max_coords[1]]
            print(f"estimated n is: {max_coords[0]} \tinterpolated_n is: {interpolated_n}")

            interpolated_m = (max_coords[1] - 1) * periodgram[max_coords[0], max_coords[1] - 1] + max_coords[1] * \
                             periodgram[max_coords] + (max_coords[1] + 1) * periodgram[
                                 max_coords[0], max_coords[1] + 1]
            interpolated_m /= periodgram[max_coords] + periodgram[max_coords[0], max_coords[1] + 1] + periodgram[
                max_coords[0], max_coords[1] - 1]
            print(f"estimated m is: {max_coords[1]} \tinterpolated_m is: {interpolated_m}")

        else:
            return None, None
            # TODO add different types of interpolation
        max_position_interpolated = (interpolated_n, interpolated_m)
        return max_position_interpolated

    def get_cfar_threshold(self, periodogram, false_alarm_prob):
        (n_per, m_per) = periodogram.shape
        # n_max = int(np.ceil((max_distance / 3e8) * 2 * self.deltaf * self.n_per))
        k = 5
        noise_periodgram = periodogram[n_per - k:, :]

        # calculate noise variance per bin
        noise_variance_est = np.mean(noise_periodgram)
        print(f"Estimated Noise variance is: {noise_variance_est}")

        threshold = - noise_variance_est * np.log(
            1 - (1 - false_alarm_prob) ** (self.upscale_factor ** 2 / (n_per * m_per)))
        print(f"Threshold is: {threshold}")

        return threshold

    def cfar_estimation(self, periodogram, false_alarm_prob, max_distance, main_lobes_norm):
        print("CFAR--------------------------------")
        (n_per, m_per) = periodogram.shape

        threshold = self.get_cfar_threshold(periodogram, false_alarm_prob)

        main_lobes = [main_lobes_norm[0] * n_per, main_lobes_norm[1] * m_per]
        print(f"Main lobes are: {main_lobes}")

        target_list, binary_map = successive_cancellation(periodogram, main_lobes, threshold)

        return target_list, binary_map

    def get_ca_sliding_window(self, l1, l2, ng1, ng2):
        center_l1 = (l1 - 1) // 2
        center_l2 = (l2 - 1) // 2

        window = np.ones((l1,l2))

        window[center_l1-ng1:center_l1+ng1+1, center_l2-ng2:center_l2+ng2+1] = 0

        return window

    def ca_cfar_tresholding(self, matrix, sliding_window, false_alarm_prob):
        sliding_sum = signal.convolve2d(matrix, sliding_window, mode='same')
        contribution_matrix = signal.convolve2d(np.ones(matrix.shape), sliding_window, mode='same')

        alpha_matrix = (false_alarm_prob ** (np.divide(-1, contribution_matrix))) - 1

        # FORMULA: threshold = alpha * sigma_est, sigma_est = sliding_sum
        threshold_matrix = np.multiply(sliding_sum, alpha_matrix)

        return threshold_matrix

    def ca_cfar_estimation(self, periodogram, expansion_factor, false_alarm_prob, main_lobes_norm):
        print("CA-CFAR--------------------------------")
        (n_per, m_per) = periodogram.shape
        main_lobes = [main_lobes_norm[0] * n_per, main_lobes_norm[1] * m_per]

        window_length_doppler = int(2 * np.ceil(main_lobes[1]))
        window_length_distance = int(2 * np.ceil(main_lobes[0]))
        window_length_doppler += 1
        window_length_distance += 1

        window_length_distance = 14
        window_length_doppler = 14

        '''
        if not is_odd(window_length_doppler):
            window_length_doppler += 1
        if not is_odd(window_length_distance):
            window_length_distance += 1
        '''

        guard_cell_try = 1

        # guard_cell_length_doppler = int((window_length_doppler + 1) // 2)
        # guard_cell_length_distance = int((window_length_distance + 1) // 2)

        window = self.get_ca_sliding_window(window_length_distance, window_length_doppler, guard_cell_try,
                                            guard_cell_try)

        plt.figure("window")
        plt.pcolormesh(window)

        threshold_matrix = self.ca_cfar_tresholding(periodogram, window, false_alarm_prob)

        line = periodogram[:,np.unravel_index(periodogram.argmax(),periodogram.shape)[1]]
        threshold_line = threshold_matrix[:,np.unravel_index(periodogram.argmax(),periodogram.shape)[1]]
        cfar_threshold = self.get_cfar_threshold(periodogram, false_alarm_prob)
        cfar_threshold = np.full(line.shape, cfar_threshold)
        plt.figure("1d thresholding")
        plt.plot(line)
        plt.plot(threshold_line)
        plt.plot(cfar_threshold)


        threshold_logical = periodogram > threshold_matrix

        periodogram_only_targets = np.copy(periodogram)
        periodogram_only_targets[~threshold_logical] = 0

        plt.figure("CA-CFAR")
        plt.pcolormesh(periodogram_only_targets)


        targetlist = []
        return targetlist
