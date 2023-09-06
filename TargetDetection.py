import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows as windows


def successive_cancellation(periodgram, main_lobes, threshold):
    print("--------------------------------")
    print("Starting SUCCESSIVE CANCELLATION:")

    (n_per, m_per) = periodgram.shape
    
    target_list = []
    
    submatrix_size = main_lobes * 2

    binary_map = np.ones(periodgram.shape)

    max_bin = periodgram.argmax()

    while max_bin > threshold:
        max_bin_position = np.unravel_index(periodgram.argmax(), periodgram.shape)

        # Calculate the indices for the submatrix
        start_row = int(max(0, max_bin_position[0] - submatrix_size[0] // 2))
        end_row = int(min(n_per, max_bin_position[0] + submatrix_size[0] // 2 + 1))
        start_col = int(max(0, max_bin_position[1] - submatrix_size[1] // 2))
        end_col = int(min(m_per, max_bin_position[1] + submatrix_size[1] // 2 + 1))

        # Extract the submatrix from the periodogram
        for i in range(start_row, end_row):
            for j in range(start_col, end_col):
                binary_map[i, j] = 1 if ((i - max_bin_position[0]) ** 2 / (main_lobes[0] / 2) ** 2) + (
                            (j - max_bin_position[1]) ** 2 / (main_lobes[1] / 2) ** 2) >= 1 else 0

        target_list.append(max_bin_position)

        # find the successive max value, given that it's corresponding bin in the binary matrix is set to 1
        valid_bins = np.argwhere(binary_map[periodgram.shape[0], periodgram.shape[1]] == 1)
        print("valid bins: ", valid_bins)
        max_bin = max(periodgram[valid_bins])

    print(F"FINISH SUCCESSIVE CANCELLATION, TARGET FOUND: {len(target_list)}")

    return target_list, binary_map


class TargetDetector(object):
    def __init__(self, n_per, m_per, deltaf, Tsym, carrier_freq):
        self.n_per = n_per
        self.m_per = m_per
        self.deltaf = deltaf
        self.Tsym = Tsym
        self.carrier_freq = carrier_freq

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

    def estimation_successive_canc(self, periodgram, false_alarm_prob, max_distance, main_lobes_norm):
        print("CFAR--------------------------------")
        (n_per, m_per) = periodgram.shape

        n_max = int(np.ceil((max_distance / 3e8) * 2 * self.deltaf * self.n_per))
        # k = 3

        noise_periodgram = periodgram[n_per - 4:, :]

        # calculate noise variance per bin
        noise_variance_est = np.mean(noise_periodgram)

        # print(f"Estimated Noise variance is: {noise_variance_est}")

        # noise_variance = 1 / SNR
        # print(f"Noise variance is: {noise_variance}")

        threshold = - noise_variance_est * np.log(1 - (1 - false_alarm_prob) ** (1 / (n_per/8 * m_per/8)))
        print(f"Threshold is: {threshold}")

        main_lobes = [main_lobes_norm[0] * n_per, main_lobes_norm[1] * m_per]
        print(f"Main lobes are: {main_lobes}")

        target_list = successive_cancellation(periodgram, main_lobes, threshold)

        return target_list
