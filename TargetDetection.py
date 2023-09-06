import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows as windows


def successive_cancellation(x, window_len=100, window='hanning'):
    return


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

        binary_map = np.ones(periodgram.shape)

        threshold = - noise_variance_est * np.log(1 - (1 - false_alarm_prob) ** (1 / (n_per * m_per)))
        print(f"Threshold is: {threshold}")

        main_lobes = [main_lobes_norm[0] * n_per, main_lobes_norm[1] * m_per]
        print(f"Main lobes are: {main_lobes}")

        print("--------------------------------")
        print("Starting SUCCESSIVE CANCELLATION:")

        submatrix_size = main_lobes * 2

        max_bin = periodgram.argmax()

        # Calculate the indices for the submatrix
        start_row = max(0, - submatrix_size[0] // 2)
        end_row = min(N, n + submatrix_size[0] // 2 + 1)
        start_col = max(0, m - submatrix_size[1] // 2)
        end_col = min(M, m + submatrix_size[1] // 2 + 1)

        # Extract the submatrix from the periodogram
        submatrix = matrix[start_row:end_row, start_col:end_col]

        return 1
