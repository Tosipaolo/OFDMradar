import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows as windows
from TargetDetection import TargetDetector
from scipy import signal as sig
from matplotlib.colors import ListedColormap


def calculate_matrix_power(matrix):
    power = 0
    c = 0

    matrix *= matrix.conj()

    for row in matrix:
        for element in row:
            power += element ** 2
            c += 1

    return power


class OFDMradar:
    def __init__(self, deltaf, n: int, N_guard: int, m: int, upsample_factor: int, QAM_order: int, SNR_db,
                 num_targets: int, target_list, decimation):
        self.n = n
        self.m = m
        self.N_guard = N_guard
        self.QAM_order = QAM_order
        self.num_targets = num_targets
        self.target_list = target_list
        self.deltaf = deltaf
        self.Tsym = 8.33e-6
        self.Tsym += 0.59e-6

        self.SNR_linear = 10 ** (SNR_db / 10.0)
        self.upsample_factor = upsample_factor
        self.padding_factor = 2
        self.range_unamb = 3e8 / (2 * deltaf)
        self.speed_unamb = 3e8 / (2 * 27.4e9 * (8.33e-6+0.59e-6) * decimation)
        self.max_eval_range = 45
        self.max_eval_speed = 4
        self.rand_phase_shift = np.exp(1j*2*np.pi* np.random.uniform(0, 2 * np.pi))

    def gen_random_data(self):
        np.random.seed(47)

        monodim_signal1 = np.random.randint(0, np.sqrt(self.QAM_order), size=(self.n, self.m))
        monodim_signal1 = np.subtract(monodim_signal1, (np.sqrt(self.QAM_order) - 1) / 2)
        monodim_signal1 = np.multiply(monodim_signal1, 2)

        monodim_signal2 = np.random.randint(0, np.sqrt(self.QAM_order), size=(self.n, self.m))
        monodim_signal2 = np.subtract(monodim_signal2, (np.sqrt(self.QAM_order) - 1) / 2)
        monodim_signal2 = np.multiply(monodim_signal2, 2)

        qam_signal = np.add(monodim_signal1, np.multiply(monodim_signal2, 1j))
        qam_signal = np.multiply(qam_signal, 1 / np.sqrt(2))
        return qam_signal

    def gen_noise_matrix(self):
        np.random.seed(47)
        N_subcarriers = self.n
        M_symbols = self.m

        print("SNR linear is:", self.SNR_linear, "\n")
        noise_std = np.sqrt(1 / self.SNR_linear)
        print("Noise variance is:", noise_std, "\n")

        noise_matrix = (1 / np.sqrt(2)) * np.random.normal(0, noise_std, size=(N_subcarriers, M_symbols))
        noise_matrix_complex = (1 / np.sqrt(2)) * 1j * np.random.normal(0, noise_std, size=(N_subcarriers, M_symbols))
        noise_matrix = np.add(noise_matrix, noise_matrix_complex)
        # print("Noise matrix is:", noise_matrix, "\n")
        avg = np.mean(np.square(np.abs(noise_matrix)))
        print("Noise matrix average is:", avg, "\n")

        return noise_matrix

    def gen_channel_matrix(self):
        global channel_matrix
        lightspeed = 3e8
        carrier_freq: float = 27.4e9
        channel_matrix = np.zeros((self.n, self.m), dtype=complex)
        target_matrix = np.zeros((self.n, self.m), dtype=complex)

        for target in self.target_list:
            # print(target.target_rcs, target.target_distance, target.target_delay, target.target_doppler)

            # b_target = np.sqrt((lightspeed * target.target_rcs) / ((4 * np.pi) ** 3 * target.target_distance ** 4 * carrier_freq ** 2))
            # print(b_target)

            b_target = 0.3
            frequency_shift = [np.exp(1j * (2 * np.pi) * target.target_doppler * self.Tsym * m) for m in range(self.m)]
            phase_shift = [np.exp(-1j * (2 * np.pi) * (n * self.deltaf) * target.target_delay) for n in range(self.n)]
            target_matrix += np.outer(frequency_shift, phase_shift).transpose() * b_target
            target_matrix *= np.exp(1j * np.random.uniform(0, 2 * np.pi))
            channel_matrix += target_matrix


            # for m in range(self.m):
            #     frequency_shift = np.exp(1j * (2 * np.pi) * target.target_doppler * self.Tsym * m)
            #
            #     for n in range(self.n):
            #         phase_shift = np.exp(-1j * (2 * np.pi) * (n * self.deltaf) * target.target_delay)
            #         target_matrix[n, m] = b_target * frequency_shift * phase_shift * \
            #                                              np.exp(1j * (- 2 * np.pi * carrier_freq * target.target_delay)
            #                                                     )
            # target_matrix *= self.rand_phase_shift
            # self.rand_phase_shift += np.exp(-1j*2*np.pi*carrier_freq*0.0099904)

        return target_matrix

    def window_generator(self, received_frame):
        # print("---------------WINDOWING-----------------")
        #
        # sidelobe_attenuation = 100
        #
        # single_window_matrix_n = windows.chebwin(self.n, sidelobe_attenuation)
        # win_normalization_n = (1 / self.n) * np.sum(single_window_matrix_n ** 2)
        # single_window_matrix_n = np.multiply(single_window_matrix_n, np.sqrt(1 / win_normalization_n))
        # check_win_n_normalization = (1 / self.n) * np.sum(single_window_matrix_n ** 2)
        # print(f"normalization of N window matrix is: {check_win_n_normalization}")
        #
        # test_window = False
        #
        # single_window_matrix_m = windows.chebwin(self.m, sidelobe_attenuation)
        # win_normalization_m = (1 / self.m) * np.sum(single_window_matrix_m ** 2)
        # single_window_matrix_m = np.multiply(single_window_matrix_m, np.sqrt(1 / win_normalization_m))
        #
        # window_matrix = np.outer(single_window_matrix_n, single_window_matrix_m)
        #
        # # check_win_2d_normalization = (1 / (self.N * self.N)) * np.sum(window_matrix ** 2)
        # # print(f"normalization of 2D window matrix is: {check_win_2d_normalization}")
        #
        # main_lobe_width_n = (1.46 * np.pi * (np.log10(2) + sidelobe_attenuation / 20)) / (self.n - 1)
        # main_lobe_width_n = np.ceil(self.n * main_lobe_width_n)
        # print(f"main lobe width in n is: {main_lobe_width_n}")
        #
        # main_lobe_width_m = (1.46 * (np.log10(2) + sidelobe_attenuation / 20)) / (self.m - 1)
        # main_lobe_width_m = np.ceil(self.m * main_lobe_width_m)
        # print(f"main lobe width in m is: {main_lobe_width_m}")
        #
        # main_lobe_width = lambda x: (1.46 * (np.log10(2) + sidelobe_attenuation / 20)) / (x - 1)
        # main_lobe_normalized = [main_lobe_width(self.n), main_lobe_width(self.m)]
        # print(f"main lobe dimensions are: {main_lobe_normalized}")
        #
        # if test_window:
        #     plt.figure(figsize=(20, 20))
        #     plt.plot(single_window_matrix_n)
        #
        #     plt.figure(figsize=(20, 20))
        #     plt.plot(20 * np.log10((np.abs(np.fft.fftshift(np.fft.fft(single_window_matrix_n, self.n))))))
        #
        #     check_win_m_normalization = (1 / self.m) * np.sum(single_window_matrix_m ** 2)
        #     print(f"normalization of M window matrix is: {check_win_m_normalization}")
        #     win_normalization_2d = (1 / (self.n * self.n)) * np.sum(window_matrix ** 2)
        #
        #     # plt.plot(X, np.fft.fftshift(np.fft.fft(single_window_matrix_n, self.n)))
        #
        #     print("--------------------------------")
        #     print(f"Norm of N window matrix is: {np.linalg.norm(single_window_matrix_n)}")
        #     print(f"Norm of M window matrix is: {np.linalg.norm(single_window_matrix_m)}")
        #     print(f"norm of window matrix is: {np.linalg.norm(window_matrix)}")
        #     print(f"window normalization constant is: {win_normalization_2d}")
        #
        #     SNR_loss_n = (1 / (self.n * np.inner(single_window_matrix_n, single_window_matrix_n))) * (
        #         np.square(np.abs(np.sum(single_window_matrix_n))))
        #     SNR_loss_n = 10 * np.log10(SNR_loss_n)
        #     SNR_loss_m = (1 / (self.m * np.inner(single_window_matrix_m, single_window_matrix_m))) * np.square(
        #         np.abs(np.sum(single_window_matrix_m)))
        #     SNR_loss_m = 10 * np.log10(SNR_loss_m)
        #
        #     print("--------------------------------")
        #     print(f"SNR_loss_N DUE TO WINDOWING: {SNR_loss_n}")
        #     print(f"SNR_loss_M DUE TO WINDOWING: {SNR_loss_m}")
        #     print(f"TOTAL SNR_loss is: {SNR_loss_n + SNR_loss_m}")

        w_rows = sig.windows.chebwin(received_frame.shape[0], 60)
        w_columns = sig.windows.chebwin(received_frame.shape[1], 60)

        sig_power_before = np.mean(np.abs(received_frame) ** 2)

        # create windowing matrix and multiply signal with it
        W = np.outer(w_rows, w_columns)
        windowed_signal = np.multiply(received_frame, W)

        # scale power of windowed signal back to original power
        windowed_signal = windowed_signal * np.sqrt(sig_power_before / np.mean(np.abs(windowed_signal) ** 2))

        return windowed_signal, 0

    def periodgram(self, received_fx, windowing: bool = False):
        N_subcarriers = self.n * self.upsample_factor
        M_symbols = self.m * self.upsample_factor
        received_frame = received_fx

        # print("--------------------------------")
        # print(f"received_frame BEFORE WINDOWING{received_frame}")

        if windowing:
            window_matrix, main_lobe_normalized = self.window_generator(received_frame)
            # received_frame = np.multiply(received_frame, window_matrix)
            main_lobe_normalized = [2.0 / self.n, 2.0 / self.m]
        else:
            main_lobe_normalized = [2.0 / self.n, 2.0 / self.m]

        Cper = np.fft.fft(received_frame, n=int(M_symbols), axis=1) / np.sqrt(M_symbols)
        Cper = np.fft.ifft(Cper, n=int(N_subcarriers), axis=0) * np.sqrt(N_subcarriers)
        Cper = np.fft.fftshift(Cper, axes=1)

        Per = np.square(np.abs(Cper))

        ''' # Test for windowing
        max_periodgram = np.max(Per)

        if windowing:
            print(f"Max windowed is: {max_periodgram}")
        else:
            print(f"Max is: {max_periodgram}")
        '''
        return Per, main_lobe_normalized

    def plot_periodgram(self, periodgram):
        N_per = self.n * self.upsample_factor
        M_per = self.m * self.upsample_factor

        lightspeed = 3e8
        carrier_freq: float = 28e9
        fontsize = 12

        fig = plt.figure(figsize=(20, 20))
        Y = np.linspace(0, N_per, N_per)
        Y = np.multiply(Y, (lightspeed / (2 * self.deltaf * N_per)))
        X = np.linspace(-M_per / 2, M_per / 2, M_per)
        X = np.multiply(X, (lightspeed / (2 * carrier_freq * self.Tsym * M_per)))
        plt.pcolormesh(X, Y, periodgram, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Relative Speed [m/s]', fontsize=fontsize)
        plt.ylabel('Range [m]', fontsize=fontsize)
        plt.title('2D Periodogram', fontsize=fontsize)
        # plt.xlim(0,30)
        # plt.ylim(-3,3)

        plt.show()

        fig = plt.figure(figsize=(20, 20))
        Y = np.linspace(0, N_per, N_per)
        Y = np.multiply(Y, (lightspeed / (2 * self.deltaf * N_per)))
        X = np.linspace(-M_per / 2, M_per / 2, M_per)
        X = np.multiply(X, (lightspeed / (2 * carrier_freq * self.Tsym * M_per)))
        plt.pcolormesh(X, Y, 10*np.log10(periodgram), cmap='viridis')
        plt.colorbar()
        plt.xlabel('Relative Speed [m/s]', fontsize=fontsize)
        plt.ylabel('Range [m]', fontsize=fontsize)
        plt.title('2D Periodogram', fontsize=fontsize)
        # plt.xlim(0, 30)
        # plt.ylim(-3, 3)
        plt.show()

        return

    def create_periodogram(self,
                           Y: np.ndarray,
                           window: str = 'Rectangular'):

        """
        Creates 2D range-speed periodogram

        Parameters
        ----------
        Y: np.ndarray
            Channel transfer function (CTF); array of shape (# of subcarriers, # of OFDM symbols) and dtype complex
        window: str, default = 'Rectangular'
            for windowing in periodogram; currently supported:
                'Chebyshev': Chebyshev window (with 50 dB sidelobe attenuation)
                'Hann': Hann window
                'Hamming': Hamming window
                'Rectangular': Rectangular window

        Returns
        -------
        PER: np.ndarray
            2D range-speed periodogram of shape (# of range bins, # of speed bins)
        CPER: np.ndarray
            complex 2D range-speed periodogram (i.e., before magnitude squared) of shape (# of range bins, # of speed bins)

        """



        # (I)DFT length should be at least as big as number of subcarriers (for range) or symbols (for speed)
        self.range_bins = int(2 ** np.ceil(np.log2(Y.shape[0]) + self.padding_factor))
        self.speed_bins = int(2 ** np.ceil(np.log2(Y.shape[1]) + self.padding_factor))

        # zero padding
        Y = zero_pad(Y, self.padding_factor)

        # windowing
        Y = apply_windowing(Y, window)

        # Y = zero_pad(Y, self.padding_factor)

        # get bin widths in terms of range and speed
        self.range_bin_width = (self.range_unamb / self.range_bins)
        self.speed_bin_width = (self.speed_unamb / self.speed_bins)

        # get number of range and speed bins from desired evaluation intervals
        range_bins_desired = int(np.ceil(self.max_eval_range / self.range_bin_width))
        speed_bins_desired = int(np.ceil(self.max_eval_speed / self.speed_bin_width))

        # range processing (accounting for desired ranges)
        W_range = DFT_matrix(0, range_bins_desired, self.range_bins).conj()
        range_processed = np.matmul(W_range.transpose(), Y)

        # speed processing (accounting for desired speeds)
        W_speed = DFT_matrix(-speed_bins_desired, speed_bins_desired, self.speed_bins)
        CPER = np.matmul(range_processed, W_speed)
        PER = np.abs(CPER) ** 2

        return PER, CPER


    def plot_periodogram1(self,
                         PER: np.ndarray,
                         scale: str = 'lin',
                         per_save_path: str = ''):

        """
        Plots 2D range-speed periodogram

        Parameters
        ----------
        PER: np.ndarray
            2D range-speed periodogram of shape (# of range bins, # of speed bins)
        scale: str, default = 'lin'
            to define how periodogram is plotted; currently supported:
                'lin': linear scale
                'db': dB scale
                'lin_norm': linear scale normalized
                'db_norm': db scale normalized
        per_save_path: default = ''
            path where periodogram is stored; by default it is an empty string, in which case the periodogram is not stored
        """

        if scale.lower() == 'lin':
            plot_PER = PER
            label = 'Power'
        elif scale.lower() == 'db':
            plot_PER = 10 * np.log10(PER)
            label = 'Power [dB]'
        elif scale.lower() == 'lin_norm':
            plot_PER = PER / np.max(PER)
            label = 'Norm. Power'
        elif scale.lower() == 'db_norm':
            plot_PER = 10 * np.log10(PER / np.max(PER))
            label = 'Norm. Power [dB]'
        else:
            print('Scale not known! Plotting in linear scale')
            plot_PER = PER
            label = 'Power'

        # calculate speeds and ranges for axes of plot
        ranges = np.linspace(0, self.range_bin_width * PER.shape[0], PER.shape[0] + 1)

        # get rid of negative ranges
        num_subzero_ranges = len(ranges[ranges < 0])
        ranges = ranges[num_subzero_ranges:]
        plot_PER = plot_PER[num_subzero_ranges:, :]

        speeds = np.linspace((-self.speed_bin_width * PER.shape[1]) / 2,
                             (self.speed_bin_width * PER.shape[1]) / 2 - self.speed_bin_width,
                             PER.shape[1])
        fontsize = 14
        plt.imshow(
            plot_PER,
            aspect="auto",
            origin="lower",
            extent=[
                speeds[0],
                speeds[-1],
                ranges[0] - (ranges[-1] - ranges[-2]) / 2,
                ranges[-1] - (ranges[-1] - ranges[-2]) / 2,
            ],
            cmap=jet_light_cm(),
        )
        cb = plt.colorbar()
        cb.set_label(label, fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize)
        plt.title("2D Periodogram", fontsize=fontsize)
        plt.xlabel("Rel. Speed [m/s]", fontsize=fontsize)
        plt.ylabel("Range [m]", fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # if save path is specified, save periodogram as .png
        if per_save_path:
            plt.savefig(per_save_path + ".png", dpi=400)
            plt.savefig(per_save_path+ '.svg', format='svg')
            plt.savefig(per_save_path+ '.eps', format='eps')
            plt.close()
            return

        plt.show()


## MISC FUNCTIONS

def apply_windowing(signal, window='Rectangular', at=45):
    """
    :param signal: signal to be windowed
    :param window: desired window
    :param at: attenuation in dB (only necessary for Chebyshev window)
    :return: windowed signal
    """

    if window == 'Chebyshev':
        w_rows = sig.windows.chebwin(signal.shape[0], at)
        w_columns = sig.windows.chebwin(signal.shape[1], at)
    elif window == 'Hann':
        w_rows = sig.windows.hann(signal.shape[0])
        w_columns = sig.windows.hann(signal.shape[1])
    elif window == 'Hamming':
        w_rows = sig.windows.hamming(signal.shape[0])
        w_columns = sig.windows.hamming(signal.shape[1])
    elif window == 'Rectangular':
        w_rows = sig.windows.boxcar(signal.shape[0])
        w_columns = sig.windows.boxcar(signal.shape[1])
    else:
        print('Window not implemented! Exiting ...')
        exit()

    # get signal power before windowing
    sig_power_before = np.mean(np.abs(signal) ** 2)

    # create windowing matrix and multiply signal with it
    W = np.outer(w_rows, w_columns)
    windowed_signal = np.multiply(signal, W)

    # scale power of windowed signal back to original power
    windowed_signal = windowed_signal * np.sqrt(sig_power_before / np.mean(np.abs(windowed_signal) ** 2))

    return windowed_signal


def DFT_matrix(low_freq_idx: int,
               high_freq_idx: int,
               N: int):
    """
    creates DFT matrix

    Parameters
    ----------
    low_freq_idx: int
        index of lowest frequency
    high_freq_idx: int
        bin index of highest frequency
    N: int
        DFT length

    Returns
    -------
    W: np.ndarray
        DFT matrix
    """

    i, j = np.meshgrid(np.arange(low_freq_idx, high_freq_idx), np.arange(N))
    omega = np.exp(-2 * np.pi * 1j / N)
    W = np.power(omega, i * j) / np.sqrt(N)

    return W

def zero_pad(Y: np.ndarray, padding_factor=0):
    """
    performs zero padding on matrix Y, where rows and columns of the array are zero padded to the next power of 2.

    Parameters
    ----------
    Y: np.ndarray
        2D array to be zero padded
    padding_factor: int, default = 0
        zero padding factor; default is 0 and means padding to the next power of 2

    Returns
    -------
    Y_padded: np.ndarray
        Zero padded 2D array
    """

    if len(Y.shape) == 3:
        new_Y = np.empty((int(2 ** np.ceil(np.log2(Y.shape[0]) + padding_factor)),
                          int(2 ** np.ceil(np.log2(Y.shape[1]) + padding_factor)), Y.shape[2]), dtype=complex)
        for i in range(Y.shape[2]):
            new_Y[:, :, i] = zero_pad(Y[:, :, i])

        return new_Y

    # pad rows
    num_zeros_scs = int((2 ** np.ceil(np.log2(Y.shape[0]) + padding_factor) - Y.shape[0]) / 2)
    Y_padded = np.concatenate((np.zeros((num_zeros_scs, Y.shape[1])),
                               Y, np.zeros((num_zeros_scs, Y.shape[1]))), axis=0)

    # pad columns
    num_zeros_syms = int((2 ** np.ceil(np.log2(Y.shape[1]) + padding_factor) - Y.shape[1]) / 2)
    Y_padded = np.concatenate((np.zeros((Y_padded.shape[0], num_zeros_syms)),
                               Y_padded, np.zeros((Y_padded.shape[0], num_zeros_syms))), axis=1)

    if Y.shape[1] % 2 != 0:
        Y_padded = np.concatenate((Y_padded, np.zeros((Y_padded.shape[0], 1))), axis=1)
    if Y.shape[0] % 2 != 0:
        Y_padded = np.concatenate((Y_padded, np.zeros((1, Y_padded.shape[1]))), axis=0)

    return Y_padded

def jet_light_cm():
    jet_light_rgb_512 = [[1, 1, 1], [0.99804, 0.99804, 0.99905], [0.99609, 0.99609, 0.99813],
                         [0.99413, 0.99413, 0.99725],
                         [0.99217, 0.99217, 0.99639], [0.99022, 0.99022, 0.99557], [0.98826, 0.98826, 0.99477],
                         [0.9863, 0.9863, 0.99401], [0.98434, 0.98434, 0.99327], [0.98239, 0.98239, 0.99257],
                         [0.98043, 0.98043, 0.9919], [0.97847, 0.97847, 0.99125], [0.97652, 0.97652, 0.99064],
                         [0.97456, 0.97456, 0.99006], [0.9726, 0.9726, 0.98951], [0.97065, 0.97065, 0.98899],
                         [0.96869, 0.96869, 0.9885], [0.96673, 0.96673, 0.98804], [0.96477, 0.96477, 0.98762],
                         [0.96282, 0.96282, 0.98722], [0.96086, 0.96086, 0.98685], [0.9589, 0.9589, 0.98652],
                         [0.95695, 0.95695, 0.98621], [0.95499, 0.95499, 0.98593], [0.95303, 0.95303, 0.98569],
                         [0.95108, 0.95108, 0.98548], [0.94912, 0.94912, 0.98529], [0.94716, 0.94716, 0.98514],
                         [0.94521, 0.94521, 0.98502], [0.94325, 0.94325, 0.98493], [0.94129, 0.94129, 0.98486],
                         [0.93933, 0.93933, 0.98483], [0.93738, 0.93738, 0.98483], [0.93542, 0.93542, 0.98486],
                         [0.93346, 0.93346, 0.98493], [0.93151, 0.93151, 0.98502], [0.92955, 0.92955, 0.98514],
                         [0.92759, 0.92759, 0.98529],
                         [0.92564, 0.92564, 0.98548], [0.92368, 0.92368, 0.98569], [0.92172, 0.92172, 0.98593],
                         [0.91977, 0.91977, 0.98621], [0.91781, 0.91781, 0.98652], [0.91585, 0.91585, 0.98685],
                         [0.91389, 0.91389, 0.98722], [0.91194, 0.91194, 0.98762], [0.90998, 0.90998, 0.98804],
                         [0.90802, 0.90802, 0.9885], [0.90607, 0.90607, 0.98899], [0.90411, 0.90411, 0.98951],
                         [0.90215, 0.90215, 0.99006], [0.9002, 0.9002, 0.99064], [0.89824, 0.89824, 0.99125],
                         [0.89628, 0.89628, 0.9919], [0.89432, 0.89432, 0.99257], [0.89237, 0.89237, 0.99327],
                         [0.89041, 0.89041, 0.99401], [0.88845, 0.88845, 0.99477], [0.8865, 0.8865, 0.99557],
                         [0.88454, 0.88454, 0.99639], [0.88258, 0.88258, 0.99725], [0.88063, 0.88063, 0.99813],
                         [0.87867, 0.87867, 0.99905], [0.87671, 0.87671, 1], [0.87476, 0.87573, 1],
                         [0.8728, 0.87479, 1],
                         [0.87084, 0.87387, 1], [0.86888, 0.87298, 1], [0.86693, 0.87213, 1], [0.86497, 0.8713, 1],
                         [0.86301, 0.87051, 1], [0.86106, 0.86974, 1], [0.8591, 0.86901, 1], [0.85714, 0.8683, 1],
                         [0.85519, 0.86763, 1], [0.85323, 0.86699, 1], [0.85127, 0.86638, 1], [0.84932, 0.8658, 1],
                         [0.84736, 0.86525, 1],
                         [0.8454, 0.86473, 1], [0.84344, 0.86424, 1], [0.84149, 0.86378, 1], [0.83953, 0.86335, 1],
                         [0.83757, 0.86295, 1], [0.83562, 0.86259, 1], [0.83366, 0.86225, 1], [0.8317, 0.86194, 1],
                         [0.82975, 0.86167, 1], [0.82779, 0.86142, 1], [0.82583, 0.86121, 1], [0.82387, 0.86103, 1],
                         [0.82192, 0.86087, 1], [0.81996, 0.86075, 1], [0.818, 0.86066, 1], [0.81605, 0.8606, 1],
                         [0.81409, 0.86057, 1], [0.81213, 0.86057, 1], [0.81018, 0.8606, 1], [0.80822, 0.86066, 1],
                         [0.80626, 0.86075, 1], [0.80431, 0.86087, 1], [0.80235, 0.86103, 1], [0.80039, 0.86121, 1],
                         [0.79843, 0.86142, 1], [0.79648, 0.86167, 1], [0.79452, 0.86194, 1], [0.79256, 0.86225, 1],
                         [0.79061, 0.86259, 1], [0.78865, 0.86295, 1], [0.78669, 0.86335, 1], [0.78474, 0.86378, 1],
                         [0.78278, 0.86424, 1], [0.78082, 0.86473, 1], [0.77886, 0.86525, 1], [0.77691, 0.8658, 1],
                         [0.77495, 0.86638, 1], [0.77299, 0.86699, 1], [0.77104, 0.86763, 1], [0.76908, 0.8683, 1],
                         [0.76712, 0.86901, 1], [0.76517, 0.86974, 1],
                         [0.76321, 0.87051, 1], [0.76125, 0.8713, 1], [0.7593, 0.87213, 1], [0.75734, 0.87298, 1],
                         [0.75538, 0.87387, 1], [0.75342, 0.87479, 1],
                         [0.75147, 0.87573, 1], [0.74951, 0.87671, 1], [0.74755, 0.87772, 1], [0.7456, 0.87876, 1],
                         [0.74364, 0.87983, 1], [0.74168, 0.88093, 1], [0.73973, 0.88206, 1], [0.73777, 0.88323, 1],
                         [0.73581, 0.88442, 1], [0.73386, 0.88564, 1], [0.7319, 0.88689, 1], [0.72994, 0.88818, 1],
                         [0.72798, 0.88949, 1], [0.72603, 0.89084, 1], [0.72407, 0.89222, 1], [0.72211, 0.89362, 1],
                         [0.72016, 0.89506, 1], [0.7182, 0.89653, 1], [0.71624, 0.89802, 1], [0.71429, 0.89955, 1],
                         [0.71233, 0.90111, 1], [0.71037, 0.9027, 1], [0.70841, 0.90432, 1], [0.70646, 0.90597, 1],
                         [0.7045, 0.90766, 1], [0.70254, 0.90937, 1], [0.70059, 0.91111, 1], [0.69863, 0.91289, 1],
                         [0.69667, 0.91469, 1], [0.69472, 0.91652, 1], [0.69276, 0.91839, 1], [0.6908, 0.92028, 1],
                         [0.68885, 0.92221, 1], [0.68689, 0.92417, 1], [0.68493, 0.92616, 1], [0.68297, 0.92817, 1],
                         [0.68102, 0.93022, 1], [0.67906, 0.9323, 1], [0.6771, 0.93441, 1], [0.67515, 0.93655, 1],
                         [0.67319, 0.93872, 1], [0.67123, 0.94092, 1],
                         [0.66928, 0.94316, 1], [0.66732, 0.94542, 1], [0.66536, 0.94771, 1], [0.66341, 0.95004, 1],
                         [0.66145, 0.95239, 1], [0.65949, 0.95478, 1],
                         [0.65753, 0.95719, 1], [0.65558, 0.95964, 1], [0.65362, 0.96211, 1], [0.65166, 0.96462, 1],
                         [0.64971, 0.96716, 1], [0.64775, 0.96973, 1], [0.64579, 0.97233, 1], [0.64384, 0.97496, 1],
                         [0.64188, 0.97762, 1], [0.63992, 0.98031, 1], [0.63796, 0.98303, 1], [0.63601, 0.98578, 1],
                         [0.63405, 0.98856, 1], [0.63209, 0.99138, 1], [0.63014, 0.99422, 1], [0.62818, 0.9971, 1],
                         [0.62622, 1, 1], [0.6272, 1, 0.99706], [0.62821, 1, 0.9941], [0.62925, 1, 0.9911],
                         [0.63032, 1, 0.98807], [0.63142, 1, 0.98502], [0.63255, 1, 0.98193], [0.63371, 1, 0.97881],
                         [0.63491, 1, 0.97566], [0.63613, 1, 0.97248], [0.63738, 1, 0.96927], [0.63867, 1, 0.96603],
                         [0.63998, 1, 0.96276], [0.64133, 1, 0.95945], [0.6427, 1, 0.95612], [0.64411, 1, 0.95276],
                         [0.64555, 1, 0.94936], [0.64702, 1, 0.94594], [0.64851, 1, 0.94248], [0.65004, 1, 0.939],
                         [0.6516, 1, 0.93548], [0.65319, 1, 0.93193], [0.65481, 1, 0.92836], [0.65646, 1, 0.92475],
                         [0.65815, 1, 0.92111], [0.65986, 1, 0.91744], [0.6616, 1, 0.91374],
                         [0.66337, 1, 0.91001], [0.66518, 1, 0.90625], [0.66701, 1, 0.90246], [0.66888, 1, 0.89864],
                         [0.67077, 1, 0.89478], [0.6727, 1, 0.8909],
                         [0.67466, 1, 0.88699], [0.67665, 1, 0.88304], [0.67866, 1, 0.87907], [0.68071, 1, 0.87506],
                         [0.68279, 1, 0.87102], [0.6849, 1, 0.86696], [0.68704, 1, 0.86286], [0.68921, 1, 0.85873],
                         [0.69141, 1, 0.85457], [0.69365, 1, 0.85039], [0.69591, 1, 0.84617], [0.6982, 1, 0.84192],
                         [0.70053, 1, 0.83763], [0.70288, 1, 0.83332], [0.70527, 1, 0.82898], [0.70768, 1, 0.82461],
                         [0.71013, 1, 0.82021], [0.7126, 1, 0.81577], [0.71511, 1, 0.81131], [0.71765, 1, 0.80681],
                         [0.72022, 1, 0.80229], [0.72282, 1, 0.79773], [0.72545, 1, 0.79314], [0.72811, 1, 0.78853],
                         [0.7308, 1, 0.78388], [0.73352, 1, 0.7792], [0.73627, 1, 0.77449], [0.73905, 1, 0.76975],
                         [0.74187, 1, 0.76498], [0.74471, 1, 0.76018], [0.74758, 1, 0.75535], [0.75049, 1, 0.75049],
                         [0.75342, 1, 0.7456], [0.75639, 1, 0.74067], [0.75939, 1, 0.73572], [0.76241, 1, 0.73074],
                         [0.76547, 1, 0.72572], [0.76856, 1, 0.72068], [0.77168, 1, 0.7156], [0.77483, 1, 0.71049],
                         [0.77801, 1, 0.70536], [0.78122, 1, 0.70019],
                         [0.78446, 1, 0.69499], [0.78773, 1, 0.68976], [0.79103, 1, 0.6845], [0.79437, 1, 0.67921],
                         [0.79773, 1, 0.67389], [0.80113, 1, 0.66854],
                         [0.80455, 1, 0.66316], [0.80801, 1, 0.65775], [0.81149, 1, 0.65231], [0.81501, 1, 0.64683],
                         [0.81855, 1, 0.64133], [0.82213, 1, 0.63579], [0.82574, 1, 0.63023], [0.82938, 1, 0.62463],
                         [0.83305, 1, 0.61901], [0.83675, 1, 0.61335], [0.84048, 1, 0.60766], [0.84424, 1, 0.60194],
                         [0.84803, 1, 0.5962], [0.85185, 1, 0.59042], [0.85571, 1, 0.58461], [0.85959, 1, 0.57877],
                         [0.8635, 1, 0.5729], [0.86745, 1, 0.56699], [0.87142, 1, 0.56106], [0.87543, 1, 0.5551],
                         [0.87946, 1, 0.54911], [0.88353, 1, 0.54308], [0.88763, 1, 0.53703], [0.89176, 1, 0.53094],
                         [0.89591, 1, 0.52483], [0.9001, 1, 0.51868], [0.90432, 1, 0.51251], [0.90857, 1, 0.5063],
                         [0.91285, 1, 0.50006], [0.91717, 1, 0.49379], [0.92151, 1, 0.48749], [0.92588, 1, 0.48116],
                         [0.93028, 1, 0.4748], [0.93472, 1, 0.46841], [0.93918, 1, 0.46199], [0.94368, 1, 0.45554],
                         [0.9482, 1, 0.44906], [0.95276, 1, 0.44255], [0.95734, 1, 0.436], [0.96196, 1, 0.42943],
                         [0.96661, 1, 0.42282], [0.97129, 1, 0.41619], [0.976, 1, 0.40952],
                         [0.98074, 1, 0.40283], [0.98551, 1, 0.3961], [0.99031, 1, 0.38934], [0.99514, 1, 0.38255],
                         [1, 1, 0.37573], [1, 0.99511, 0.37378],
                         [1, 0.99018, 0.37182], [1, 0.98523, 0.36986], [1, 0.98025, 0.36791], [1, 0.97523, 0.36595],
                         [1, 0.97019, 0.36399], [1, 0.96511, 0.36204], [1, 0.96, 0.36008], [1, 0.95487, 0.35812],
                         [1, 0.9497, 0.35616], [1, 0.9445, 0.35421], [1, 0.93927, 0.35225], [1, 0.93401, 0.35029],
                         [1, 0.92872, 0.34834], [1, 0.9234, 0.34638], [1, 0.91805, 0.34442], [1, 0.91267, 0.34247],
                         [1, 0.90726, 0.34051], [1, 0.90182, 0.33855], [1, 0.89634, 0.33659], [1, 0.89084, 0.33464],
                         [1, 0.8853, 0.33268], [1, 0.87974, 0.33072], [1, 0.87414, 0.32877], [1, 0.86852, 0.32681],
                         [1, 0.86286, 0.32485], [1, 0.85717, 0.3229], [1, 0.85146, 0.32094], [1, 0.84571, 0.31898],
                         [1, 0.83993, 0.31703], [1, 0.83412, 0.31507], [1, 0.82828, 0.31311], [1, 0.82241, 0.31115],
                         [1, 0.81651, 0.3092], [1, 0.81057, 0.30724], [1, 0.80461, 0.30528], [1, 0.79862, 0.30333],
                         [1, 0.79259, 0.30137], [1, 0.78654, 0.29941], [1, 0.78045, 0.29746], [1, 0.77434, 0.2955],
                         [1, 0.76819, 0.29354], [1, 0.76202, 0.29159], [1, 0.75581, 0.28963],
                         [1, 0.74957, 0.28767], [1, 0.7433, 0.28571], [1, 0.737, 0.28376], [1, 0.73068, 0.2818],
                         [1, 0.72432, 0.27984], [1, 0.71792, 0.27789],
                         [1, 0.7115, 0.27593], [1, 0.70505, 0.27397], [1, 0.69857, 0.27202], [1, 0.69206, 0.27006],
                         [1, 0.68551, 0.2681], [1, 0.67894, 0.26614], [1, 0.67233, 0.26419], [1, 0.6657, 0.26223],
                         [1, 0.65903, 0.26027], [1, 0.65234, 0.25832], [1, 0.64561, 0.25636], [1, 0.63885, 0.2544],
                         [1, 0.63206, 0.25245], [1, 0.62524, 0.25049], [1, 0.6184, 0.24853], [1, 0.61152, 0.24658],
                         [1, 0.6046, 0.24462], [1, 0.59766, 0.24266], [1, 0.59069, 0.2407], [1, 0.58369, 0.23875],
                         [1, 0.57666, 0.23679], [1, 0.56959, 0.23483], [1, 0.5625, 0.23288], [1, 0.55538, 0.23092],
                         [1, 0.54822, 0.22896], [1, 0.54103, 0.22701], [1, 0.53382, 0.22505], [1, 0.52657, 0.22309],
                         [1, 0.51929, 0.22114], [1, 0.51199, 0.21918], [1, 0.50465, 0.21722], [1, 0.49728, 0.21526],
                         [1, 0.48988, 0.21331], [1, 0.48245, 0.21135], [1, 0.47499, 0.20939], [1, 0.4675, 0.20744],
                         [1, 0.45997, 0.20548], [1, 0.45242, 0.20352], [1, 0.44484, 0.20157], [1, 0.43722, 0.19961],
                         [1, 0.42958, 0.19765], [1, 0.42191, 0.19569], [1, 0.4142, 0.19374],
                         [1, 0.40646, 0.19178], [1, 0.3987, 0.18982], [1, 0.3909, 0.18787], [1, 0.38307, 0.18591],
                         [1, 0.37521, 0.18395], [1, 0.36733, 0.182],
                         [1, 0.35941, 0.18004], [1, 0.35146, 0.17808], [1, 0.34347, 0.17613], [1, 0.33546, 0.17417],
                         [1, 0.32742, 0.17221], [1, 0.31935, 0.17025], [1, 0.31125, 0.1683], [1, 0.30311, 0.16634],
                         [1, 0.29495, 0.16438], [1, 0.28675, 0.16243], [1, 0.27853, 0.16047], [1, 0.27027, 0.15851],
                         [1, 0.26199, 0.15656], [1, 0.25367, 0.1546], [1, 0.24532, 0.15264], [1, 0.23694, 0.15068],
                         [1, 0.22853, 0.14873], [1, 0.2201, 0.14677], [1, 0.21163, 0.14481], [1, 0.20312, 0.14286],
                         [1, 0.19459, 0.1409], [1, 0.18603, 0.13894], [1, 0.17744, 0.13699], [1, 0.16882, 0.13503],
                         [1, 0.16016, 0.13307], [1, 0.15148, 0.13112], [1, 0.14277, 0.12916], [1, 0.13402, 0.1272],
                         [1, 0.12524, 0.12524], [0.99315, 0.12329, 0.12329], [0.98627, 0.12133, 0.12133],
                         [0.97936, 0.11937, 0.11937], [0.97242, 0.11742, 0.11742], [0.96545, 0.11546, 0.11546],
                         [0.95845, 0.1135, 0.1135], [0.95141, 0.11155, 0.11155], [0.94435, 0.10959, 0.10959],
                         [0.93726, 0.10763, 0.10763], [0.93013, 0.10568, 0.10568], [0.92298, 0.10372, 0.10372],
                         [0.91579, 0.10176, 0.10176], [0.90857, 0.099804, 0.099804], [0.90133, 0.097847, 0.097847],
                         [0.89405, 0.09589, 0.09589],
                         [0.88674, 0.093933, 0.093933], [0.8794, 0.091977, 0.091977], [0.87203, 0.09002, 0.09002],
                         [0.86463, 0.088063, 0.088063], [0.8572, 0.086106, 0.086106], [0.84974, 0.084149, 0.084149],
                         [0.84225, 0.082192, 0.082192], [0.83473, 0.080235, 0.080235], [0.82718, 0.078278, 0.078278],
                         [0.81959, 0.076321, 0.076321], [0.81198, 0.074364, 0.074364], [0.80434, 0.072407, 0.072407],
                         [0.79666, 0.07045, 0.07045], [0.78896, 0.068493, 0.068493], [0.78122, 0.066536, 0.066536],
                         [0.77345, 0.064579, 0.064579], [0.76566, 0.062622, 0.062622], [0.75783, 0.060665, 0.060665],
                         [0.74997, 0.058708, 0.058708], [0.74208, 0.056751, 0.056751], [0.73416, 0.054795, 0.054795],
                         [0.72621, 0.052838, 0.052838], [0.71823, 0.050881, 0.050881], [0.71022, 0.048924, 0.048924],
                         [0.70218, 0.046967, 0.046967], [0.6941, 0.04501, 0.04501], [0.686, 0.043053, 0.043053],
                         [0.67787, 0.041096, 0.041096], [0.6697, 0.039139, 0.039139], [0.66151, 0.037182, 0.037182],
                         [0.65328, 0.035225, 0.035225], [0.64503, 0.033268, 0.033268],
                         [0.63674, 0.031311, 0.031311], [0.62842, 0.029354, 0.029354],
                         [0.62008, 0.027397, 0.027397], [0.6117, 0.02544, 0.02544], [0.60329, 0.023483, 0.023483],
                         [0.59485, 0.021526, 0.021526], [0.58638, 0.019569, 0.019569], [0.57788, 0.017613, 0.017613],
                         [0.56935, 0.015656, 0.015656], [0.56079, 0.013699, 0.013699], [0.5522, 0.011742, 0.011742],
                         [0.54357, 0.0097847, 0.0097847], [0.53492, 0.0078278, 0.0078278],
                         [0.52624, 0.0058708, 0.0058708],
                         [0.51752, 0.0039139, 0.0039139], [0.50878, 0.0019569, 0.0019569], [0.5, 0, 0]]

    return (ListedColormap(jet_light_rgb_512))