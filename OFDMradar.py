import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows as windows
from TargetDetection import TargetDetector


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
                 num_targets: int, target_list):
        self.n = n
        self.m = m
        self.N_guard = N_guard
        self.QAM_order = QAM_order
        self.num_targets = num_targets
        self.target_list = target_list
        self.deltaf = deltaf
        self.Tsym = 1 / deltaf
        self.Tsym += self.Tsym / 8
        self.SNR_linear = 10 ** (SNR_db / 10.0)
        self.upsample_factor = upsample_factor

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
        carrier_freq: float = 28e9
        channel_matrix = np.zeros((self.n + self.N_guard, self.m), dtype=complex)
        target_matrix = np.zeros((self.n + self.N_guard, self.m), dtype=complex)

        for target in self.target_list:
            print(target.target_rcs, target.target_distance, target.target_delay, target.target_doppler)

            # b_target = np.sqrt((lightspeed * target.target_rcs) / ((4 * np.pi) ** 3 * target.target_distance ** 4 * carrier_freq ** 2))
            # print(b_target)

            b_target = 1
            for m in range(self.m):
                frequency_shift = np.exp(1j * (2 * np.pi) * target.target_doppler * self.Tsym * m)

                for n in range(self.n):
                    phase_shift = np.exp(-1j * (2 * np.pi) * (n * self.deltaf) * target.target_delay)
                    target_matrix[n + self.N_guard, m] = b_target * frequency_shift * phase_shift * \
                                                         np.exp(1j * (- 2 * np.pi * carrier_freq * target.target_delay)
                                                                )
            target_matrix *= np.exp(1j * np.random.uniform(0, 2 * np.pi))
            channel_matrix += target_matrix
        return channel_matrix

    def window_generator(self, received_frame):
        print("---------------WINDOWING-----------------")

        sidelobe_attenuation = 100

        single_window_matrix_n = windows.chebwin(self.n, sidelobe_attenuation)
        win_normalization_n = (1 / self.n) * np.sum(single_window_matrix_n ** 2)
        single_window_matrix_n = np.multiply(single_window_matrix_n, np.sqrt(1 / win_normalization_n))
        check_win_n_normalization = (1 / self.n) * np.sum(single_window_matrix_n ** 2)
        print(f"normalization of N window matrix is: {check_win_n_normalization}")

        test_window = False

        single_window_matrix_m = windows.chebwin(self.m, sidelobe_attenuation)
        win_normalization_m = (1 / self.m) * np.sum(single_window_matrix_m ** 2)
        single_window_matrix_m = np.multiply(single_window_matrix_m, np.sqrt(1 / win_normalization_m))

        window_matrix = np.outer(single_window_matrix_n, single_window_matrix_m)

        # check_win_2d_normalization = (1 / (self.N * self.N)) * np.sum(window_matrix ** 2)
        # print(f"normalization of 2D window matrix is: {check_win_2d_normalization}")

        main_lobe_width_n = (1.46 * np.pi * (np.log10(2) + sidelobe_attenuation / 20)) / (self.n - 1)
        main_lobe_width_n = np.ceil(self.n * main_lobe_width_n)
        print(f"main lobe width in n is: {main_lobe_width_n}")

        main_lobe_width_m = (1.46 * (np.log10(2) + sidelobe_attenuation / 20)) / (self.m - 1)
        main_lobe_width_m = np.ceil(self.m * main_lobe_width_m)
        print(f"main lobe width in m is: {main_lobe_width_m}")

        main_lobe_width = lambda x: (1.46 * (np.log10(2) + sidelobe_attenuation / 20)) / (x - 1)
        main_lobe_normalized = [main_lobe_width(self.n), main_lobe_width(self.m)]
        print(f"main lobe dimensions are: {main_lobe_normalized}")

        if test_window:
            plt.figure(figsize=(20, 20))
            plt.plot(single_window_matrix_n)

            plt.figure(figsize=(20, 20))
            plt.plot(20 * np.log10((np.abs(np.fft.fftshift(np.fft.fft(single_window_matrix_n, self.n))))))

            check_win_m_normalization = (1 / self.m) * np.sum(single_window_matrix_m ** 2)
            print(f"normalization of M window matrix is: {check_win_m_normalization}")
            win_normalization_2d = (1 / (self.n * self.n)) * np.sum(window_matrix ** 2)

            # plt.plot(X, np.fft.fftshift(np.fft.fft(single_window_matrix_n, self.n)))

            print("--------------------------------")
            print(f"Norm of N window matrix is: {np.linalg.norm(single_window_matrix_n)}")
            print(f"Norm of M window matrix is: {np.linalg.norm(single_window_matrix_m)}")
            print(f"norm of window matrix is: {np.linalg.norm(window_matrix)}")
            print(f"window normalization constant is: {win_normalization_2d}")

            SNR_loss_n = (1 / (self.n * np.inner(single_window_matrix_n, single_window_matrix_n))) * (
                np.square(np.abs(np.sum(single_window_matrix_n))))
            SNR_loss_n = 10 * np.log10(SNR_loss_n)
            SNR_loss_m = (1 / (self.m * np.inner(single_window_matrix_m, single_window_matrix_m))) * np.square(
                np.abs(np.sum(single_window_matrix_m)))
            SNR_loss_m = 10 * np.log10(SNR_loss_m)

            print("--------------------------------")
            print(f"SNR_loss_N DUE TO WINDOWING: {SNR_loss_n}")
            print(f"SNR_loss_M DUE TO WINDOWING: {SNR_loss_m}")
            print(f"TOTAL SNR_loss is: {SNR_loss_n + SNR_loss_m}")

        return window_matrix, main_lobe_normalized

    def periodgram(self, received_fx, windowing: bool = False):
        N_subcarriers = self.n * self.upsample_factor
        M_symbols = self.m * self.upsample_factor
        received_frame = received_fx

        # print("--------------------------------")
        # print(f"received_frame BEFORE WINDOWING{received_frame}")

        if windowing:
            window_matrix, main_lobe_normalized = self.window_generator(received_frame)
            received_frame = np.multiply(received_frame, window_matrix)
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

        fig = plt.figure(figsize=(20, 20))
        Y = np.linspace(0, N_per, N_per)
        # Y = np.multiply(Y, (lightspeed / (2 * self.deltaf * N_per)))
        X = np.linspace(-M_per / 2, M_per / 2, M_per)
        # X = np.multiply(X, (lightspeed / (2 * carrier_freq * self.Tsym * M_per)))
        plt.pcolormesh(X, Y, periodgram, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Relative Speed')
        plt.ylabel('Distance')
        plt.title('Periodogram')

        return
