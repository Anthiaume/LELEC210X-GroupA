# ruff: noqa: N806
from tracemalloc import start

import numpy as np
import scipy.signal as signal

BIT_RATE = 100e3
PREAMBLE = np.array([int(bit) for bit in f"{0xAAAAAAAA:0>32b}"])
SYNC_WORD = np.array([int(bit) for bit in f"{0x3E2A54B7:0>32b}"])

FPGA_FIR_TAPS = np.array(
    [
        -0.001201261290430126,
        0.0020488944185569607,
        -0.0020751053507837938,
        4.910806933254215e-18,
        0.004754535968663148,
        -0.00987450755161552,
        0.00995675888032359,
        -1.4391882903962387e-17,
        -0.018922538981281996,
        0.036214375130954504,
        -0.03468641976116993,
        2.4803862788187382e-17,
        0.06848299151299582,
        -0.15293237705130486,
        0.22297239138994396,
        0.7505245253702963,
        0.22297239138994396,
        -0.15293237705130486,
        0.06848299151299582,
        2.4803862788187385e-17,
        -0.034686419761169936,
        0.036214375130954504,
        -0.018922538981282003,
        -1.4391882903962393e-17,
        0.00995675888032359,
        -0.009874507551615532,
        0.004754535968663151,
        4.910806933254215e-18,
        -0.0020751053507837946,
        0.0020488944185569607,
        -0.001201261290430126,
    ]
)  # Example coefficients


class Chain:
    name: str = ""

    # Communication parameters

    bit_rate: float = BIT_RATE
    freq_dev: float = BIT_RATE/2
    # freq_dev: float = BIT_RATE/4

    osr_tx: int = 64
    osr_rx: int = 8

    preamble: np.ndarray = PREAMBLE
    sync_word: np.ndarray = SYNC_WORD

    payload_len: int = 800  # Number of bits per packet

    # Simulation parameters
    n_packets: int = 200  # Number of sent packets

    # Channel parameters
    sto_val: float = 2
    sto_range: float = 10 / BIT_RATE  # defines the delay range when random

    cfo_val: float = np.nan
    cfo_range: tuple[float, float] = (
        8_000,
        10_000,  # defines the CFO range when random (in Hz) #(1000 in old repo)
    )

    EsN0_range: np.ndarray = np.arange(0, 30, 1)

    # Lowpass filter parameters
    taps: np.ndarray = FPGA_FIR_TAPS  # specify None to make the simulator recompute the filter based on below spec
    numtaps: int = 100
    cutoff: float = 150e3  # BIT_RATE * osr_rx / 2.0001  # or 2*BIT_RATE,...

    # Tx methods

    def modulate(self, bits: np.array) -> np.array:
        """
        Modulates a stream of bits of size N
        with a given TX oversampling factor R (osr_tx).

        Uses Continuous-Phase FSK modulation.

        :param bits: The bit stream, (N,).
        :return: The modulates bit sequence, (N * R,).
        """
        fd = self.freq_dev  # Frequency deviation, Delta_f
        B = self.bit_rate  # B=1/T
        h = 2 * fd / B  # Modulation index
        R = self.osr_tx  # Oversampling factor

        x = np.zeros(len(bits) * R, dtype=np.complex64)
        ph = 2 * np.pi * fd * (np.arange(R) / R) / B  # Phase of reference waveform

        phase_shifts = np.zeros(
            len(bits) + 1
        )  # To store all phase shifts between symbols
        phase_shifts[0] = 0  # Initial phase

        for i, b in enumerate(bits):
            x[i * R : (i + 1) * R] = np.exp(1j * phase_shifts[i]) * np.exp(
                1j * (1 if b else -1) * ph
            )  # Sent waveforms, with starting phase coming from previous symbol
            phase_shifts[i + 1] = phase_shifts[i] + h * np.pi * (
                1 if b else -1
            )  # Update phase to start with for next symbol

        return x

    # Rx methods
    ideal_preamble_detect: bool = False
    use_dynamic_ppd: bool = True

    def preamble_detect(self, y: np.array) -> int | None:
        """
        Detect the preamble in a given received signal with hard thresholding.

        :param y: The received signal, (N * R,).
        :return: The index where the preamble starts,
            or None if not found.
        """
        raise NotImplementedError

    def preamble_detect_ppd(self, y: np.array) -> int | None:
        """
        Detect the preamble in a given received signal with sofft thresholding.

        :param y: The received signal, (N * R,).
        :return: The index where the preamble starts,
            or None if not found.
        """
        raise NotImplementedError

    ideal_cfo_estimation: bool = True

    #ideal_cfo_estimation: bool = False

    def cfo_estimation(self, y: np.array) -> float:
        """
        Estimates the CFO based on the received signal.

        :param y: The received signal, (N * R,).
        :return: The estimated CFO.
        """
        raise NotImplementedError

    ideal_sto_estimation: bool = True

    def sto_estimation(self, y: np.array, TYPE) -> float:
        """
        Estimates the STO based on the received signal.

        :param y: The received signal, (N * R,).
        :return: The estimated STO.
        """
        raise NotImplementedError

    def demodulate(self, y: np.array) -> np.array:
        """
        Demodulates the received signal.

        :param y: The received signal, (N * R,).
        :return: The signal, after demodulation.
        """

        raise NotImplementedError
    
    def Viterbi_code_m2(self, bits : np.array) -> np.array:

        raise NotImplementedError
    
    def viterbi_decode(self, received, g1, g2, m):
        """
        Décodage Viterbi pour code convolutif R=1/2, mémoire m
        received: liste de bits reçus [b0, b1, b2, ...]
        g1, g2: générateurs en octal
        m: mémoire du code
        """
        raise NotImplementedError
    
    def int_to_bits(self, n, width):
        raise NotImplementedError

    def xor_bits(self, bits):
        raise NotImplementedError
    def number2binary(self, x0,length):
        raise NotImplementedError
    def binary2number(self, x):
        raise NotImplementedError
    def poly2trellis(self, gn,gd):
        raise NotImplementedError
    def viterbi_decoder(self, R1,R0,symb_R1,symb_R0,len_b,x_tilde):
        raise NotImplementedError

class BasicChain(Chain):
    name = "Basic Tx/Rx chain"

    cfo_val, sto_val = np.nan, np.nan  # CFO and STO are random

    ideal_preamble_detect = False

    use_dynamic_ppd = True





    def preamble_detect_ppd(self, y, x_pr, PPD_algo="DEFAULT"):
        """Detect a preamble computing the received energy (average on a window)."""
        R = self.osr_rx
        fd = self.freq_dev
        B = self.bit_rate
        if PPD_algo == "DEFAULT":
            long_term_sum_W = 256
            short_term_sum_W = 32

            K = 2.3 * (short_term_sum_W / long_term_sum_W)

            long_window = np.ones(long_term_sum_W)
            short_window = np.ones(short_term_sum_W)
            yabs = (np.abs(y))**2  # Energy of the received signal (squared magnitude)
            ylen = len(y)
            
            long_sum = np.convolve(yabs, long_window, mode="valid")
            short_sum = np.convolve(yabs, short_window, mode="valid")
            offset = long_term_sum_W - short_term_sum_W
            short_sum_aligned = short_sum[offset : offset + len(long_sum)]

            detection = short_sum_aligned > (long_sum * K)
            detected_indices = np.where(detection)[0]
            first_idx = (
                (detected_indices[0] + long_term_sum_W + short_term_sum_W-40)
                if detected_indices.size > 0
                else None
            )
            return first_idx
        
        elif(PPD_algo == "DUALCORR"): 
            long_term_sum_W = 256*2
            short_term_sum_W = 32

            K = 2.3 * (short_term_sum_W / long_term_sum_W)

            long_window = np.ones(long_term_sum_W)
            short_window = np.ones(short_term_sum_W)
            yabs = (np.abs(y))**2  # Energy of the received signal (squared magnitude)
            ylen = len(y)
            
            long_sum = np.convolve(yabs, long_window, mode="valid")
            short_sum = np.convolve(yabs, short_window, mode="valid")
            offset = long_term_sum_W - short_term_sum_W
            short_sum_aligned = short_sum[offset : offset + len(long_sum)]

            detection = short_sum_aligned > (long_sum * K)
            detected_indices = np.where(detection)[0]

            # --- Correlation-based preamble detection ---
            if(detected_indices.size == 0):
                return None
            start = detected_indices[0] + long_term_sum_W + short_term_sum_W
            end   = start + 256   # marge de sécurité
            y_shifted = np.concatenate([np.zeros(2*self.osr_rx), y])
            corr_signal = np.abs(signal.correlate(y_shifted[start:end], x_pr[::64//self.osr_rx], mode='full'))
            n_peaks = 1  # nombre de pics
            idx = np.argpartition(corr_signal, -n_peaks)[-n_peaks:]   # indices des n plus grands
            top_values = corr_signal[idx]                 # valeurs correspondantes
            Decision_variable = np.mean(top_values)              # moyenne des n plus grands
            idx_first_peak = idx[np.argmax(top_values)]  # index du pic le plus grand

            noise_zone = yabs[:long_term_sum_W]  # Zone supposée sans signal pour estimer le bruit
            sigma2 = np.median(noise_zone)
            threshold = 5 * sigma2 * np.sqrt(256)  # Seuil basé sur l'énergie du bruit et la longueur du préambule
            # pattern_detected = np.max(corr_signal) > threshold
            pattern_detected = Decision_variable > threshold


            # --- Final index ---
            if detected_indices.size > 0 and pattern_detected:
                first_idx = detected_indices[0] + long_term_sum_W + short_term_sum_W -40
            else:
                first_idx = None

            return first_idx
        else:
            corr = np.correlate(y, x_pr, mode='valid')

            # Détecter le pic
            start_index = np.argmax(np.abs(corr)**2)
            print("Préambule détecté à l'échantillon :", start_index)

            yabs = (np.abs(y))**2  # Energy of the received signal (squared magnitude)
            noise_zone = yabs[:256]  # Zone supposée sans signal pour estimer le bruit
            sigma2 = np.median(noise_zone)
            threshold = 5 * sigma2 * np.sqrt(256)  # Seuil basé sur l'énergie du bruit et la longueur du préambule
            pattern_detected = np.max(np.abs(corr)**2) > threshold


            # --- Final index ---
            if pattern_detected:
                first_idx = start_index
            else:
                first_idx = None

            return first_idx

        


    def preamble_detect(self, y):
        """Detect a preamble computing the received energy (average on a window)."""
        L = 4 * self.osr_rx
        y_abs = np.abs(y)

        for i in range(0, int(len(y) / L)):
            sum_abs = np.sum(y_abs[i * L : (i + 1) * L])
            if sum_abs > (L - 1):  # fix threshold
                return i * L

        return None

    ideal_cfo_estimation = True

    def cfo_estimation(self, y):
        # """Estimates CFO using Moose algorithm, on first samples of preamble."""
        # # TO DO: extract 2 blocks of size N*R at the start of y
        # R = self.osr_rx
        # N = 4  # You can change this value if needed
        # y_sequence = y[0 : 2*R*N]  # Extract first 2 symbols (4 samples each)
        # # TO DO: apply the Moose algorithm on these two blocks to estimate the CFO
        
        # alpha_hat = np.sum(y_sequence[-N*R:]*np.conj(y_sequence[:N*R]))
        # cfo_est = np.angle(alpha_hat)/(2*np.pi*N/(self.bit_rate))
        # print("CFO estimation (Hz): ", cfo_est)

        # return cfo_est
        R = self.osr_rx
        B = self.bit_rate
        """
        Estimation ultra-rapide du CFO (Carrier Frequency Offset)
        avec l'algorithme de Moose, sur les premiers échantillons du préambule.
        """
        N = 4  # Nombre de symboles utilisés (modifiable selon le préambule)
        M = N * R  # Taille d’un bloc en échantillons

        # Pas de copie mémoire inutile : utilisation directe de vues sur y
        y1 = y[:M]
        y2 = y[M:2*M]

        # Produit conjugué cumulatif vectorisé (Moose)
        alpha_hat = np.vdot(y1, y2)  # np.vdot = sum(y1*conj(y2)), très optimisé en C

        # Estimation du décalage fréquentiel
        cfo_est = np.angle(alpha_hat) * B / (2 * np.pi * N)
        # print("CFO estimation (Hz): ", cfo_est )
        return cfo_est

    ideal_sto_estimation = True

    def sto_estimation(self, y, TYPE="ML", preamble=None):
        """Estimates symbol timing (fractional) based on phase shifts."""
        R = self.osr_rx

        if TYPE == "GARDNER":
            # Computation of Gardner timing error
            mu = 0.0                 # phase fractionnaire
            i = 0                    # index dans le signal
            out = []                 # symboles récupérés
            err = []                 # erreur Gardner
            mu_hist = []             # historique du timing
            sps = R                   # samples per symbol
            gain = 0.01              # gain de la boucle de timing
            rx  = y                     # signal reçu

            while i + sps < len(rx)-1:

                # interpolation linéaire pour échantillonner à phase mu
                idx = int(i + mu)
                frac = mu - int(mu)

                s0 = rx[idx]
                s1 = rx[idx+1]


                sample = (1-frac)*s0 + frac*s1

                # échantillons pour Gardner
                mid = rx[idx + sps//2]
                prev = rx[idx - sps//2] if idx - sps//2 >= 0 else 0

                # erreur Gardner
                e = np.real((mid - prev) * np.conj(rx[idx]))


                err.append(e)

                # mise à jour timing
                mu = (mu + gain * e)%1

                mu_hist.append(mu)

                out.append(sample)

                # avancer d'un symbole
                i += sps
                # estimation tau
            tau_est = np.array(mu_hist) / sps
            tau_final = np.mean(tau_est[-100:])

            # print(tau_final)
            return int(tau_final)  # Retourne la partie fractionnaire du timing offset


        if TYPE == "ML":

            metric_best = -np.inf
            best_tau = 0

            s_ref = preamble[::self.osr_tx//R]  # Référence de préambule échantillonnée à la bonne fréquence
            L = len(s_ref)

            for tau in range(R):
                if tau + L > len(y):
                    break

                segment = y[tau:tau+L]

                # Noncoherent ML-like metric (phase marginalized)
                # corr = np.sum(segment * np.conj(s_ref))
                corr = np.correlate(segment, s_ref, mode='valid')[0]  # Correlation linéaire
                metric = (np.abs(corr)**2) / np.sum(np.abs(s_ref)**2)  # Normalized energy of correlation

                if metric > metric_best:
                    metric_best = metric
                    best_tau = tau

            return np.int64(best_tau)
        if TYPE == "DER":
            # Computation of derivatives of phase function
            phase_function = np.unwrap(np.angle(y))
            phase_derivative_1 = phase_function[1:] - phase_function[:-1]
            phase_derivative_2 = np.abs(phase_derivative_1[1:] - phase_derivative_1[:-1])

            sum_der_saved = -np.inf
            save_i = 0
            for i in range(0, R):
                sum_der = np.sum(phase_derivative_2[i::R])  # Sum every R samples

                if sum_der > sum_der_saved:
                    sum_der_saved = sum_der
                    save_i = i
            return np.mod(save_i + 1, R)

    def demodulate(self, y):
        # """Non-coherent demodulator."""
        R = self.osr_rx  # Receiver oversampling factor
        # nb_syms = len(y) // R  # Number of CPFSK symbols in y
        fd = self.freq_dev  # Frequency deviation, Delta_f
        B = self.bit_rate  # B=1/T
        """
        Démodulateur non-cohérent vectorisé et ultra-rapide (FSK).
        """
        nb_syms = len(y) // R
        y = y[:nb_syms * R].reshape(nb_syms, R)  # Vue sans recopie

        # Références de phase
        ph = 2 * np.pi * fd * np.arange(R) / (R * B)
        s_0 = np.exp(-1j * ph)
        s_1 = np.exp( 1j * ph)

        # Corrélation vectorisée sur tout le signal
        r0 = np.abs(y @ s_0)  # Produit matriciel = somme sur R pour chaque symbole
        r1 = np.abs(y @ s_1)

        # Décision binaire vectorisée
        bits_hat = (r1 < r0).astype(np.uint8)

        return bits_hat

    def richardson_extrapolation(estimates, hs, p):
        """
        estimates : valeurs E(h_i)
        hs : pas correspondants
        p : ordre de l'erreur dominante
        """
        n = len(estimates)

        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                A[i, j] = hs[i]**(p + j)

        b = np.array(estimates)

        coeffs = np.linalg.solve(A, b)
        return coeffs[0]
    
    # def Viterbi_code(self, bits):
    #     datas = np.zeros((len(bits), 2))
    #     datas[0, 0] = (bits[0])%2
    #     datas[0, 1] = bits[0]
    #     datas[1, 0] = (bits[0] + bits[1])%2
    #     datas[1, 1] = bits[0]
    #     datas[2, 0] = (bits[0] + bits[1] + bits[2])%2
    #     datas[2, 1] = (bits[0] + bits[2])%2
    #     datas[3, 0] = (bits[0] + bits[1] + bits[2] + bits[3])%2
    #     datas[3, 1] = (bits[0] + bits[1] + bits[3])%2
    #     datas[4, 0] = (bits[1] + bits[2] + bits[3] + bits[4])%2
    #     datas[4, 1] = (bits[1] + bits[2] + bits[4])%2
    #     datas[5, 0] = (bits[2] + bits[3] + bits[4] + bits[5])%2
    #     datas[5, 1] = (bits[0] + bits[2] + bits[3] + bits[5])%2
    #     # datas[6, 0] = bits[0] + bits[3] + bits[4] + bits[5] + bits[6]
    #     # datas[6, 1] = bits[0] + bits[1] + bits[3] + bits[4] + bits[6]

    #     for i in range(6, len(bits)):
    #         datas[i,0] = (bits[i]+bits[i-1]+bits[i-2]+bits[i-3]+bits[i-6]) % 2
    #         datas[i,1] = (bits[i]+bits[i-2]+bits[i-3]+bits[i-5]+bits[i-6]) % 2

    #     bits_Viterbi = np.zeros(2*len(bits))
    #     for i in range(len(bits)):
    #         bits_Viterbi[i] = datas[i, 0]
    #         bits_Viterbi[i+1] = datas[i, 1]
    #     return bits_Viterbi



    # def Viterbi_code_m2(self, bits):
    #     """
    #     Encodage convolutif R=1/2, mémoire m=2 (K=3)
    #     bits: tableau 1D des bits à encoder
    #     Retourne bits encodés (séquence 2x plus longue)
    #     """
    #     m = 2
    #     K = m + 1
    #     n = len(bits)
    #     datas = np.zeros((n, 2), dtype=int)
        
    #     # Générateurs classiques m=2, R=1/2
    #     # g1=111b -> 7 octal, g2=101b -> 5 octal
    #     g1_bits = [1, 1, 1]
    #     g2_bits = [1, 0, 1]
        
    #     # Encodage bit par bit
    #     for i in range(n):
    #         # récupérer les K bits du registre (0 si i-j <0)
    #         reg_bits = [bits[i - j] if i - j >= 0 else 0 for j in range(K)]
            
    #         # calcul des sorties modulo 2
    #         out1 = sum([b & g for b, g in zip(reg_bits, g1_bits)]) % 2
    #         out2 = sum([b & g for b, g in zip(reg_bits, g2_bits)]) % 2
            
    #         datas[i, 0] = out1
    #         datas[i, 1] = out2
        
    #     # Construire la séquence codée finale (alternance des sorties)
    #     bits_Viterbi = np.zeros(2 * n, dtype=int)
    #     for i in range(n):
    #         bits_Viterbi[2*i] = datas[i, 0]
    #         bits_Viterbi[2*i + 1] = datas[i, 1]
        
    #     return bits_Viterbi
    

    # def int_to_bits(self, n, width):
    #     return [int(b) for b in format(n, f'0{width}b')]

    # def xor_bits(self, bits):
    #     result = 0
    #     for b in bits:
    #         result ^= b
    #     return result
        
    # def viterbi_decode(self, received, g1, g2, m):
    #     """
    #     Décodage Viterbi pour code convolutif R=1/2, mémoire m
    #     received: liste de bits reçus [b0, b1, b2, ...]
    #     g1, g2: générateurs en octal
    #     m: mémoire du code
    #     """
    #     K = m + 1
    #     n_states = 2 ** m
    #     path_metric = [float('inf')] * n_states
    #     path_metric[0] = 0
    #     paths = [[] for _ in range(n_states)]

    #     # Préparer les générateurs en binaire
    #     g1_bits = self.int_to_bits(g1, K)
    #     g2_bits = self.int_to_bits(g2, K)

    #     n_steps = len(received) // 2  # 2 bits par étape

    #     for t in range(n_steps):
    #         rec_pair = received[2*t : 2*t+2]
    #         new_metric = [float('inf')] * n_states
    #         new_paths = [[] for _ in range(n_states)]

    #         for state in range(n_states):
    #             for bit_in in [0,1]:
    #                 prev_state = ((state >> 1) | (bit_in << (m-1))) & (n_states-1)
                    
    #                 # Générer les bits codés pour cette transition
    #                 reg_bits = self.int_to_bits(prev_state, m)
    #                 reg_bits = [bit_in] + reg_bits  # registre complet K bits
    #                 out1 = self.xor_bits([b & g for b,g in zip(reg_bits, g1_bits)])
    #                 out2 = self.xor_bits([b & g for b,g in zip(reg_bits, g2_bits)])
    #                 metric = path_metric[prev_state] + (out1 != rec_pair[0]) + (out2 != rec_pair[1])

    #                 if metric < new_metric[state]:
    #                     new_metric[state] = metric
    #                     new_paths[state] = paths[prev_state] + [bit_in]

    #         path_metric = new_metric
    #         paths = new_paths

    #     # Trouver le chemin minimal
    #     min_state = path_metric.index(min(path_metric))
    #     decoded_bits = paths[min_state]  # bits estimés envoyés
    #     reconstructed_coded_bits = []
    #     state = 0
    #     for bit in decoded_bits:
    #         reg_bits = [bit] + self.int_to_bits(state, m)
    #         out1 = self.xor_bits([b & g for b,g in zip(reg_bits, g1_bits)])
    #         out2 = self.xor_bits([b & g for b,g in zip(reg_bits, g2_bits)])
    #         reconstructed_coded_bits.extend([out1, out2])
    #         state = ((state >> 1) | (bit << (m-1))) & (2**m-1)
    #     return np.array(reconstructed_coded_bits)

    # @jit(nopython=True,error_model="numpy")
    def number2binary(x0,length):
        binary_array = np.zeros((length,))
    
        x = x0
        i = 0
    
        while x > 1 and i < length:
            binary_array[i] = x % 2
            x = int(x / 2)
            i = i + 1
    
        if x > 0 and i < length:
            binary_array[i] = 1
    
        return binary_array[::-1]
    
    def binary2number(self, x):
        out = 0
        for i in x:
            out = 2*out + i
        return out
    
    def poly2trellis(self, gn,gd):
        M = max(len(gn),len(gd)) - 1
        nb_states = 2**M
    
        alpha = np.zeros((M+1,))
        beta = np.zeros((M+1,))
    
        alpha[:len(gn)] = gn
        beta[:len(gd)] = gd

        R1 = np.zeros((nb_states,),dtype=np.int32)
        R0 = np.zeros((nb_states,),dtype=np.int32)
    
        out_R1 = np.zeros((nb_states,2),dtype=np.int32)
        out_R0 = np.zeros((nb_states,2),dtype=np.int32)
    
        out_R1[:,0] = 1
    
        for i in range(nb_states):
            states = np.zeros((M+1,))
            states[:M] = self.number2binary(i,M)[::-1]
        
            y_1 = (alpha[0] + states[0]) % 2
            y_0 = states[0]
        
            new_states_1 = (alpha[1:] + beta[1:]*y_1 + states[1:]) % 2
            new_states_0 = (beta[1:]*y_0 + states[1:]) % 2
        
            R1[i] = self.binary2number(new_states_1[::-1])
            R0[i] = self.binary2number(new_states_0[::-1])
        
            out_R1[i,1] = int(y_1)
            out_R0[i,1] = int(y_0)
    
        return R1,R0,out_R1,out_R0



    def viterbi_decoder(self, R1,R0,symb_R1,symb_R0,len_b,x_tilde):
        def dist(a,b):
            return np.abs(a-b)**2
    
        N_b = int(len(x_tilde)/len_b)
    
        x_tilde_b = np.reshape(x_tilde,(N_b,len_b))
        u_hat_b = np.zeros(x_tilde_b.shape,dtype=np.int32)
    
        nb_states = len(R1)

        for i in range(N_b):          
            x_tilde_i  = x_tilde_b[i,:]
            u_hat_i = u_hat_b[i,:]
        
            bits = np.zeros((nb_states,len_b))
            weights = np.inf*np.ones((nb_states,))
            weights[0] = 0
        
            new_states = np.zeros((2,nb_states))
            new_weights = np.zeros((2,nb_states))
            new_bits = np.zeros((2,nb_states,len_b))  
        
            for j in range(len_b):
                for k in range(nb_states):
                    new_states[1,k] = R1[k]
                    new_states[0,k] = R0[k]
                    new_weights[1,k] = weights[k] + dist(x_tilde_i[j],symb_R1[k])
                    new_weights[0,k] = weights[k] + dist(x_tilde_i[j],symb_R0[k])      
                    new_bits[1,k,:] = bits[k,:]
                    new_bits[0,k,:] = bits[k,:]
                    new_bits[1,k,j] = 1
                
                for k in range(nb_states):
                    idx_0_filled = False
                    for l in range(nb_states):
                        if new_states[0,l] == k:
                            if idx_0_filled:
                                idx_10 = 0
                                idx_11 = l
                            else:
                                idx_00 = 0
                                idx_01 = l
                                idx_0_filled = True
                            
                        if new_states[1,l] == k:
                            if idx_0_filled:
                                idx_10 = 1
                                idx_11 = l
                            else:
                                idx_00 = 1
                                idx_01 = l
                                idx_0_filled = True
                
                    if new_weights[idx_00,idx_01] <= new_weights[idx_10,idx_11]:
                        weights[k] = new_weights[idx_00,idx_01]
                        bits[k,:] = new_bits[idx_00,idx_01,:]
                    else:
                        weights[k] = new_weights[idx_10,idx_11]
                        bits[k,:] = new_bits[idx_10,idx_11,:]

            final_weight = np.inf
            for k in range(nb_states):
                if weights[k] < final_weight:
                    final_weight = weights[k]
                    u_hat_i[:] = bits[k,:]
    
        u_hat = np.reshape(u_hat_b,(u_hat_b.size,))
        return u_hat

    # @jit(nopython=True,error_model="numpy")
    def interleaver(x,pattern):
        Nb = int(len(x)/len(pattern))
        x_matrix = np.reshape(x,(Nb,len(pattern)))
        y_matrix = x_matrix[:,pattern-1]
        y = np.reshape(y_matrix,(len(x),))
        return y



        