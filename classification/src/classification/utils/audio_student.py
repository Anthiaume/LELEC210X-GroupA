import random

import librosa
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from numpy import ndarray
from scipy import signal
from scipy.signal import fftconvolve

# -----------------------------------------------------------------------------
"""
Synthesis of the classes in :
- AudioUtil : util functions to process an audio signal.
- Feature_vector_DS : Create a dataset class for the feature vectors.
"""
# -----------------------------------------------------------------------------


class AudioUtil:
    """
    Define a new class with util functions to process an audio signal.
    """

    def open(audio_file) -> tuple[ndarray, int]:
        """
        Load an audio file.

        :param audio_file: The path to the audio file.
        :return: The audio signal as a tuple (signal, sample_rate).
        """
        sig, sr = sf.read(audio_file)
        if sig.ndim > 1:
            sig = sig[:, 0]
        return (sig, sr)

    def play(audio):
        """
        Play an audio file.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        """
        sig, sr = audio
        sd.play(sig, sr)

    def normalize(audio, target_dB=52) -> tuple[ndarray, int]:
        """
        Normalize the energy of the signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param target_dB: The target energy in dB.
        """
        sig, sr = audio
        sign = sig / np.sqrt(np.sum(np.abs(sig) ** 2))
        C = np.sqrt(10 ** (target_dB / 10))
        sign *= C
        return (sign, sr)

    def resample(audio, newsr=11025) -> tuple[np.ndarray, int]:
        """
        Resample to target sampling frequency.
    
        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param newsr: The target sampling frequency.
        """

        resig = signal.resample(audio[0], int(round(len(audio[0]) * newsr / audio[1])))
    
        return (resig, newsr)

    def pad_trunc(audio, max_ms) -> tuple[ndarray, int]:
        """
        Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param max_ms: The target length in milliseconds.
        """
        sig, sr = audio
        sig_len = len(sig)
        max_len = int(sr * max_ms / 1000)

        if sig_len > max_len:
            # Truncate the signal to the given length at random position
            # begin_len = random.randint(0, max_len)
            begin_len = 0
            sig = sig[begin_len : begin_len + max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = np.zeros(pad_begin_len)
            pad_end = np.zeros(pad_end_len)

            # sig = np.append([pad_begin, sig, pad_end])
            sig = np.concatenate((pad_begin, sig, pad_end))

        return (sig, sr)

    def scaling(audio, scaling_limit=5) -> tuple[ndarray, int]:
        """
        Augment the audio signal by scaling it by a random factor.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param scaling_limit: The maximum scaling factor.
        """
        ### TO COMPLETE
        sig, sr = audio
        scaling_factor = random.uniform(0, scaling_limit)
        sig = sig * scaling_factor
        return (sig, sr)

    def add_noise(audio, sigma=0.05) -> tuple[ndarray, int]:
        """
        Augment the audio signal by adding gaussian noise.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param sigma: Standard deviation of the gaussian noise.
        """
        ### TO COMPLETE
        sig, sr = audio
        sig += np.random.normal(0, sigma, len(sig))
        return (sig, sr)

    def echo(audio, nechos=2) -> tuple[ndarray, int]:
        """
        Add echo to the audio signal by convolving it with an impulse response. The taps are regularly spaced in time and each is twice smaller than the previous one.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param nechos: The number of echoes.
        """
        sig, sr = audio
        sig_len = len(sig)
        echo_sig = np.zeros(sig_len)
        echo_sig[0] = 1
        echo_sig[(np.arange(nechos) / nechos * sig_len).astype(int)] = (
            1 / 2
        ) ** np.arange(nechos)

        sig = fftconvolve(sig, echo_sig, mode="full")[:sig_len]
        return (sig, sr)

    def filter(audio, filt) -> tuple[ndarray, int]:
        """
        Filter the audio signal with a provided filter. Note the filter is given for positive frequencies only and is thus
        symmetrized in the function.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param filt: The filter to apply.
        """
        ### TO COMPLETE
        sig, sr = audio
        filt_full = np.concatenate((filt, filt[-2:0:-1]))
        sig_fft = np.fft.fft(sig)
        sig_fft_filtered = sig_fft * filt_full
        sig_filtered = np.fft.ifft(sig_fft_filtered).real
        return (sig_filtered, sr)

    def add_bg(audio, dataset, num_sources=1, max_ms=5000, amplitude_limit=0.1) -> tuple[ndarray, int]:
        """
        Adds up sounds uniformly chosen at random to audio.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param dataset: The dataset to sample from.
        :param num_sources: The number of sounds to add.
        :param max_ms: The maximum duration of the sounds to add.
        :param amplitude_limit: The maximum amplitude of the added sounds.
        """
        ### TO COMPLETE
        sig, sr = audio
        sig_len = len(sig)
        extraction = Feature_vector_DS(dataset)

        for i in range(num_sources):
            # Choisir une classe et un fichier au hasard et recuperer le signal audio
            bg_cls = random.choice(list(dataset.files.keys()))
            bg_file = random.choice(dataset.files[bg_cls])
            bg_audio, sr_bg = extraction.get_audiosignal([bg_cls, dataset.files[bg_cls].index(bg_file)])

            # Vérifier que le signal n'est pas trop long
            if len(bg_audio) > sr_bg * max_ms / 1000:
                bg_audio = bg_audio[: int(sr_bg * max_ms / 1000)]

            # Vérifier que le signal peut rentrer dans le signal principal
            if len(bg_audio) >= sig_len:
                bg_audio = bg_audio[:sig_len]

            # Ajouter le signal avec une amplitude aléatoire à une position aléatoire
            amplitude = random.uniform(0, amplitude_limit)
            index_slice = random.randint(0, sig_len - len(bg_audio))
            sig[index_slice : index_slice+len(bg_audio)] += amplitude * bg_audio

        return (sig, sr)
    
    def q15_saturate(x):
        return np.clip(x, -32768, 32767).astype(np.int16)

    def q15_mult(a, b):
        prod = ((a.astype(np.int32) * b.astype(np.int32)) >> 15).astype(np.int16)
        return AudioUtil.q15_saturate(prod)
    
    def adc_to_q15(samples):
        
        #samples = np.round(samples.astype(np.int32)*(1<<15))
        #samples = np.clip(samples,-32768,32767).astype(np.int16)
        # STEP 0.1
        # samples = samples.mean(axis=1) if samples.ndim>1 else samples
        # samples = samples / np.max(np.abs(samples))
        # samples = np.clip(samples, -1.0, 0.999969482421875)

        # samples = ((samples + 1.0) * 2047.5).astype(np.int16)
        # samples = samples << 3
        # samples = samples - (1 << 14)
        # return samples
        
        samples = samples.mean(axis=1) if samples.ndim>1 else samples
        samples = samples / np.max(np.abs(samples))
        samples = np.clip(samples, -1.0, 0.999969482421875)

        samples = (samples * 32767).astype(np.int16)
        samples = samples << 3

        # STEP 0.2
        samples = samples - (1 << 14)

        return samples
        
    def apply_window_q15(frames, Nft):

        #hamming = np.hamming(Nft)
        hamming = np.array([2621, 
2622, 
2626, 
2632, 
2640, 
2650, 
2662, 
2677, 
2694, 
2714, 
2735, 
2759, 
2785, 
2814, 
2844, 
2877, 
2912, 
2949, 
2989, 
3031, 
3075, 
3121, 
3169, 
3220, 
3273, 
3328, 
3385, 
3444, 
3506, 
3569, 
3635, 
3703, 
3773, 
3845, 
3919, 
3996, 
4074, 
4155, 
4237, 
4321, 
4408, 
4496, 
4587, 
4680, 
4774, 
4870, 
4969, 
5069, 
5171, 
5275, 
5381, 
5489, 
5599, 
5710, 
5824, 
5939, 
6056, 
6174, 
6295, 
6417, 
6541, 
6666, 
6793, 
6922, 
7052, 
7185, 
7318, 
7453, 
7590, 
7728, 
7868, 
8010, 
8152, 
8296, 
8442, 
8589, 
8737, 
8887, 
9038, 
9191, 
9344, 
9499, 
9655, 
9813, 
9971, 
10131, 
10292, 
10454, 
10617, 
10781, 
10946, 
11113, 
11280, 
11448, 
11617, 
11787, 
11958, 
12130, 
12303, 
12476, 
12650, 
12825, 
13001, 
13178, 
13355, 
13533, 
13711, 
13890, 
14070, 
14250, 
14431, 
14612, 
14793, 
14975, 
15158, 
15341, 
15524, 
15708, 
15892, 
16076, 
16260, 
16445, 
16629, 
16814, 
16999, 
17185, 
17370, 
17555, 
17741, 
17926, 
18111, 
18296, 
18481, 
18667, 
18851, 
19036, 
19221, 
19405, 
19589, 
19773, 
19956, 
20139, 
20322, 
20504, 
20686, 
20867, 
21048, 
21229, 
21409, 
21588, 
21767, 
21945, 
22122, 
22299, 
22475, 
22651, 
22825, 
22999, 
23172, 
23344, 
23516, 
23686, 
23856, 
24025, 
24192, 
24359, 
24525, 
24689, 
24853, 
25016, 
25177, 
25337, 
25496, 
25654, 
25811, 
25967, 
26121, 
26274, 
26426, 
26576, 
26725, 
26873, 
27019, 
27164, 
27308, 
27450, 
27590, 
27729, 
27867, 
28003, 
28137, 
28270, 
28401, 
28531, 
28659, 
28785, 
28910, 
29033, 
29154, 
29274, 
29391, 
29507, 
29622, 
29734, 
29845, 
29953, 
30060, 
30165, 
30268, 
30370, 
30469, 
30566, 
30662, 
30755, 
30847, 
30936, 
31024, 
31109, 
31193, 
31274, 
31354, 
31431, 
31506, 
31579, 
31650, 
31719, 
31786, 
31851, 
31913, 
31974, 
32032, 
32088, 
32142, 
32194, 
32243, 
32291, 
32336, 
32379, 
32419, 
32458, 
32494, 
32528, 
32560, 
32589, 
32617, 
32642, 
32664, 
32685, 
32703, 
32719, 
32733, 
32744, 
32753, 
32760, 
32764, 
32767, 
32767, 
32764, 
32760, 
32753, 
32744, 
32733, 
32719, 
32703, 
32685, 
32664, 
32642, 
32617, 
32589, 
32560, 
32528, 
32494, 
32458, 
32419, 
32379, 
32336, 
32291, 
32243, 
32194, 
32142, 
32088, 
32032, 
31974, 
31913, 
31851, 
31786, 
31719, 
31650, 
31579, 
31506, 
31431, 
31354, 
31274, 
31193, 
31109, 
31024, 
30936, 
30847, 
30755, 
30662, 
30566, 
30469, 
30370, 
30268, 
30165, 
30060, 
29953, 
29845, 
29734, 
29622, 
29507, 
29391, 
29274, 
29154, 
29033, 
28910, 
28785, 
28659, 
28531, 
28401, 
28270, 
28137, 
28003, 
27867, 
27729, 
27590, 
27450, 
27308, 
27164, 
27019, 
26873, 
26725, 
26576, 
26426, 
26274, 
26121, 
25967, 
25811, 
25654, 
25496, 
25337, 
25177, 
25016, 
24853, 
24689, 
24525, 
24359, 
24192, 
24025, 
23856, 
23686, 
23516, 
23344, 
23172, 
22999, 
22825, 
22651, 
22475, 
22299, 
22122, 
21945, 
21767, 
21588, 
21409, 
21229, 
21048, 
20867, 
20686, 
20504, 
20322, 
20139, 
19956, 
19773, 
19589, 
19405, 
19221, 
19036, 
18851, 
18667, 
18481, 
18296, 
18111, 
17926, 
17741, 
17555, 
17370, 
17185, 
16999, 
16814, 
16629, 
16445, 
16260, 
16076, 
15892, 
15708, 
15524, 
15341, 
15158, 
14975, 
14793, 
14612, 
14431, 
14250, 
14070, 
13890, 
13711, 
13533, 
13355, 
13178, 
13001, 
12825, 
12650, 
12476, 
12303, 
12130, 
11958, 
11787, 
11617, 
11448, 
11280, 
11113, 
10946, 
10781, 
10617, 
10454, 
10292, 
10131, 
9971, 
9813, 
9655, 
9499, 
9344, 
9191, 
9038, 
8887, 
8737, 
8589, 
8442, 
8296, 
8152, 
8010, 
7868, 
7728, 
7590, 
7453, 
7318, 
7185, 
7052, 
6922, 
6793, 
6666, 
6541, 
6417, 
6295, 
6174, 
6056, 
5939, 
5824, 
5710, 
5599, 
5489, 
5381, 
5275, 
5171, 
5069, 
4969, 
4870, 
4774, 
4680, 
4587, 
4496, 
4408, 
4321, 
4237, 
4155, 
4074, 
3996, 
3919, 
3845, 
3773, 
3703, 
3635, 
3569, 
3506, 
3444, 
3385, 
3328, 
3273, 
3220, 
3169, 
3121, 
3075, 
3031, 
2989, 
2949, 
2912, 
2877, 
2844, 
2814, 
2785, 
2759, 
2735, 
2714, 
2694, 
2677, 
2662, 
2650, 
2640, 
2632, 
2626, 
2622, 
2621 
]).astype(np.int16)
       # hamming_q15 = (hamming * 32767).astype(np.int16)

       # hamming_q15 = np.round(hamming * (1<<15))
        #hamming_q15=np.clip(hamming_q15,-32768,32767).astype(np.int16)
        #print(hamming_q15)
        return AudioUtil.q15_mult(frames, hamming)


    def rfft_q15(frames, Nft):

        fft = np.fft.rfft(frames, axis=1)

        real = np.real(fft)
        imag = np.imag(fft)

        # approximate CMSIS internal scaling
        real = real / Nft
        imag = imag / Nft

        return real.astype(np.int32), imag.astype(np.int32)

    def normalize_fft(real, imag):

        vmax = np.max(np.abs(np.concatenate([real.flatten(), imag.flatten()])))

        real_n = ((real << 15) // vmax).astype(np.int16)
        imag_n = ((imag << 15) // vmax).astype(np.int16)

        return real_n, imag_n, vmax
    
    def complex_mag_q15(real, imag):

        mag = np.sqrt(real.astype(np.int32)**2 +
                    imag.astype(np.int32)**2)

        return mag.astype(np.int16)
    
    def denormalize(mag, vmax):

        return ((mag.astype(np.int32) * vmax) >> 15).astype(np.int16)


    def specgram(audio, Nft=512, fs2=11025) -> ndarray:

        """
        Compute a Spectrogram.
    
        Args:
        ----
            :param aud: The audio signal as a tuple (signal, sample_rate).
            :param Nft: The number of points of the FFT.
            :param fs2: The sampling frequency.
        
        Returns:
        -------
          stft (numpy array 2D): spectrogram of y : stored in a matrix of size (Nft//2, N/Nft)
    
        """
        
       # y, _ = AudioUtil.resample(audio, fs2)
        
        y = AudioUtil.adc_to_q15(np.array(audio))
        y = y[:Nft*20]
        return y
       # print("y[1] :",y[1])
        #print(y)
        #frames = y.reshape(-1, Nft)
        #print("y",y[0:20])
        #windowed = AudioUtil.apply_window_q15(frames, Nft)

        # Homemade computation of stft
        "Crop the signal such that its length is a multiple of Nft"
       # L = len(y)
        #print(L)
        #y = y[: L - L % Nft]
       # L = len(y)
        
        """
        "Reshape the signal with a piece for each row"
        audiomat = np.reshape(y, (L // Nft, Nft))
        audioham = audiomat * np.hamming(Nft)  # Windowing. Hamming, Hanning, Blackman,..
        z = np.reshape(audioham, -1)  # y windowed by pieces
        "FFT row by row"
        stft = np.fft.fft(audioham, axis=1)
        stft = np.abs(stft[:, : Nft // 2].T)  # Taking only positive frequencies and computing the magnitude
    
        return stft
        """ 
    def get_hz2mel(fs2=11025, Nft=512, Nmel=20) -> ndarray:
        """
        Get the hz2mel conversion matrix.

        :param fs2: The sampling frequency.
        :param Nft: The number of points of the FFT.
        :param Nmel: The number of mel bands.
        """
        mels = librosa.filters.mel(sr=fs2, n_fft=Nft, n_mels=Nmel)
        mels = mels[:, :-1]
        mels = mels / np.max(mels)

        return mels

    def melspectrogram(audio, Nmel=20, Nft=512, fs2=11025) -> ndarray:
        """
        Generate a Melspectrogram.
    
        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param Nmel: The number of mel bands.
        :param Nft: The number of points of the FFT.
        :param fs2: The sampling frequency.
    
        Returns:
        -------
          melspec ((numpy array 2D): compressed spectrogram of x using the Mel's method stored in a matrix of size (Nmel,N/Nft)
    
        """
        ### TO COMPLETE, using the functions resample() and specgram() defined above
    
        "Obtain the Hz2Mel transformation matrix"
        # mels = librosa.filters.mel(sr=fs2, n_fft=Nft, n_mels=Nmel)
        

        mels = librosa.filters.mel(sr=fs2, n_fft=Nft, n_mels=Nmel)
        mels = mels[:, :-1]
        mels = mels / np.max(mels)

        ### Normalize the mels matrix such that its maximum value is one.
        # mels = mels / np.max(mels)
    
        "Getting the spec of the downsampled signal"
        stft = AudioUtil.specgram(audio, Nft=Nft, fs2=fs2)
    
        "Melspectrogram computation"
        ###  Perform the matrix multiplication between the Hz2Mel matrix and stft.
        #print(mels.shape, stft.shape)
        melspec = mels @ stft
        
        return melspec

    def spectro_aug_timefreq_masking(
        spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1
    ) -> ndarray:
        """
        Augment the Spectrogram by masking out some sections of it in both the frequency dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent overfitting and to help the model generalise better. The masked sections are replaced with the mean value.


        :param spec: The spectrogram.
        :param max_mask_pct: The maximum percentage of the spectrogram to mask out.
        :param n_freq_masks: The number of frequency masks to apply.
        :param n_time_masks: The number of time masks to apply.
        """
        Nmel, n_steps = spec.shape
        mask_value = np.mean(spec)
        aug_spec = np.copy(spec)  # avoids modifying spec

        freq_mask_param = max_mask_pct * Nmel
        for _ in range(n_freq_masks):
            height = int(np.round(random.random() * freq_mask_param))
            pos_f = np.random.randint(Nmel - height)
            aug_spec[pos_f : pos_f + height, :] = mask_value

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            width = int(np.round(random.random() * time_mask_param))
            pos_t = np.random.randint(n_steps - width)
            aug_spec[:, pos_t : pos_t + width] = mask_value

        return aug_spec


class Feature_vector_DS:
    """
    Dataset of Feature vectors.
    """

    def __init__(
        self,
        dataset,
        Nft=512,
        nmel=20,
        duration=500,
        normalize=False,
        data_aug=None,
        pca=None,
        step=np.inf,
    ):
        self.dataset = dataset
        self.Nft = Nft
        self.nmel = nmel
        self.duration = duration  # ms
        self.sr = 11025
        self.normalize = normalize
        self.data_aug = data_aug
        self.data_aug_factor = 1
        if isinstance(self.data_aug, list):
            self.data_aug_factor += len(self.data_aug)
        else:
            self.data_aug = [self.data_aug]
        self.ncol = int(
            self.duration * self.sr / (1e3 * self.Nft)
        )  # number of columns in melspectrogram
        self.pca = pca
        self.step = step

    def __len__(self) -> int:
        """
        Number of items in dataset.
        """
        return len(self.dataset) * self.data_aug_factor

    def get_audiosignal(self, cls_index: tuple[str, int]) -> tuple[ndarray, int]:
        """
        Get temporal signal of i'th item in dataset.

        :param cls_index: Class name and index.
        """
        audio_file = self.dataset[cls_index]
        #print(audio_file)
        aud = AudioUtil.open(audio_file)
        aud = AudioUtil.resample(aud, self.sr)
        if self.data_aug is not None:
            if "add_bg" in self.data_aug:
                aud = AudioUtil.add_bg(
                    aud,
                    self.dataset,
                    num_sources=1,
                    max_ms=self.duration,
                    amplitude_limit=0.1,
                )
            if "echo" in self.data_aug:
                aud = AudioUtil.add_echo(aud)
            if "noise" in self.data_aug:
                aud = AudioUtil.add_noise(aud, sigma=0.05)
            if "scaling" in self.data_aug:
                aud = AudioUtil.scaling(aud, scaling_limit=5)

        # aud = AudioUtil.normalize(aud, target_dB=10)
        aud = (aud[0] / np.max(np.abs(aud[0])), aud[1])
        return aud

    def __getitem__(self, cls_index: tuple[str, int]) -> tuple[ndarray, int]:
        """
        Get i'th item in dataset.

        :param cls_index: Class name and index.
        """
        aud = self.get_audiosignal(cls_index)
        sgram = AudioUtil.melspectrogram(aud, Nmel=self.nmel, Nft=self.Nft)
        if self.data_aug is not None:
            if "aug_sgram" in self.data_aug:
                sgram = AudioUtil.spectro_aug_timefreq_masking(
                    sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
                )

        return sgram

    def display(self, cls_index: tuple[str, int], show_features=False):
        """
        Play sound and display i'th item in dataset.

        :param cls_index: Class name and index.
        """
        audio = self.get_audiosignal(cls_index)
        AudioUtil.play(audio)
        plt.figure(figsize=(2 + 2 * len(audio[0]) / self.sr, 3))
        sgram = AudioUtil.melspectrogram(audio, Nmel=self.nmel, Nft=self.Nft)
        plt.imshow(
            sgram,
            cmap="jet",
            origin="lower",
            aspect="auto",
        )
        plt.colorbar()

        if show_features:
            indexes = np.arange(0, len(sgram[0]) - self.ncol, self.step)
            for start in indexes:
                # (x, y) = lower left corner of rectangle
                rect = patches.Rectangle(
                    (start, 0),  # x, y
                    self.ncol,  # width
                    self.nmel - 1,  # height
                    linewidth=2,
                    edgecolor="magenta",
                    facecolor="none",
                    alpha=1,
                )
                plt.gca().add_patch(rect)

        plt.title(audio)
        plt.title(self.dataset.__getname__(cls_index))
        plt.show()

    def get_feature_vectors(self) -> tuple[ndarray, ndarray]:
        """
        Returns all feature vectors and their labels.
        """
        classnames = self.dataset.list_classes()

        y = []
        X = []

        for class_idx, classname in enumerate(classnames):
            for s in range(self.data_aug_factor):
                for idx in range(self.dataset.naudio[classname]):
                    sgram = self[classname, idx]
                    fv = self.treat_spec(sgram)

                    X += list(fv)
                    y += [classname] * len(fv)

        return np.array(X), np.array(y)

    def mod_data_aug(self, data_aug=[]) -> None:
        """
        Modify the data augmentation options.

        :param data_aug: The new data augmentation options.
        """
        self.data_aug = data_aug
        if not isinstance(self.data_aug, list):
            self.data_aug = [self.data_aug]

        self.data_aug_factor = 1 + len(self.data_aug)

    def treat_spec(self, sgram):
        """
        Turns a melspectrogram into a feature vector.

        :param sgram: The melspectrogram to treat.
        """
        indexes = np.arange(0, len(sgram[0]) - self.ncol, self.step, dtype=int)
        sgrams = [sgram[:, i : i + self.ncol] for i in indexes]
        sgrams = np.array(sgrams)

        fv = sgrams.reshape(sgrams.shape[0], -1)  # feature vector

        if self.normalize:
            fv /= np.linalg.norm(fv, axis=1, keepdims=True)

        if self.pca is not None:
            fv = np.array([self.pca.transform(n_components=[i])[0] for i in fv])

        return fv
