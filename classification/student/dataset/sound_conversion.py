# # Source - https://stackoverflow.com/a/44800492
# # Posted by Tom Wyllie, modified by community. See post 'Timeline' for change history
# # Retrieved 2026-02-20, License - CC BY-SA 4.0

# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.io import wavfile

# sample_rate, samples = wavfile.read('8._Bang_Bang.wav')
# frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

# plt.pcolormesh(times, frequencies, spectrogram)
# plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np

sample_rate, samples = wavfile.read('chainsaw_37.wav')
fs_down = 11025  # Desired downsampled frequency
fs      = sample_rate  # Original sampling frequency
M = int(fs / fs_down)  # Downsampling factor

specgram = np.fft.fft(samples)
specgram = np.fft.fftshift(specgram)

"Low-pass filtering before downsampling"
N = 100  # number of taps
taps = signal.firwin(numtaps=N, cutoff=fs_down / 2, window="hamming", fs=fs)
x_filt = np.convolve(samples, taps, mode="full")
x_filt = x_filt[::M]


freqs    = np.fft.fftfreq(len(x_filt), 1/fs_down)
freqs    = np.fft.fftshift(freqs)

plt.plot(freqs, np.abs(x_filt))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Magnitude Spectrum')
plt.xlim(-sample_rate/2, sample_rate/2)  # Limit x-axis to positive frequencies
plt.grid()
plt.show()