from scipy import fftpack
import numpy as np
import pandas as pd

def calc_fft(signal):
    return fftpack.fft(signal)

def plot_power(sig_fft, signal, time_step):
    power = np.abs(sig_fft)**2

    # The corresponding frequencies
    sample_freq = fftpack.fftfreq(signal.size, d=time_step)

    # Plot the FFT power
    plt_data = pd.DataFrame({"power": power}, index=sample_freq)
    return (
        plt_data[plt_data.index > 0]
          .plot(logx=True,
               xlabel='Frequency (Hz)')
    )

def filter_freq(signal, cutoff, time_step):
    signal_fft = calc_fft(signal)
    sample_freq = fftpack.fftfreq(signal.size, d=time_step)
    high_freq_fft = signal_fft.copy()
    high_freq_fft[np.abs(sample_freq) > cutoff] = 0
    return fftpack.ifft(high_freq_fft)
    