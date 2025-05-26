import argparse
import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
# For wavelet transform, we'll use PyWavelets
import pywt
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, lowcut=0.5, highcut=4.0, order=10):
    """Bandpass filter the PPG signal to isolate heart rate range."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def load_wfdb_signal(filepath):
    """Load signal data from a WFDB (.dat/.hea) file."""
    record = wfdb.rdrecord(filepath)
    signal = record.p_signal[:, 0]
    sampling_rate = record.fs
    ft_signal = bandpass_filter(signal, sampling_rate)
    t = np.arange(len(ft_signal)) / sampling_rate
    return t, ft_signal, sampling_rate

def dft(x):
    """Compute the Discrete Fourier Transform (DFT) manually."""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def plot_signal_fft(t, x, title="Signal", filename="signal_plot.png"):
    plt.figure(figsize=(12, 10))
    
    # Time domain plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title(f"{title} - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    N = len(x)
    T = t[1] - t[0]
    xf = np.fft.fftfreq(N, T)[:N // 2]

    # FFT plot
    yf_fft = np.fft.fft(x)
    magnitude = 2.0 / N * np.abs(yf_fft[:N // 2])

    plt.subplot(2, 1, 2)
    plt.plot(xf, magnitude, label="FFT Magnitude")
    
    # Highlight the peak frequency
    peak_index = np.argmax(magnitude)  # Index of the peak frequency
    peak_frequency = xf[peak_index]  # Peak frequency
    peak_magnitude = magnitude[peak_index]  # Peak magnitude
    
    plt.plot(peak_frequency, peak_magnitude, 'ro')  # Red dot at peak frequency
    plt.text(peak_frequency, peak_magnitude, f'{peak_frequency:.2f} Hz', color='red', 
            fontsize=12, ha='right', va='bottom')  # Annotate the peak frequency

    plt.title(f"{title} - Frequency Domain (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Signal plot saved to {filename}")
    plt.close()

    # Calculate heart rate
    heart_rate = peak_frequency * 60  # Convert frequency to heart rate
    print(f"The highest frequency is: {peak_frequency:.2f} Hz")
    print(f"Heart rate: {heart_rate:.2f} beats per minute")


def plot_wavelet(t, x, title="Wavelet Transform", filename="wavelet_plot.png"):
    scales = np.arange(1, 128)
    coefficients, _ = pywt.cwt(x, scales, 'morl', sampling_period=t[1] - t[0])

    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(coefficients), extent=[t[0], t[-1], scales[-1], scales[0]],
            aspect='auto', cmap='jet')
    plt.colorbar(label='Magnitude')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Scale")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Wavelet transform plot saved to {filename}")
    plt.close()

def plot_dft(t, x, title="DFT Transform", filename="dft_plot.png"):
    plt.figure(figsize=(12, 10))
    # Time domain plot
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.title(f"{title} - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    yf_dft = dft(x)
    N = len(x)
    T = t[1] - t[0]
    xf = np.fft.fftfreq(N, T)[:N // 2]

    plt.subplot(2, 1, 2)
    plt.plot(xf, 2.0 / N * np.abs(yf_dft[:N // 2]))
    plt.title(f"{title} - Frequency Domain (DFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.savefig(filename)
    print(f"DFT transform plot saved to {filename}")
    plt.close()
    

def main():
    parser = argparse.ArgumentParser(description="Signal Analyzer for WFDB data")
    parser.add_argument('--file', type=str, required=True, help='Path to the .dat or .hea file')
    parser.add_argument('--dft', action='store_true', help='Include DFT')
    parser.add_argument('--wavelet', action='store_true', help='Include Wavelet Transform')

    args = parser.parse_args()

    if not os.path.exists(args.file + ".hea"):
        print(f"File not found: {args.file}")
        return

    # Load the signal data
    t, x, fs = load_wfdb_signal(args.file)
    base_name = os.path.splitext(os.path.basename(args.file))[0]

    plot_signal_fft(t, x, title=f"{base_name} Signal", filename=f"{base_name}_plot.png")

    # Optionally plot the wavelet transform
    if args.wavelet:
        plot_wavelet(t, x, title="Wavelet Transform", filename=f"{base_name}_wavelet.png")
    # Optionally plot the DFT transform
    if args.dft:
        if len(x) <= 1024:
            plot_dft(t, x, title="DFT Transform", filename=f"{base_name}_dft.png")
        else:
            print("Signal too long for DFT (N > 1024). Skipping DFT.")

if __name__ == "__main__":
    main()

# python signal_plotter.py --file "C:\Users\hp\Downloads\pro0\pro1\129083\129083_PPG" --dft --wavelet => to run 