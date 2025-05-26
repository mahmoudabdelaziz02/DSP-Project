import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import argparse
import matplotlib
matplotlib.use('Agg')  # non-interactive backend

# For wavelet transform, we'll use PyWavelets
import pywt

def generate_sinusoidal(duration=1.0, sampling_rate=1000, frequency=10):
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    x = np.sin(2 * np.pi * frequency * t)
    return t, x

def generate_random(duration=1.0, sampling_rate=1000):
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    x = np.random.randn(len(t))
    return t, x

def dft(x):
    """Compute the Discrete Fourier Transform (DFT) manually."""
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def plot_signal(t, x, title="Signal", filename="signal_plot.png", show_dft=False):
    """
    Plot the time-domain signal along with its FFT (and DFT if selected).
    """
    plt.figure(figsize=(12, 10))
    
    # Plot the time domain signal.
    plt.subplot(4, 1, 1)
    plt.plot(t, x)
    plt.title(f"{title} - Time Domain")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    N = len(x)
    T = t[1] - t[0]
    xf = fftfreq(N, T)[:N // 2]

    # FFT Plot
    yf_fft = fft(x)
    plt.subplot(4, 1, 2)
    plt.plot(xf, 2.0 / N * np.abs(yf_fft[:N // 2]))
    plt.title(f"{title} - Frequency Domain (FFT)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    # DFT Plot (if requested and signal is short enough)
    if show_dft and N <= 1024:  # Avoid long computation times
        yf_dft = dft(x)
        plt.subplot(4, 1, 3)
        plt.plot(xf, 2.0 / N * np.abs(yf_dft[:N // 2]))
        plt.title(f"{title} - Frequency Domain (DFT)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
    elif show_dft:
        print("Signal too long for DFT (slower). Skipping DFT plot.")

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Signal plot saved to {filename}")
    plt.close()

def plot_wavelet(t, x, title="Wavelet Transform", filename="wavelet_plot.png"):
    """
    Compute and plot the Continuous Wavelet Transform (CWT) of the signal.
    A scalogram is produced showing the absolute coefficients of the CWT.
    """
    # Define scales. Adjust this range according to the expected frequency content.
    scales = np.arange(1, 128)
    
    # Using the Morlet wavelet ("morl") which works well for time-frequency analysis.
    coefficients, frequencies = pywt.cwt(x, scales, 'morl', sampling_period=t[1]-t[0])
    
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

def main():
    parser = argparse.ArgumentParser(description="Signal Generator and Analyzer")
    parser.add_argument('--type', choices=['sin', 'rand'], default='sin', 
                        help='Type of signal: sin (default) or rand')
    parser.add_argument('--duration', type=float, default=5.0, 
                        help='Duration of signal in seconds (default: 5)')
    parser.add_argument('--rate', type=float, default=1000, 
                        help='Sampling rate in Hz (default: 1000)')
    parser.add_argument('--freq', type=float, default=10.0, 
                        help='Frequency for sinusoidal signal (default: 10 Hz)')
    parser.add_argument('--dft', action='store_true', 
                        help='Include DFT in frequency analysis (slower, use for short signals)')
    parser.add_argument('--wavelet', action='store_true', 
                        help='Include Continuous Wavelet Transform analysis')

    args = parser.parse_args()

    if args.type == 'sin':
        t, x = generate_sinusoidal(args.duration, args.rate, args.freq)
        plot_signal(t, x, "Sinusoidal Signal", "sinusoidal_signal.png", show_dft=args.dft)
    elif args.type == 'rand':
        t, x = generate_random(args.duration, args.rate)
        plot_signal(t, x, "Random Signal", "random_signal.png", show_dft=args.dft)
    
    # If the wavelet flag is used, plot the wavelet transform.
    if args.wavelet:
        plot_wavelet(t, x, title="Wavelet Transform", filename="wavelet_signal.png")

if __name__ == "__main__":
    main()
#to run the project =>  python signal_plotter.py --type sin --duration 1 --freq 10 --dft --wavelet
# python signal_plotter.py --type rand --duration 2 --rate 500  