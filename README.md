Signal Analyzer for WFDB Data
This Python project analyzes photoplethysmography (PPG) signals stored in WFDB format (.dat and .hea files). It performs signal preprocessing, visualization, and frequency domain analysis using FFT, DFT, and Wavelet transforms.

Features
Load PPG signals from WFDB data files.

Bandpass filter to isolate heart rate frequencies (0.5–4 Hz).

Plot signal in the time domain.

Frequency analysis via FFT (Fast Fourier Transform).

Optional DFT (Discrete Fourier Transform) plot for smaller signals.

Optional Wavelet Transform visualization.

Computes heart rate estimate from dominant frequency peak.

Saves plots as images for later use.

Usage
Run the script with the WFDB record file path (without extension):

```bash
python signal_plotter.py --file "path_to_file/record_name" [--dft] [--wavelet]
```
Arguments
--file (required): Path to the WFDB file (exclude .dat or .hea extension).

--dft (optional): Enable plotting of the Discrete Fourier Transform (only for signals with length ≤ 1024).

--wavelet (optional): Enable plotting of the Wavelet Transform.

Example
```bash
python signal_plotter.py --file "C:\Users\hp\Downloads\pro0\pro1\129083\129083_PPG" --dft --wavelet
```
Technical Details
Uses wfdb to read WFDB-format signals.

Applies a Butterworth bandpass filter (0.5–4 Hz) to clean PPG signals.

Performs FFT with NumPy for fast frequency domain analysis.

Implements manual DFT computation as an optional feature.

Uses PyWavelets for continuous wavelet transform.

Uses Matplotlib for visualization and saves plots as PNG images.

Requirements
Python 3.x

numpy

matplotlib

wfdb

pywt

scipy

Install dependencies via:

```bash
pip install numpy matplotlib wfdb pywt scipy
```
