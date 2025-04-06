#!/usr/bin/python3
#########################
# Signal Processing Examples with Fourier Analysis
# PHYS 4840 - Math and Computational Methods II
# Week 11 - Fourier Analysis/Signal Processing in Real Life
#########################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import signal
import time

def generate_signal_with_noise(duration=1.0, sample_rate=1000, freq_components=None):
    """
    Generate a test signal with specified frequency components and noise.
    
    Parameters:
        duration (float): Signal duration in seconds
        sample_rate (int): Sampling rate in Hz
        freq_components (list): List of (frequency, amplitude, phase) tuples
        
    Returns:
        tuple: (time_array, clean_signal, noisy_signal)
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # Default frequency components if none specified
    if freq_components is None:
        freq_components = [
            (5, 1.0, 0),      # 5 Hz, amplitude 1.0, phase 0
            (20, 0.5, np.pi/4), # 20 Hz, amplitude 0.5, phase π/4
            (50, 0.25, np.pi/2) # 50 Hz, amplitude 0.25, phase π/2
        ]
    
    # Generate clean signal
    clean_signal = np.zeros_like(t)
    for freq, amp, phase in freq_components:
        clean_signal += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # Add noise
    np.random.seed(42)  # For reproducibility
    noise = 0.2 * np.random.randn(len(t))
    noisy_signal = clean_signal + noise
    
    return t, clean_signal, noisy_signal

def apply_filters(signal, sample_rate=1000):
    """
    Apply various filters to a signal and return the filtered versions.
    
    Parameters:
        signal (array): Input signal
        sample_rate (int): Sampling rate in Hz
        
    Returns:
        dict: Dictionary of filtered signals
    """
    # Design filters
    nyquist = 0.5 * sample_rate
    
    # Low-pass filter (< 15 Hz)
    b_low, a_low = signal.butter(4, 15 / nyquist, 'low')
    
    # High-pass filter (> 30 Hz)
    b_high, a_high = signal.butter(4, 30 / nyquist, 'high')
    
    # Band-pass filter (15-30 Hz)
    b_band, a_band = signal.butter(4, [15 / nyquist, 30 / nyquist], 'band')
    
    # Notch filter (remove 50 Hz, e.g., power line interference)
    b_notch, a_notch = signal.iirnotch(50, 30, sample_rate)
    
    # Apply filters
    filtered_signals = {
        'Original': signal,
        'Low-Pass': signal.filtfilt(b_low, a_low, signal),
        'High-Pass': signal.filtfilt(b_high, a_high, signal),
        'Band-Pass': signal.filtfilt(b_band, a_band, signal),
        'Notch': signal.filtfilt(b_notch, a_notch, signal)
    }
    
    return filtered_signals

def plot_signal_and_spectrum(t, signals_dict, sample_rate=1000, title='Signal Filtering'):
    """
    Plot signals in time domain and their corresponding frequency spectra.
    
    Parameters:
        t (array): Time array
        signals_dict (dict): Dictionary of signals to plot
        sample_rate (int): Sampling rate in Hz
        title (str): Plot title
    """
    n_signals = len(signals_dict)
    
    plt.figure(figsize=(14, 3*n_signals))
    
    # Calculate frequency axis once
    n_samples = len(t)
    freq = np.fft.rfftfreq(n_samples, d=1/sample_rate)
    
    # Plot each signal and its spectrum
    for i, (label, sig) in enumerate(signals_dict.items()):
        # Time domain
        plt.subplot(n_signals, 2, 2*i + 1)
        plt.plot(t, sig)
        plt.grid(True, alpha=0.3)
        plt.title(f'{label} - Time Domain')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Frequency domain
        plt.subplot(n_signals, 2, 2*i + 2)
        spectrum = np.abs(np.fft.rfft(sig))
        plt.plot(freq, spectrum)
        plt.grid(True, alpha=0.3)
        plt.title(f'{label} - Frequency Domain')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, sample_rate / 2)  # Nyquist frequency
        
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()

def demonstrate_filtering():
    """
    Demonstrate basic filtering operations using Fourier transform.
    """
    # Generate a test signal with noise
    sample_rate = 1000  # Hz
    duration = 1.0  # seconds
    
    # Define frequency components: (frequency, amplitude, phase)
    freq_components = [
        (5, 1.0, 0),       # 5 Hz component
        (25, 0.6, np.pi/3),  # 25 Hz component
        (50, 0.4, np.pi/6),  # 50 Hz component (power line interference)
        (75, 0.3, np.pi/4)   # 75 Hz component
    ]
    
    t, clean_signal, noisy_signal = generate_signal_with_noise(
        duration, sample_rate, freq_components
    )
    
    # Apply various filters
    filtered_signals = apply_filters(noisy_signal, sample_rate)
    
    # Plot signals and their spectra
    plot_signal_and_spectrum(t, filtered_signals, sample_rate, 'Signal Filtering Demo')
    
    # Also show the clean signal for comparison
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, clean_signal, label='Clean')
    plt.plot(t, noisy_signal, alpha=0.7, label='Noisy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Clean vs. Noisy Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Show spectra
    plt.subplot(1, 2, 2)
    freq = np.fft.rfftfreq(len(t), d=1/sample_rate)
    clean_spectrum = np.abs(np.fft.rfft(clean_signal))
    noisy_spectrum = np.abs(np.fft.rfft(noisy_signal))
    plt.plot(freq, clean_spectrum, label='Clean')
    plt.plot(freq, noisy_spectrum, alpha=0.7, label='Noisy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Frequency Spectra')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, sample_rate/2)
    
    plt.tight_layout()
    plt.savefig('clean_vs_noisy.png')
    plt.show()

def demonstrate_windowing():
    """
    Demonstrate the effect of windowing on spectral leakage.
    """
    # Generate a simple sinusoid
    sample_rate = 1000  # Hz
    duration = 1.0  # seconds
    N = int(duration * sample_rate)
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Create a sine wave with a frequency that doesn't align with FFT bins
    freq = 12.5  # Hz (non-integer multiple of the fundamental frequency)
    x = np.sin(2 * np.pi * freq * t)
    
    # Apply different window functions
    windows = {
        'Rectangular': np.ones(N),  # No windowing
        'Hamming': np.hamming(N),
        'Hann': np.hanning(N),
        'Blackman': np.blackman(N),
        'Kaiser': np.kaiser(N, beta=8.6)  # Comparable to Blackman
    }
    
    # Apply windows and compute FFT
    windowed_signals = {}
    spectra = {}
    
    for name, window in windows.items():
        windowed_signals[name] = x * window
        spectra[name] = np.abs(np.fft.rfft(windowed_signals[name]))
    
    # Plot the windows
    plt.figure(figsize=(12, 6))
    for name, window in windows.items():
        plt.plot(t[:100], window[:100], label=name)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Window Functions (first 100 samples)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('window_functions.png')
    plt.show()
    
    # Plot the spectra on linear scale
    plt.figure(figsize=(12, 6))
    freq_axis = np.fft.rfftfreq(N, d=1/sample_rate)
    
    for name, spectrum in spectra.items():
        plt.plot(freq_axis, spectrum, label=name)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title('Effect of Windowing on Spectral Leakage (Linear Scale)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 50)  # Focus on the region around our signal
    plt.savefig('spectral_leakage_linear.png')
    plt.show()
    
    # Plot the spectra on logarithmic scale to better see the sidelobes
    plt.figure(figsize=(12, 6))
    
    for name, spectrum in spectra.items():
        plt.semilogy(freq_axis, spectrum, label=name)
    
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.title('Effect of Windowing on Spectral Leakage (Log Scale)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (log scale)')
    plt.xlim(0, 50)
    plt.ylim(1e-4, 1e3)
    plt.savefig('spectral_leakage_log.png')
    plt.show()
    
    # Print some metrics about the windows
    print("Window Function Characteristics:")
    print("-------------------------------")
    print(f"{'Window':<12}{'Main Lobe Width (bins)':<25}{'Highest Sidelobe (dB)':<25}")
    
    for name, spectrum in spectra.items():
        # Find the main lobe width at -3 dB
        main_lobe_width = 0
        max_val = np.max(spectrum)
        threshold = max_val / np.sqrt(2)  # -3 dB
        
        # Find where the spectrum crosses the threshold on each side of the peak
        peak_idx = np.argmax(spectrum)
        left_idx = peak_idx
        while left_idx > 0 and spectrum[left_idx] > threshold:
            left_idx -= 1
        
        right_idx = peak_idx
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > threshold:
            right_idx += 1
        
        main_lobe_width = right_idx - left_idx
        
        # Find the highest sidelobe
        # Zero out the main lobe region
        temp_spectrum = spectrum.copy()
        temp_spectrum[max(0, left_idx-1):min(len(spectrum), right_idx+2)] = 0
        highest_sidelobe = np.max(temp_spectrum)
        highest_sidelobe_db = 20 * np.log10(highest_sidelobe / max_val)
        
        print(f"{name:<12}{main_lobe_width:<25.2f}{highest_sidelobe_db:<25.2f}")

def demonstrate_spectrogram():
    """
    Demonstrate spectrogram analysis for time-varying frequency content.
    """
    # Create a signal with time-varying frequency content
    sample_rate = 1000  # Hz
    duration = 5.0  # seconds
    N = int(duration * sample_rate)
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Create a chirp signal (frequency increases with time)
    f0 = 5  # start frequency (Hz)
    f1 = 100  # end frequency (Hz)
    chirp_signal = signal.chirp(t, f0=f0, f1=f1, t1=duration, method='linear')
    
    # Add some additional frequency components at specific times
    # 20 Hz burst in the first second
    burst1 = np.zeros_like(t)
    burst1[t < 1.0] = 0.5 * np.sin(2 * np.pi * 20 * t[t < 1.0])
    
    # 50 Hz burst in the middle
    burst2 = np.zeros_like(t)
    mask = (t >= 2.0) & (t < 3.0)
    burst2[mask] = 0.7 * np.sin(2 * np.pi * 50 * t[mask])
    
    # 80 Hz burst at the end
    burst3 = np.zeros_like(t)
    burst3[t >= 4.0] = 0.6 * np.sin(2 * np.pi * 80 * t[t >= 4.0])
    
    # Combine all components
    combined_signal = chirp_signal + burst1 + burst2 + burst3
    
    # Add some noise
    np.random.seed(42)
    noisy_signal = combined_signal + 0.1 * np.random.randn(N)
    
    # Plot the time domain signal
    plt.figure(figsize=(12, 4))
    plt.plot(t, noisy_signal)
    plt.grid(True, alpha=0.3)
    plt.title('Time-Varying Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('time_varying_signal.png')
    plt.show()
    
    # Compute and plot a spectrogram with different window sizes
    window_sizes = [128, 256, 512, 1024]
    plt.figure(figsize=(15, 10))
    
    for i, nperseg in enumerate(window_sizes):
        plt.subplot(2, 2, i + 1)
        f, t_spec, Sxx = signal.spectrogram(
            noisy_signal, fs=sample_rate, nperseg=nperseg, noverlap=nperseg//2, 
            window=('tukey', 0.25)
        )
        plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        plt.title(f'Spectrogram (Window Size: {nperseg})')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.ylim(0, 120)  # Focus on relevant frequencies
    
    plt.tight_layout()
    plt.savefig('spectrograms.png')
    plt.show()
    
    # Compare different window functions in the spectrogram
    window_functions = ['boxcar', 'hamming', 'blackman', ('tukey', 0.25)]
    plt.figure(figsize=(15, 10))
    
    for i, window in enumerate(window_functions):
        plt.subplot(2, 2, i + 1)
        f, t_spec, Sxx = signal.spectrogram(
            noisy_signal, fs=sample_rate, nperseg=512, noverlap=256, 
            window=window
        )
        plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        plt.colorbar(label='Power/Frequency (dB/Hz)')
        window_name = window if isinstance(window, str) else f"{window[0]}({window[1]})"
        plt.title(f'Spectrogram (Window: {window_name})')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.ylim(0, 120)
    
    plt.tight_layout()
    plt.savefig('spectrogram_windows.png')
    plt.show()

def demonstrate_fft_image_processing():
    """
    Demonstrate Fourier Transform applications in 2D image processing.
    """
    # Create a simple test image
    size = 256
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Create a pattern with different frequency components
    # Low frequency pattern
    low_freq = np.sin(2 * np.pi * 2 * X) * np.sin(2 * np.pi * 2 * Y)
    
    # Medium frequency pattern
    med_freq = 0.5 * np.sin(2 * np.pi * 8 * X) * np.sin(2 * np.pi * 8 * Y)
    
    # High frequency pattern (noise-like)
    np.random.seed(42)
    high_freq = 0.2 * np.random.randn(size, size)
    
    # Combine patterns
    image = low_freq + med_freq + high_freq
    
    # Compute 2D FFT
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)  # Shift zero frequency to center
    
    # Create filters in the frequency domain
    center = size // 2
    Y_freq, X_freq = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(size)),
                                np.fft.fftshift(np.fft.fftfreq(size)))
    R_freq = np.sqrt(X_freq**2 + Y_freq**2)
    
    # Low-pass filter (keep low frequencies, remove high)
    lowpass = R_freq <= 0.1
    
    # High-pass filter (keep high frequencies, remove low)
    highpass = R_freq >= 0.1
    
    # Band-pass filter (keep medium frequencies)
    bandpass = (R_freq >= 0.05) & (R_freq <= 0.15)
    
    # Apply filters in frequency domain
    filtered_fft_low = fft_shifted * lowpass
    filtered_fft_high = fft_shifted * highpass
    filtered_fft_band = fft_shifted * bandpass
    
    # Convert back to spatial domain
    filtered_image_low = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft_low)))
    filtered_image_high = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft_high)))
    filtered_image_band = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft_band)))
    
    # Plot the original and filtered images
    plt.figure(figsize=(15, 12))
    
    # Original image
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Magnitude spectrum (log scale for better visualization)
    plt.subplot(3, 3, 2)
    magnitude_spectrum = np.log1p(np.abs(fft_shifted))
    plt.imshow(magnitude_spectrum, cmap='viridis')
    plt.title('Magnitude Spectrum (log scale)')
    plt.axis('off')
    
    # Phase spectrum
    plt.subplot(3, 3, 3)
    phase_spectrum = np.angle(fft_shifted)
    plt.imshow(phase_spectrum, cmap='hsv')
    plt.title('Phase Spectrum')
    plt.axis('off')
    
    # Low-pass filter visualization
    plt.subplot(3, 3, 4)
    plt.imshow(lowpass, cmap='gray')
    plt.title('Low-Pass Filter')
    plt.axis('off')
    
    # High-pass filter visualization
    plt.subplot(3, 3, 5)
    plt.imshow(highpass, cmap='gray')
    plt.title('High-Pass Filter')
    plt.axis('off')
    
    # Band-pass filter visualization
    plt.subplot(3, 3, 6)
    plt.imshow(bandpass, cmap='gray')
    plt.title('Band-Pass Filter')
    plt.axis('off')
    
    # Low-pass filtered image
    plt.subplot(3, 3, 7)
    plt.imshow(filtered_image_low, cmap='gray')
    plt.title('Low-Pass Filtered Image')
    plt.axis('off')
    
    # High-pass filtered image
    plt.subplot(3, 3, 8)
    plt.imshow(filtered_image_high, cmap='gray')
    plt.title('High-Pass Filtered Image')
    plt.axis('off')
    
    # Band-pass filtered image
    plt.subplot(3, 3, 9)
    plt.imshow(filtered_image_band, cmap='gray')
    plt.title('Band-Pass Filtered Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('image_filtering.png')
    plt.show()

def demonstrate_periodogram():
    """
    Demonstrate power spectral density estimation with periodogram and Welch's method.
    """
    # Create a signal with multiple frequency components and noise
    sample_rate = 1000  # Hz
    duration = 5.0  # seconds
    N = int(duration * sample_rate)
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Signal components
    freq_components = [10, 25, 50, 100]
    amplitudes = [1.0, 0.6, 0.4, 0.3]
    phases = [0, np.pi/4, np.pi/3, np.pi/2]
    
    x = np.zeros_like(t)
    for f, a, p in zip(freq_components, amplitudes, phases):
        x += a * np.sin(2 * np.pi * f * t + p)
    
    # Add noise
    np.random.seed(42)
    noise_level = 0.5
    x_noisy = x + noise_level * np.random.randn(N)
    
    # Compute periodogram
    # Standard periodogram (raw FFT)
    f_pgram, psd_pgram = signal.periodogram(x_noisy, fs=sample_rate, window='boxcar', 
                                           scaling='density')
    
    # Welch's method with different window sizes
    segment_sizes = [256, 512, 1024, 2048]
    psd_welch = {}
    
    for nperseg in segment_sizes:
        f_welch, psd = signal.welch(x_noisy, fs=sample_rate, window='hann', 
                                    nperseg=nperseg, scaling='density')
        psd_welch[nperseg] = psd
    
    # Plot time domain signal
    plt.figure(figsize=(12, 4))
    plt.plot(t, x_noisy)
    plt.grid(True, alpha=0.3)
    plt.title('Noisy Signal with Multiple Frequency Components')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('psd_time_signal.png')
    plt.show()
    
    # Plot PSDs
    plt.figure(figsize=(12, 8))
    
    # Raw periodogram
    plt.subplot(3, 1, 1)
    plt.semilogy(f_pgram, psd_pgram, label='Standard Periodogram')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.title('Standard Periodogram (Noisy)')
    plt.ylabel('PSD (V²/Hz)')
    plt.xlim(0, 120)
    
    # Welch's method with different segment sizes
    plt.subplot(3, 1, 2)
    for nperseg, psd in psd_welch.items():
        plt.semilogy(f_welch, psd, label=f'Welch (nperseg={nperseg})')
    
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.title('Welch\'s Method (Different Window Sizes)')
    plt.ylabel('PSD (V²/Hz)')
    plt.xlim(0, 120)
    
    # Ground truth spectrum (clean signal)
    f_truth, psd_truth = signal.periodogram(x, fs=sample_rate, window='boxcar')
    plt.subplot(3, 1, 3)
    plt.semilogy(f_truth, psd_truth, 'k-', label='Ground Truth (Clean Signal)')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.title('Ground Truth Spectrum (Clean Signal)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V²/Hz)')
    plt.xlim(0, 120)
    
    plt.tight_layout()
    plt.savefig('psd_comparison.png')
    plt.show()
    
    # Compare SNR improvements with Welch's method
    snr_pgram = estimate_snr(f_pgram, psd_pgram, freq_components)
    print(f"SNR for Standard Periodogram: {snr_pgram:.2f} dB")
    
    for nperseg, psd in psd_welch.items():
        snr_welch = estimate_snr(f_welch, psd, freq_components)
        print(f"SNR for Welch's Method (nperseg={nperseg}): {snr_welch:.2f} dB")
    
    snr_truth = estimate_snr(f_truth, psd_truth, freq_components)
    print(f"SNR for Ground Truth: {snr_truth:.2f} dB")

def estimate_snr(f, psd, signal_freqs, bandwidth=2.0):
    """
    Estimate SNR from PSD by comparing signal power to noise floor.
    
    Parameters:
        f (array): Frequency axis
        psd (array): Power spectral density
        signal_freqs (list): List of signal frequencies
        bandwidth (float): Bandwidth around each signal frequency
        
    Returns:
        float: SNR in dB
    """
    # Identify signal regions
    signal_mask = np.zeros_like(f, dtype=bool)
    for freq in signal_freqs:
        signal_mask |= (f >= freq - bandwidth/2) & (f <= freq + bandwidth/2)
    
    # Estimate signal power (in signal regions)
    signal_power = np.sum(psd[signal_mask])
    
    # Estimate noise power (outside signal regions)
    noise_mask = ~signal_mask
    noise_power = np.sum(psd[noise_mask]) * (np.sum(signal_mask) / np.sum(noise_mask))
    
    # Compute SNR
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    return snr

def demonstrate_astronomy_applications():
    """
    Demonstrate Fourier analysis applications in astronomy (periodicity detection).
    """
    # Simulate astronomical time series data (e.g., variable star light curve)
    np.random.seed(42)
    days = 100  # Observation period in days
    t = np.linspace(0, days, 1000)  # Time array (days)
    
    # True periods in days
    periods = [3.5, 7.2, 12.8]
    amplitudes = [1.0, 0.8, 0.5]
    
    # Generate signal with periodic components
    signal = np.zeros_like(t)
    for period, amp in zip(periods, amplitudes):
        freq = 1.0 / period  # Frequency in cycles per day
        signal += amp * np.sin(2 * np.pi * freq * t)
    
    # Add some red noise (common in astronomical time series)
    # Generate white noise first
    white_noise = 0.5 * np.random.randn(len(t))
    
    # Apply a low-pass filter to create red noise
    b, a = signal.butter(2, 0.1, 'low')
    red_noise = signal.filtfilt(b, a, white_noise)
    red_noise *= 0.5 / np.std(red_noise)  # Scale noise
    
    # Add the noise to the signal
    noisy_signal = signal + red_noise
    
    # Create an unevenly sampled time series (common in astronomy)
    # Randomly select 200 points from the full time series
    np.random.seed(123)
    indices = np.sort(np.random.choice(len(t), 200, replace=False))
    t_uneven = t[indices]
    signal_uneven = noisy_signal[indices]
    
    # Add some measurement errors (typical in astronomy)
    errors = 0.1 * np.random.randn(len(t_uneven))
    signal_uneven += errors
    
    # Plot the simulated light curve
    plt.figure(figsize=(12, 4))
    plt.errorbar(t_uneven, signal_uneven, yerr=0.1, fmt='o', alpha=0.7, markersize=3)
    plt.grid(True, alpha=0.3)
    plt.title('Simulated Astronomical Light Curve (Unevenly Sampled)')
    plt.xlabel('Time (days)')
    plt.ylabel('Brightness')
    plt.savefig('astro_light_curve.png')
    plt.show()
    
    # Perform Lomb-Scargle periodogram analysis (suitable for unevenly sampled data)
    frequency = np.linspace(0.01, 1.0, 1000)  # Frequency range to search (cycles/day)
    power_ls = signal.lombscargle(t_uneven, signal_uneven, 2 * np.pi * frequency)
    
    # Convert to periods
    period = 1.0 / frequency
    
    # Plot the periodogram
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(frequency, power_ls)
    plt.grid(True, alpha=0.3)
    plt.title('Lomb-Scargle Periodogram (Frequency Domain)')
    plt.xlabel('Frequency (cycles/day)')
    plt.ylabel('Power')
    
    # Add vertical lines at the true frequencies
    for p in periods:
        plt.axvline(1.0/p, color='r', linestyle='--', alpha=0.7)
    
    # Plot in period domain (more common in astronomy)
    plt.subplot(2, 1, 2)
    mask = (period >= 1.0) & (period <= 20.0)  # Focus on 1-20 day periods
    plt.plot(period[mask], power_ls[mask])
    plt.grid(True, alpha=0.3)
    plt.title('Lomb-Scargle Periodogram (Period Domain)')
    plt.xlabel('Period (days)')
    plt.ylabel('Power')
    
    # Add vertical lines at the true periods
    for p in periods:
        plt.axvline(p, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('lomb_scargle_periodogram.png')
    plt.show()
    
    # Find peaks in the periodogram
    peak_indices = signal.find_peaks(power_ls, height=0.1)[0]
    peak_periods = period[peak_indices]
    peak_powers = power_ls[peak_indices]
    
    # Sort by power
    sorted_indices = np.argsort(peak_powers)[::-1]
    peak_periods = peak_periods[sorted_indices]
    peak_powers = peak_powers[sorted_indices]
    
    # Print the top periods
    print("Top detected periods:")
    for i, (p, power) in enumerate(zip(peak_periods[:5], peak_powers[:5])):
        print(f"{i+1}. Period: {p:.2f} days, Power: {power:.4f}")
    
    # Compare with true periods
    print("\nTrue periods:", periods)
    
    # Calculate phase-folded light curve for the strongest period
    best_period = peak_periods[0]
    phase = (t_uneven / best_period) % 1.0
    
    # Sort by phase for plotting
    sort_indices = np.argsort(phase)
    phase_sorted = phase[sort_indices]
    signal_sorted = signal_uneven[sort_indices]
    
    # Plot phase-folded light curve
    plt.figure(figsize=(10, 5))
    plt.scatter(phase_sorted, signal_sorted, alpha=0.7, s=10)
    
    # Plot the data points again shifted by 1 period for continuity
    plt.scatter(phase_sorted + 1, signal_sorted, alpha=0.7, s=10)
    
    # Fit a sinusoid to the folded data
    def sinusoid(x, amplitude, phase_shift, offset):
        return amplitude * np.sin(2 * np.pi * x + phase_shift) + offset
    
    from scipy.optimize import curve_fit
    
    # Initial parameter guess
    p0 = [1.0, 0, 0]
    
    try:
        # Fit the sinusoid
        params, _ = curve_fit(sinusoid, phase, signal_uneven, p0=p0)
        
        # Plot the fitted curve
        x_fit = np.linspace(0, 2, 1000)
        y_fit = sinusoid(x_fit, *params)
        plt.plot(x_fit, y_fit, 'r-', lw=2)
        
        print(f"\nFitted sinusoid parameters for period {best_period:.2f} days:")
        print(f"Amplitude: {params[0]:.4f}")
        print(f"Phase shift: {params[1]:.4f} radians")
        print(f"Offset: {params[2]:.4f}")
        
    except RuntimeError:
        print("Curve fitting failed")
    
    plt.grid(True, alpha=0.3)
    plt.title(f'Phase-Folded Light Curve (Period: {best_period:.2f} days)')
    plt.xlabel('Phase')
    plt.ylabel('Brightness')
    plt.xlim(0, 2)
    plt.savefig('phase_folded_curve.png')
    plt.show()

# Example usage
if __name__ == "__main__":
    print("Demonstrating Signal Processing Applications with Fourier Analysis")
    
    # Basic filtering demo
    print("\nDemonstrating basic filtering...")
    demonstrate_filtering()
    
    # Windowing demo
    print("\nDemonstrating windowing effects...")
    demonstrate_windowing()
    
    # Spectrogram demo
    print("\nDemonstrating spectrograms for time-frequency analysis...")
    demonstrate_spectrogram()
    
    # Image processing demo
    print("\nDemonstrating 2D FFT for image processing...")
    demonstrate_fft_image_processing()
    
    # Power spectral density estimation
    print("\nDemonstrating power spectral density estimation...")
    demonstrate_periodogram()
    
    # Astronomy applications
    print("\nDemonstrating astronomy applications...")
    demonstrate_astronomy_applications()
    