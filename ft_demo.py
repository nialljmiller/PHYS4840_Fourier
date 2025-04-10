#!/usr/bin/env python3
"""
Simplified Fourier Transform Demonstration
-----------------------------------------
This script demonstrates basic Fourier Transform applications:
1. Analysis of a sinusoid with noise
2. Simple audio processing with note identification

PHYS 4840 - Mathematical and Computational Methods II
"""

import numpy as np
import matplotlib.pyplot as plt
import fourier_transform as ft
from scipy.io import wavfile


def create_frequency_grid(signal_length, sample_rate):
    """
    Create a frequency grid for the given signal length and sample rate.
    """
    return np.linspace(0, sample_rate/2, signal_length//2)


def find_peaks(spectrum, frequencies, threshold=0.1, min_distance=50):
    """
    Find peaks in the frequency spectrum with better isolation.
    Simple but robust implementation that finds truly significant peaks.
    """
    # Find local maxima in isolated regions
    peak_indices = []
    max_val = np.max(spectrum)
    min_val = threshold * max_val
    
    # First find candidate peaks
    candidates = []
    for i in range(1, len(spectrum)-1):
        # Must be a local maximum and above threshold
        if (spectrum[i] > spectrum[i-1] and 
            spectrum[i] > spectrum[i+1] and 
            spectrum[i] > min_val):
            candidates.append(i)
    
    # Sort candidates by magnitude
    candidates.sort(key=lambda idx: spectrum[idx], reverse=True)
    
    # Take peaks in order of magnitude, ensuring minimum distance between them
    for candidate in candidates:
        # Check if this peak is far enough from all existing peaks
        isolated = True
        for peak in peak_indices:
            dist = abs(candidate - peak)
            if dist < min_distance:
                isolated = False
                break
        
        if isolated:
            peak_indices.append(candidate)
    
    # Sort peak indices by frequency
    peak_indices.sort()
    
    # Extract the frequencies and magnitudes
    peak_freqs = frequencies[peak_indices]
    peak_mags = spectrum[peak_indices]
    
    return peak_freqs, peak_mags


def identify_note(frequency):
    """
    Identify musical note from frequency.
    """
    # Define A4 = 440 Hz
    A4 = 440.0
    
    # Define note names
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Calculate semitones from A4
    if frequency <= 0:
        return "Unknown"
    
    semitones = 12 * np.log2(frequency / A4)
    semitones_rounded = round(semitones)
    
    # Calculate octave and note index
    octave = 4 + (semitones_rounded + 9) // 12
    note_idx = (semitones_rounded + 9) % 12
    
    # Calculate cents (how far from the exact note)
    cents = 100 * (semitones - semitones_rounded)
    
    return f"{note_names[note_idx]}{octave} ({cents:+.0f} cents)"


def demo_noisy_sinusoid():

    # Create a signal
    fs = 1000  # Sampling frequency (Hz)
    duration = 1.0  # Signal duration (seconds)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Signal with two frequencies
    f1, f2 = 50, 120  # Frequencies (Hz)
    signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    
    # Add some noise
    np.random.seed(42)
    noisy_signal = signal + 0.2 * np.random.randn(len(t))
    
    # Compute DFT using our module
    X = ft.dft(noisy_signal)
    
    # Get only the first half of the spectrum (positive frequencies)
    half_n = len(X) // 2
    magnitudes = np.abs(X[:half_n]) / len(noisy_signal)
    
    # Create frequency grid
    freqs = np.linspace(0, fs/2, half_n)
    
    # Find peaks in the spectrum
    peak_freqs, peak_mags = find_peaks(magnitudes, freqs)
    
    # Plot time domain signal and frequency spectrum
    plt.figure(figsize=(10, 8))
    
    # Time domain
    plt.subplot(2, 1, 1)
    plt.plot(t, noisy_signal)
    plt.grid(True, alpha=0.3)
    plt.title('Noisy Sinusoid (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Frequency domain
    plt.subplot(2, 1, 2)
    plt.plot(freqs, magnitudes)
    plt.grid(True, alpha=0.3)
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, fs/2)  # Nyquist frequency
    
    # Mark the true frequencies
    plt.axvline(f1, color='r', linestyle='--', label=f'{f1} Hz')
    plt.axvline(f2, color='g', linestyle='--', label=f'{f2} Hz')
    
    # Mark detected peaks
    for i, (freq, mag) in enumerate(zip(peak_freqs, peak_mags)):
        plt.plot(freq, mag, 'ro', markersize=8)
        plt.text(freq, mag*1.1, f"{freq:.1f} Hz", ha='center')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('sinusoid_analysis.png')
    plt.show()


def demo_audio_processing():
    
    # Load the audio file
    fs, audio_data = wavfile.read('audio.wav')
    
    # Convert to mono if stereo and normalize
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Process the ENTIRE audio
    audio_segment = audio_data
    n_samples = len(audio_segment)
    duration = n_samples / fs
    
    # Create time axis for the full audio
    t = np.linspace(0, duration, n_samples)
    
    # Compute FFT for all samples
    X = np.fft.fft(audio_segment)
    
    # Get only the first half of the spectrum (positive frequencies)
    half_n = len(X) // 2
    magnitudes = np.abs(X[:half_n]) / len(audio_segment)
    
    # Create frequency grid limited to human hearing range (20 Hz - 20 kHz)
    # But still maintain the correct number of points
    human_hearing_max = min(20000, fs/2)  # Either 20 kHz or Nyquist frequency, whichever is lower
    freqs = np.linspace(0, human_hearing_max, half_n)
    
    # Find peaks in the spectrum
    peak_freqs, peak_mags = find_peaks(magnitudes, freqs, threshold=0.2)
    
    # Identify musical notes from peaks
    notes = []
    for freq in peak_freqs:
        notes.append(identify_note(freq))
    
    # Plot the audio and its spectrum
    plt.figure(figsize=(12, 8))
    
    # Time domain - downsample for display only
    plt.subplot(2, 1, 1)
    max_plot_points = 10000
    if len(t) > max_plot_points:
        plot_step = len(t) // max_plot_points
        plt.plot(t[::plot_step], audio_segment[::plot_step])
    else:
        plt.plot(t, audio_segment)
    plt.grid(True, alpha=0.3)
    plt.title('Complete Audio Signal (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Frequency domain - focus on 0-1000 Hz range
    plt.subplot(2, 1, 2)
    # Find the index corresponding to 1000 Hz for plotting
    idx_1000hz = int(1000 * len(freqs) / (fs/2))
    plt.plot(freqs[:idx_1000hz], magnitudes[:idx_1000hz])
    plt.grid(True, alpha=0.3)
    plt.title('Frequency Spectrum with Identified Notes')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    
    # Mark detected peaks and notes
    for i, (freq, mag, note) in enumerate(zip(peak_freqs, peak_mags, notes)):
        if freq <= 1000:  # Only annotate peaks below 1000 Hz
            plt.plot(freq, mag, 'ro', markersize=8)
            plt.text(freq, mag*1.1, f"{note}", ha='center')
    
    plt.tight_layout()
    plt.savefig('audio_analysis.png')
    plt.show()
    
    # Define a simple bandpass filter around the main frequency component
    if len(peak_freqs) > 0:
        main_freq = peak_freqs[0]
        filter_width = 50  # Hz
        
        print(f"Applying bandpass filter around {main_freq:.1f} Hz")
        
        # Create a filtered spectrum - by copying the original spectrum
        X_filtered = np.zeros_like(X, dtype=complex)
        
        # Apply bandpass filter in frequency domain
        for i in range(half_n):
            freq = freqs[i]
            if abs(freq - main_freq) < filter_width:
                X_filtered[i] = X[i]
                # Also set the corresponding negative frequency
                if i > 0:  # Skip DC component
                    X_filtered[len(X)-i] = X[len(X)-i]
        


        filtered_signal = np.fft.ifft(X_filtered)
        
        # Plot the original and filtered signals
        plt.figure(figsize=(12, 8))
        
        # Time domain comparison - downsample for display only
        plt.subplot(2, 1, 1)
        if len(t) > max_plot_points:
            plot_step = len(t) // max_plot_points
            plt.plot(t[::plot_step], audio_segment[::plot_step], alpha=0.7, label='Original')
            plt.plot(t[::plot_step], np.real(filtered_signal[::plot_step]), label='Filtered')
        else:
            plt.plot(t, audio_segment, alpha=0.7, label='Original')
            plt.plot(t, np.real(filtered_signal), label='Filtered')
        plt.grid(True, alpha=0.3)
        plt.title(f'Original vs Filtered Signal (Bandpass around {main_freq:.1f} Hz)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Frequency domain comparison - focus on 0-1000 Hz range
        plt.subplot(2, 1, 2)
        filtered_mags = np.abs(X_filtered[:half_n]) / len(filtered_signal)
        plt.plot(freqs[:idx_1000hz], magnitudes[:idx_1000hz], alpha=0.7, label='Original Spectrum')
        plt.plot(freqs[:idx_1000hz], filtered_mags[:idx_1000hz], label='Filtered Spectrum')
        plt.axvline(main_freq, color='r', linestyle='--', 
                   label=f'Main Frequency: {main_freq:.1f} Hz ({identify_note(main_freq)})')
        plt.grid(True, alpha=0.3)
        plt.title('Original vs Filtered Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('audio_filtering.png')
        plt.show()
        
        # Save the filtered audio
        wavfile.write('audio_filtered.wav', fs, np.real(filtered_signal).astype(np.float32))
        print(f"Filtered audio saved as 'audio_filtered.wav' (kept frequencies around {main_freq:.1f} Hz)")


def main():
    
    # Demo 1: Noisy Sinusoid Analysis
    demo_noisy_sinusoid()
    
    # Demo 2: Audio Analysis with Note Identification
    demo_audio_processing()


if __name__ == "__main__":
    main()