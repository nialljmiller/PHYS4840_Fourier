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
    
    Parameters:
        signal_length (int): Length of the signal
        sample_rate (float): Sampling rate in Hz
        
    Returns:
        array: Frequency grid
    """
    # Create frequency grid from 0 to sample_rate/2 (Nyquist frequency)
    # Instead of dense grid, create a sparser grid for better performance with large signals
    # Only use as many points as needed for reasonable resolution (max 8192 points)
    max_points = min(signal_length//2, 8192)
    
    return np.linspace(0, sample_rate/2, max_points)


def find_peaks(spectrum, frequencies, threshold=0.1, min_distance=10):
    """
    Find peaks in the frequency spectrum.
    
    Parameters:
        spectrum (array): Magnitude spectrum
        frequencies (array): Frequency grid
        threshold (float): Threshold for peak detection (relative to max)
        min_distance (int): Minimum distance between peaks (in array indices)
        
    Returns:
        tuple: (peak_frequencies, peak_magnitudes)
    """
    # Find local maxima
    peak_indices = []
    max_val = np.max(spectrum)
    min_val = threshold * max_val
    
    for i in range(1, len(spectrum)-1):
        if (spectrum[i] > spectrum[i-1] and 
            spectrum[i] > spectrum[i+1] and 
            spectrum[i] > min_val):
            
            # Check if this peak is far enough from other peaks
            if not peak_indices or i - peak_indices[-1] >= min_distance:
                peak_indices.append(i)
    
    peak_freqs = frequencies[peak_indices]
    peak_mags = spectrum[peak_indices]
    
    return peak_freqs, peak_mags


def identify_note(frequency):
    """
    Identify musical note from frequency.
    
    Parameters:
        frequency (float): Frequency in Hz
        
    Returns:
        str: Note name
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
    
    # Calculate cents (how far from the exact note, in 1/100 of a semitone)
    cents = 100 * (semitones - semitones_rounded)
    
    return f"{note_names[note_idx]}{octave} ({cents:+.0f} cents)"


def demo_noisy_sinusoid():
    """
    Simple demonstration of Fourier Transform with a noisy sinusoid
    """
    print("\n--- Simple Sinusoid Analysis ---")
    
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
    freqs = create_frequency_grid(len(noisy_signal), fs)
    
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
    """
    Simple demonstration of audio processing with Fourier Transform
    """
    print("\n--- Audio Analysis with Note Identification ---")
    
    try:
        # Load the audio file
        fs, audio_data = wavfile.read('audio.wav')
        
        # Convert to mono if stereo and normalize
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Process the ENTIRE audio as requested
        audio_segment = audio_data  # Use the full audio file
        n_samples = len(audio_segment)
        duration = n_samples / fs
        
        # Create time axis for the full audio
        t = np.linspace(0, duration, n_samples)
        
        print(f"Processing FULL audio: {n_samples} samples, {fs} Hz sample rate, {duration:.2f} seconds")
        
        # For performance reasons with the FULL audio, use NumPy's FFT implementation
        # Our pedagogical DFT would be too slow for the entire file
        print(f"Computing Fourier Transform for all {n_samples} samples...")
        print("Using NumPy's FFT for performance with the full audio file")
        X = np.fft.fft(audio_segment)
        
        # Note: We're using NumPy's FFT here for performance, but in an educational context
        # you could use ft.dft() to see the direct implementation (though it would be very slow)
        
        # Get only the first half of the spectrum (positive frequencies)
        # But downsample to a reasonable number of points for plotting
        max_points = 8192
        half_n = len(X) // 2
        
        # If there are too many points, downsample for visualization
        if half_n > max_points:
            # Downsample by taking every nth point
            step = half_n // max_points
            indices = np.arange(0, half_n, step)
            magnitudes = np.abs(X[indices]) / len(audio_segment)
        else:
            magnitudes = np.abs(X[:half_n]) / len(audio_segment)
        
        # Create frequency grid - for visualization we'll use a sparser grid
        # This ensures we process all the audio but plot at a reasonable resolution
        freqs = create_frequency_grid(len(audio_segment), fs)
        
        # If we downsampled the magnitudes, adjust the frequency grid to match
        if half_n > max_points:
            freqs = np.linspace(0, fs/2, len(magnitudes))
        
        # Find peaks in the spectrum
        peak_freqs, peak_mags = find_peaks(magnitudes, freqs, threshold=0.2)
        
        # Identify musical notes from peaks
        notes = []
        for freq in peak_freqs:
            notes.append(identify_note(freq))
            
        print(f"Found {len(peak_freqs)} significant frequency peaks")
        
        # Plot the audio and its spectrum
        plt.figure(figsize=(12, 8))
        
        # Time domain - plotting entire signal might be too dense, so downsample for visualization
        plt.subplot(2, 1, 1)
        # Downsample time domain signal for plotting if needed
        max_plot_points = 10000
        if len(t) > max_plot_points:
            plot_step = len(t) // max_plot_points
            plt.plot(t[::plot_step], audio_segment[::plot_step])
            plt.title(f'Complete Audio Signal (Downsampled for Display, Full {n_samples} samples processed)')
        else:
            plt.plot(t, audio_segment)
            plt.title('Complete Audio Signal (Time Domain)')
        plt.grid(True, alpha=0.3)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Frequency domain
        plt.subplot(2, 1, 2)
        plt.plot(freqs, magnitudes)
        plt.grid(True, alpha=0.3)
        plt.title('Frequency Spectrum with Identified Notes')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 1000)  # Focus on 0-1000 Hz
        
        # Mark detected peaks and notes
        for i, (freq, mag, note) in enumerate(zip(peak_freqs, peak_mags, notes)):
            if freq <= 1000:  # Only annotate peaks below 1000 Hz
                plt.plot(freq, mag, 'ro', markersize=8)
                plt.text(freq, mag*1.1, f"{note}", ha='center')
        
        plt.tight_layout()
        plt.savefig('audio_analysis.png')
        plt.show()
        
        # Simple filtering demonstration
        print("\nSimple Frequency Filtering Demonstration")
        
        # Define a simple bandpass filter (e.g., keep only the main frequency component)
        if len(peak_freqs) > 0:
            main_freq = peak_freqs[0]
            filter_width = 50  # Hz
            
            print(f"Applying bandpass filter around {main_freq:.1f} Hz")
            
            # Create a filtered spectrum
            X_filtered = np.zeros_like(X, dtype=complex)
            
            # Apply bandpass filter more efficiently
            filter_mask = np.zeros_like(X, dtype=bool)
            
            # Create a mask for the bandpass filter
            for i, freq in enumerate(freqs):
                if abs(freq - main_freq) < filter_width:
                    filter_mask[i] = True
                    # Also set the corresponding negative frequency
                    if i > 0:  # Skip DC component
                        filter_mask[len(X)-i] = True
            
            # Apply the mask
            X_filtered[filter_mask] = X[filter_mask]
            
            # Reconstruct the filtered signal
            print("Reconstructing filtered signal for the entire audio...")
            # Use NumPy's IFFT for better performance with large signals
            filtered_signal = np.fft.ifft(X_filtered)
            
            # Note: Using numpy's IFFT for performance with full audio
            
            # Plot the original and filtered signals
            plt.figure(figsize=(12, 8))
            
            # Time domain comparison
            plt.subplot(2, 1, 1)
            plt.plot(t, audio_segment, alpha=0.7, label='Original')
            plt.plot(t, np.real(filtered_signal), label='Filtered')
            plt.grid(True, alpha=0.3)
            plt.title(f'Original vs Filtered Signal (Bandpass around {main_freq:.1f} Hz)')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()
            
            # Frequency domain comparison
            plt.subplot(2, 1, 2)
            filtered_mags = np.abs(X_filtered[:half_n]) / len(filtered_signal)
            plt.plot(freqs, magnitudes, alpha=0.7, label='Original Spectrum')
            plt.plot(freqs, filtered_mags, label='Filtered Spectrum')
            plt.axvline(main_freq, color='r', linestyle='--', 
                       label=f'Main Frequency: {main_freq:.1f} Hz ({identify_note(main_freq)})')
            plt.grid(True, alpha=0.3)
            plt.title('Original vs Filtered Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.xlim(0, 1000)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('audio_filtering.png')
            plt.show()
            
            # Save the filtered audio
            try:
                wavfile.write('audio_filtered.wav', fs, np.real(filtered_signal).astype(np.float32))
                print(f"Filtered audio saved as 'audio_filtered.wav' (kept frequencies around {main_freq:.1f} Hz)")
            except Exception as e:
                print(f"Error saving filtered audio: {e}")
        else:
            print("No significant peaks found for filtering demonstration")
            
    except Exception as e:
        print(f"Error in audio analysis: {e}")
        print("Make sure you have an 'audio.wav' file in the current directory")


def main():
    print("Simplified Fourier Transform Demonstration")
    print("Using our custom Fourier Transform implementation")
    
    # Demo 1: Noisy Sinusoid Analysis
    demo_noisy_sinusoid()
    
    # Demo 2: Audio Analysis with Note Identification
    demo_audio_processing()


if __name__ == "__main__":
    main()