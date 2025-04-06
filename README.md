# Week 11 - Fourier Analysis & Signal Processing
**PHYS 4840 - Math and Computational Methods II - Spring 2025**
**Sections: 7.1, 7.2, 7.3, 7.4 -- COMPUTATIONAL PHYSICS - Mark Newman**
**MSMA 11.5 - Modern Statistical Methods for Astronomy**

## Introduction to Fourier Analysis

Fourier analysis is a mathematical technique that decomposes functions into oscillatory components (sines and cosines). This powerful approach allows us to transform signals between the time/space domain and the frequency domain, revealing insights that may not be apparent in the original representation.

### Key Concepts

- **Fourier Series**: Representation of a periodic function as a sum of sines and cosines
- **Fourier Transform**: Extension of Fourier series to non-periodic functions
- **Discrete Fourier Transform (DFT)**: Fourier transform for discrete, finite-length signals
- **Fast Fourier Transform (FFT)**: Efficient algorithm to compute the DFT

## Mathematical Foundations

### Fourier Series

For a periodic function $f(x)$ with period $2\pi$, the Fourier series is given by:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(nx) + b_n \sin(nx) \right]$$

where the coefficients are calculated as:

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) \, dx$$
$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx$$

### Fourier Transform

For a non-periodic function $f(x)$, the Fourier transform is defined as:

$$F(k) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i k x} \, dx$$

And the inverse transform:

$$f(x) = \int_{-\infty}^{\infty} F(k) e^{2\pi i k x} \, dk$$

### Discrete Fourier Transform (DFT)

For a discrete signal $\{x_0, x_1, ..., x_{N-1}\}$, the DFT is defined as:

$$X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i k n / N} \quad \text{for} \quad k = 0, 1, ..., N-1$$

And the inverse DFT:

$$x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{2\pi i k n / N} \quad \text{for} \quad n = 0, 1, ..., N-1$$

## Fast Fourier Transform (FFT)

The FFT is an algorithm that computes the DFT in $O(N \log N)$ time, a significant improvement over the $O(N^2)$ time complexity of the naive DFT implementation. The most common FFT algorithm is the Cooley-Tukey algorithm, which recursively divides the DFT into smaller DFTs.

### Key Properties of FFT

- **Computational Efficiency**: $O(N \log N)$ vs $O(N^2)$ for direct DFT calculation
- **Memory Usage**: Can be implemented in-place with $O(N)$ memory
- **Numerical Stability**: Generally stable, especially for power-of-2 sized inputs

## Applications of Fourier Analysis

- **Signal Processing**: Filtering, compression, spectral analysis
- **Image Processing**: Image enhancement, feature extraction, compression
- **Acoustics**: Sound analysis, noise reduction
- **Physics**: Wave phenomena, quantum mechanics
- **Astronomy**: Time series analysis, detecting periodic signals in data
- **Communications**: Modulation, multiplexing, bandwidth analysis

## Files in this Repository

- `fourier_series.py`: Implementation and visualization of Fourier series
- `dft_implementation.py`: Direct implementation of the Discrete Fourier Transform
- `fft_implementation.py`: Fast Fourier Transform implementation and analysis
- `signal_processing_examples.py`: Real-world applications of Fourier analysis

## Error Considerations in Fourier Analysis

### Aliasing
When sampling a continuous signal, frequencies above the Nyquist frequency (half the sampling rate) can appear as lower frequencies in the transform, causing aliasing. This is a fundamental limitation of discrete sampling.

### Leakage
When the signal's period doesn't match the sampling window, energy "leaks" from the main frequency peak into adjacent frequency bins. This can be mitigated using windowing functions.

### Picket Fence Effect
Discrete frequency bins can miss the true peak of a frequency component that falls between bins. Zero-padding or interpolation can help improve frequency resolution.

## Windowing Functions

Windowing functions taper the signal at the edges of the analysis frame to reduce spectral leakage:

| Window Type | Main Lobe Width | Side Lobe Attenuation | Use Case |
|-------------|-----------------|------------------------|----------|
| Rectangular | Narrowest | Poorest (-13 dB) | Transient analysis |
| Hamming | Wider | Better (-43 dB) | General purpose |
| Hann | Wider | Good (-32 dB) | Spectral analysis |
| Blackman | Widest | Best (-58 dB) | Harmonic analysis |

## References

1. Newman, M. (2013). Computational Physics. CreateSpace Independent Publishing Platform.
2. Feigelson, E. D., & Babu, G. J. (2012). Modern Statistical Methods for Astronomy. Cambridge University Press.
3. Brigham, E. O. (1988). The Fast Fourier Transform and Its Applications. Prentice-Hall.
4. Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007). Numerical Recipes: The Art of Scientific Computing (3rd ed.). Cambridge University Press.