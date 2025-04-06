# Week 11 - Fourier Analysis & Signal Processing
**PHYS 4840 - Math and Computational Methods II - Spring 2025**
**Sections: 7.1, 7.2, 7.3, 7.4 -- COMPUTATIONAL PHYSICS - Mark Newman**
**MSMA 11.5 - Modern Statistical Methods for Astronomy**

## Introduction to Fourier Analysis

Fourier analysis is a mathematical technique that decomposes functions into oscillatory components (sines and cosines). This powerful approach allows us to transform signals between the time/space domain and the frequency domain, revealing insights that may not be apparent in the original representation.

### Why Fourier Analysis is Important in Physics

In physics, many systems exhibit oscillatory or wave-like behavior. Fourier analysis provides us with the tools to:

1. Decompose complex signals into their fundamental frequency components
2. Identify dominant frequencies in experimental data
3. Simplify differential equations by transforming them to algebraic equations
4. Analyze the frequency response of physical systems
5. Extract periodic signals from noisy data

For computational physicists, Fourier techniques are essential because they:
- Provide efficient algorithms for solving certain types of problems
- Allow for filtering and processing of data in ways not possible in the original domain
- Enable compression of data by focusing on the most significant frequency components

## Mathematical Foundations

### Fourier Series: Breaking Down Periodic Functions

For any periodic function $f(x)$ with period $2\pi$, the Fourier series representation is:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(nx) + b_n \sin(nx) \right]$$

where the coefficients are calculated as:

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) \, dx$$
$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx$$

#### Physical Interpretation:
- $a_0/2$ represents the average value (DC component) of the function
- Each $a_n$ and $b_n$ pair represents the contribution of the frequency $n$ to the overall signal
- Higher values of $n$ correspond to higher frequencies (faster oscillations)
- The magnitude $\sqrt{a_n^2 + b_n^2}$ indicates the strength of frequency $n$
- The phase angle $\arctan(b_n/a_n)$ indicates the phase shift of frequency $n$

#### Complex Form:
We can also express the Fourier series using complex exponentials, which is often more convenient mathematically:

$$f(x) = \sum_{n=-\infty}^{\infty} c_n e^{inx}$$

where:

$$c_n = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x) e^{-inx} \, dx$$

### Fourier Transform: Extending to Non-Periodic Functions

For non-periodic functions $f(x)$, we use the Fourier transform:

$$F(k) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i k x} \, dx$$

And the inverse transform:

$$f(x) = \int_{-\infty}^{\infty} F(k) e^{2\pi i k x} \, dk$$

#### Key Differences from Fourier Series:
- Deals with aperiodic functions (signals of infinite length)
- Produces a continuous spectrum rather than discrete frequency components
- $F(k)$ gives the amplitude density at frequency $k$

#### Physical Interpretation:
- $|F(k)|$ represents the amplitude of frequency $k$ in the original signal
- $\arg(F(k))$ represents the phase of frequency $k$
- The Fourier transform preserves the total energy of the signal (Parseval's theorem)

### Discrete Fourier Transform (DFT): Working with Sampled Data

In computational physics, we typically work with discrete, finite-length signals. For a signal $\{x_0, x_1, ..., x_{N-1}\}$ sampled at equally spaced points, the DFT is defined as:

$$X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i k n / N} \quad \text{for} \quad k = 0, 1, ..., N-1$$

And the inverse DFT:

$$x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{2\pi i k n / N} \quad \text{for} \quad n = 0, 1, ..., N-1$$

#### Implementation Considerations:
- Direct calculation is computationally expensive: $O(N^2)$
- The outcome $X_k$ represents the discrete frequency components of the original signal
- The frequency resolution is determined by the sampling rate and the number of samples

#### Important DFT Properties:
1. **Periodicity**: The DFT assumes the input signal is periodic with period N
2. **Linearity**: The DFT of a sum of signals equals the sum of their DFTs
3. **Conjugate Symmetry**: For real inputs, $X_{N-k} = X_k^*$ (complex conjugate)
4. **Circular Shift**: Shifting the input signal results in a phase change in frequency domain
5. **Parseval's Theorem**: Energy is conserved between time and frequency domains

## Fast Fourier Transform (FFT): Making DFT Computationally Feasible

The Fast Fourier Transform is an algorithm for computing the DFT with reduced time complexity: $O(N \log N)$ instead of $O(N^2)$. This dramatic improvement makes Fourier analysis practical for large datasets.

### How FFT Works:

The most common FFT algorithm (Cooley-Tukey) works by:

1. Recursively dividing the DFT calculation into smaller DFTs
2. Exploiting symmetries and periodicities in the complex exponential terms
3. Combining results using the "butterfly" pattern to build up the complete transform

The key insight is to divide a length-N DFT into two length-N/2 DFTs, one for even-indexed inputs and one for odd-indexed inputs:

$$X_k = \sum_{m=0}^{N/2-1} x_{2m} e^{-2\pi i k (2m) / N} + \sum_{m=0}^{N/2-1} x_{2m+1} e^{-2\pi i k (2m+1) / N}$$

This can be rewritten as:

$$X_k = E_k + e^{-2\pi i k / N} O_k$$

where $E_k$ and $O_k$ are the DFTs of the even and odd elements, respectively.

### Implementation Example:

```python
def fft_recursive(x):
    """Basic recursive FFT implementation for signals with length = power of 2"""
    N = len(x)
    
    # Base case
    if N == 1:
        return x
    
    # Divide
    even = fft_recursive(x[0::2])  # Even-indexed elements
    odd = fft_recursive(x[1::2])   # Odd-indexed elements
    
    # Combine
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    result = np.zeros(N, dtype=complex)
    half_N = N // 2
    
    for k in range(half_N):
        result[k] = even[k] + factor[k] * odd[k]
        result[k + half_N] = even[k] - factor[k] * odd[k]
    
    return result
```

### FFT Performance Considerations:

- Most efficient when N is a power of 2
- Memory access patterns can significantly affect performance
- Modern FFT libraries (like FFTW or NumPy's implementation) use various optimizations
- Can be parallelized for further performance improvements

## Critical Sampling and Frequency Considerations

### Nyquist-Shannon Sampling Theorem

The Nyquist-Shannon sampling theorem is fundamental to digital signal processing and states:

**A continuous signal with maximum frequency component $f_{max}$ must be sampled at a rate of at least $2f_{max}$ to be reconstructed without aliasing.**

#### Why This Matters:
- If we sample too slowly, high-frequency components "fold back" and appear as lower frequencies (aliasing)
- This limitation is inherent to discrete sampling and cannot be overcome after sampling
- Real-world signals often need low-pass filtering before digitization to prevent aliasing

#### Mathematical Formulation:
To avoid aliasing, we need:

$$f_s > 2f_{max}$$

where $f_s$ is the sampling frequency and $f_{max}$ is the highest frequency in the signal.

The maximum frequency that can be represented at a given sampling rate is called the **Nyquist frequency**:

$$f_{Nyquist} = \frac{f_s}{2}$$

### Frequency Resolution and Spectral Leakage

#### Frequency Resolution
The ability to distinguish between nearby frequencies depends on the total sampling duration:

$$\Delta f = \frac{f_s}{N}$$

where $N$ is the number of samples and $f_s$ is the sampling rate.

This means that longer data records (more samples) provide better frequency resolution.

#### Spectral Leakage
When the signal contains a frequency component that doesn't align with the DFT frequency bins, energy "leaks" into adjacent bins. This happens because the DFT implicitly assumes the signal is periodic with period $N$.

To mitigate spectral leakage:
1. Ensure the sampling window contains an integer number of periods (not always possible)
2. Apply windowing functions to taper the signal at the edges
3. Use zero-padding to increase the apparent frequency resolution

## Windowing Functions: Reducing Spectral Leakage

Windowing functions multiply the time-domain signal before performing the Fourier transform to reduce the discontinuities at the boundaries.

### Popular Windowing Functions:

| Window Type | Main Lobe Width | Side Lobe Attenuation | Trade-offs |
|-------------|-----------------|------------------------|----------|
| Rectangular | Narrowest | Poorest (-13 dB) | Best frequency resolution, worst spectral leakage |
| Hamming | Medium | Better (-43 dB) | Good general-purpose window |
| Hann | Medium | Good (-32 dB) | Reduces side lobes at cost of main lobe width |
| Blackman | Widest | Best (-58 dB) | Excellent side lobe suppression, worst frequency resolution |

### Implementation Example:

```python
def apply_window(signal, window_type='hann'):
    """Apply a windowing function to a signal"""
    N = len(signal)
    
    if window_type == 'rectangular':
        window = np.ones(N)
    elif window_type == 'hamming':
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    elif window_type == 'hann':
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))
    elif window_type == 'blackman':
        window = (0.42 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1)) + 
                 0.08 * np.cos(4 * np.pi * np.arange(N) / (N - 1)))
    
    return signal * window
```

### Window Selection Guidelines:

- Choose **rectangular** when analyzing transients or measuring exact amplitudes in high SNR signals
- Choose **Hamming** for general-purpose spectral analysis
- Choose **Hann** when the signal contains closely spaced frequencies of similar amplitudes
- Choose **Blackman** when looking for weak spectral features near strong ones

## Zero-Padding and Interpolation

Zero-padding involves appending zeros to the time-domain signal before computing the DFT.

### Benefits of Zero-Padding:
1. Increases the number of frequency points in the DFT output
2. Provides smoother appearance of the spectrum (interpolation)
3. Can help identify peaks that fall between frequency bins

### Limitations of Zero-Padding:
1. Does NOT improve the fundamental frequency resolution (determined by signal duration)
2. Cannot reveal frequency components that aren't present in the original data
3. Introduces computational overhead

### Implementation Example:

```python
def zero_pad_fft(signal, pad_factor=2):
    """Compute FFT with zero padding for smoother spectrum"""
    N = len(signal)
    padded_signal = np.zeros(N * pad_factor)
    padded_signal[:N] = signal
    return np.fft.fft(padded_signal)
```

## Applications of Fourier Analysis in Physics

### 1. Signal Processing
- **Filtering**: Remove noise or unwanted frequency components
- **Deconvolution**: Extract the original signal from a convolved or blurred version
- **Correlation Analysis**: Find patterns or similarities in signals

### 2. Wave Phenomena
- **Wave Propagation**: Analyze and predict wave behavior
- **Optics**: Study diffraction and interference patterns
- **Acoustic Analysis**: Identify resonant frequencies and harmonics

### 3. Astronomy
- **Periodicity Detection**: Find periodic sources like pulsars or variable stars
- **Redshift Analysis**: Analyze spectral lines from distant galaxies
- **Time Series Analysis**: Process data from telescopes and other instruments

### 4. Quantum Mechanics
- **Momentum Space Representations**: Transform between position and momentum space
- **Energy Level Analysis**: Study energy spectra of quantum systems
- **Quantum Computing Algorithms**: Some quantum algorithms rely on Fourier transforms

### 5. Computational Fluid Dynamics
- **Spectral Methods**: Solve PDEs using Fourier basis functions
- **Turbulence Analysis**: Study energy cascades across different scales
- **Flow Visualization**: Identify dominant flow structures

## Practical Implementation in Python

### Basic FFT Example:

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a test signal (sum of sine waves + noise)
fs = 1000  # Sampling rate (Hz)
T = 1      # Duration (seconds)
t = np.linspace(0, T, fs*T, endpoint=False)  # Time vector

# Signal with 50 Hz and 120 Hz components
signal = (1.0 * np.sin(2 * np.pi * 50 * t) + 
          0.5 * np.sin(2 * np.pi * 120 * t) + 
          0.2 * np.random.randn(len(t)))  # Add noise

# Compute the FFT
N = len(signal)
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(N, 1/fs)  # Frequency bins

# Plot only the positive frequencies (since signal is real)
positive_freq_mask = frequencies >= 0
plt.figure(figsize=(10, 6))
plt.plot(frequencies[positive_freq_mask], np.abs(fft_result[positive_freq_mask])/N)
plt.grid(True)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum')
plt.xlim(0, fs/2)  # Display up to Nyquist frequency
```

### FFT-Based Filtering:

```python
def bandpass_filter(signal, fs, lowcut, highcut):
    """Apply a frequency domain bandpass filter to a signal"""
    # Compute FFT
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/fs)
    
    # Create a mask for the bandpass region
    mask = (abs(frequencies) >= lowcut) & (abs(frequencies) <= highcut)
    
    # Apply the filter
    filtered_fft = fft_result * mask
    
    # Inverse FFT to return to time domain
    return np.real(np.fft.ifft(filtered_fft))
```

## Error Considerations in Fourier Analysis

### 1. Aliasing

Aliasing occurs when a signal contains frequencies above the Nyquist frequency, causing those components to appear as lower frequencies in the sampled signal.

#### Mitigation:
- **Anti-aliasing filters**: Apply a low-pass filter before digitization
- **Oversampling**: Sample at a higher rate than strictly required
- **Analytical verification**: Check if results make physical sense

### 2. Spectral Leakage

Spectral leakage occurs when the signal is not periodic within the sampling window, causing energy to "leak" into adjacent frequency bins.

#### Mitigation:
- **Windowing**: Apply tapered windows to reduce discontinuities
- **Appropriate record length**: Try to capture an integer number of periods
- **Zero-padding**: Increase apparent frequency resolution for peak identification

### 3. Picket Fence Effect

The picket fence effect refers to the limited frequency resolution in the DFT, where the true peak of a frequency component may fall between discrete frequency bins.

#### Mitigation:
- **Zero-padding**: Add zeros to increase the number of frequency points
- **Interpolation**: Estimate the true peak location using interpolation
- **Longer data record**: Increase the fundamental frequency resolution

## Computing Considerations

### Performance Optimization:
1. **Use power-of-2 lengths** for most efficient FFT computation
2. **Pre-allocate arrays** to avoid memory reallocation during computation
3. **Consider batch processing** for multiple transforms of the same size
4. **Use in-place FFT algorithms** for large datasets to reduce memory usage

### Numerical Precision:
1. **Normalize properly** to minimize floating-point errors
2. **Consider using double precision** for sensitive applications
3. **Monitor for accumulation of roundoff errors** in iterative algorithms

## Files in this Repository

- `fourier_series.py`: Implementation and visualization of Fourier series
- `dft_implementation.py`: Direct implementation of the Discrete Fourier Transform
- `fft_implementation.py`: Fast Fourier Transform implementation and analysis
- `signal_processing_examples.py`: Real-world applications of Fourier analysis

Each file includes detailed comments and examples to help you understand the implementation details.

## References

1. Newman, M. (2013). Computational Physics. CreateSpace Independent Publishing Platform.
2. Feigelson, E. D., & Babu, G. J. (2012). Modern Statistical Methods for Astronomy. Cambridge University Press.