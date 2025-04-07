# Fourier Analysis in Computational Physics

This repository contains implementations and explanations of Fourier analysis techniques for computational physics applications. Fourier methods are fundamental tools for solving problems in physics that involve periodic phenomena, wave propagation, signal processing, and partial differential equations.

## Table of Contents

1. [Fourier Series](#fourier-series)
2. [Discrete Fourier Transform](#discrete-fourier-transform)
3. [Discrete Cosine and Sine Transforms](#discrete-cosine-and-sine-transforms)
4. [Fast Fourier Transform](#fast-fourier-transform)
5. [Advanced Topics](#advanced-topics)
   - [Gibbs Phenomenon](#gibbs-phenomenon)
   - [Windowing Functions](#windowing-functions)
   - [Filtering and Smoothing](#filtering-and-smoothing)
6. [Applications](#applications)
7. [Implementation Examples](#implementation-examples)
8. [References](#references)

## Fourier Series

### Mathematical Foundations

The Fourier series represents any periodic function as a sum of sines and cosines. For a function $f(x)$ with period $2\pi$, the Fourier series is:

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos(nx) + b_n \sin(nx) \right]$$

Where the coefficients are calculated as:

$$a_0 = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \, dx$$

$$a_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos(nx) \, dx$$

$$b_n = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin(nx) \, dx$$

### Visualizing Fourier Series

A Fourier series can be understood as:
- $a_0/2$: The average value (DC component)
- $a_1\cos(x) + b_1\sin(x)$: The fundamental frequency (first harmonic)
- $a_2\cos(2x) + b_2\sin(2x)$: The second harmonic
- Higher terms: Progressively faster oscillations adding finer details

### Convergence Properties

The convergence of a Fourier series depends on the function's smoothness:
- Continuous functions: Pointwise convergence
- Differentiable functions: Uniform convergence
- Discontinuous functions: Gibbs phenomenon occurs at jumps

### Implementation Example

```python
def compute_coefficients(func, n_terms, period=2*np.pi, num_points=1000):
    """
    Compute Fourier coefficients up to a specified number of terms.
    
    Parameters:
        func (callable): Function to approximate
        n_terms (int): Number of terms in the Fourier series
        period (float): Period of the function
        num_points (int): Points for numerical integration
    
    Returns:
        tuple: (a0, an_coefficients, bn_coefficients)
    """
    # Compute a0 (constant term)
    x = np.linspace(0, period, num_points)
    y = func(x)
    a0 = np.mean(y)
    
    # Compute an and bn coefficients
    an = np.zeros(n_terms)
    bn = np.zeros(n_terms)
    
    for n in range(1, n_terms + 1):
        # an coefficients (cosine terms)
        integrand = y * np.cos(2 * np.pi * n * x / period)
        an[n-1] = 2 * np.mean(integrand)
        
        # bn coefficients (sine terms)
        integrand = y * np.sin(2 * np.pi * n * x / period)
        bn[n-1] = 2 * np.mean(integrand)
    
    return a0, an, bn
```

## Discrete Fourier Transform

### From Continuous to Discrete

While the Fourier series applies to continuous, periodic functions, the Discrete Fourier Transform (DFT) is designed for sampled, finite-length signals. The DFT transforms a sequence of $N$ complex numbers into another sequence of complex numbers representing frequency components.

### Mathematical Formulation

For a sequence $x_0, x_1, ..., x_{N-1}$, the DFT is defined as:

$$X_k = \sum_{n=0}^{N-1} x_n e^{-i\frac{2\pi}{N}kn} \quad \text{for } k = 0, 1, ..., N-1$$

The inverse DFT (IDFT) is:

$$x_n = \frac{1}{N}\sum_{k=0}^{N-1} X_k e^{i\frac{2\pi}{N}kn} \quad \text{for } n = 0, 1, ..., N-1$$

In terms of sines and cosines:

$$X_k = \sum_{n=0}^{N-1} x_n \cos\left(\frac{2\pi k n}{N}\right) - i \sum_{n=0}^{N-1} x_n \sin\left(\frac{2\pi k n}{N}\right)$$

### Key Properties of the DFT

| Property | Description |
|----------|-------------|
| Linearity | DFT of a sum equals sum of DFTs |
| Circular Shift | Time shift results in phase change |
| Conjugate Symmetry | For real inputs, $X_{N-k} = X_k^*$ |
| Parseval's Theorem | Energy conservation between domains |
| Convolution | Time domain convolution = frequency domain multiplication |

### Implementation Example (Naive DFT)

```python
def dft_naive(x):
    """
    Compute the Discrete Fourier Transform (DFT) using direct implementation.
    
    Parameters:
        x (array): Input signal array
        
    Returns:
        array: DFT of x (complex array)
    """
    N = len(x)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    
    return X
```

### Computational Complexity

The naive DFT implementation has a time complexity of $O(N^2)$, making it impractical for large signals. This led to the development of the Fast Fourier Transform algorithm.

## Discrete Cosine and Sine Transforms

### DCT: Mathematical Formulation

The Discrete Cosine Transform (DCT) uses only cosine functions and produces real-valued outputs for real-valued inputs. The most common form (DCT-II) is:

$$X_k = \sum_{n=0}^{N-1} x_n \cos\left[\frac{\pi}{N}(n+\frac{1}{2})k\right] \quad \text{for } k = 0, 1, ..., N-1$$

### DST: Mathematical Formulation

Similarly, the Discrete Sine Transform (DST) uses only sine functions:

$$X_k = \sum_{n=0}^{N-1} x_n \sin\left[\frac{\pi}{N}(n+\frac{1}{2})(k+1)\right] \quad \text{for } k = 0, 1, ..., N-1$$

### Advantages and Applications

- **Energy Compaction**: DCT concentrates energy in fewer coefficients
- **Real-valued**: Computationally more efficient for real data
- **Applications**: JPEG image compression (DCT), MPEG video compression
- **Boundary Conditions**: Better suited for certain boundary conditions in differential equations

### Implementation Example (DCT-II)

```python
def dct_ii(x):
    """
    Compute the DCT-II of input signal x.
    
    Parameters:
        x (array): Input signal array (real)
        
    Returns:
        array: DCT of x (real array)
    """
    N = len(x)
    X = np.zeros(N)
    
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.cos(np.pi * (n + 0.5) * k / N)
    
    # Scale the DC component differently
    X[0] *= np.sqrt(1/N)
    # Scale AC components
    X[1:] *= np.sqrt(2/N)
    
    return X
```

## Fast Fourier Transform

### The Breakthrough Algorithm

The Fast Fourier Transform (FFT) is an algorithm that computes the DFT in $O(N \log N)$ time, making Fourier analysis practical for real-world applications.

### Divide and Conquer Strategy

The key insight of the FFT (specifically the Cooley-Tukey algorithm) is to recursively split the DFT computation into smaller DFTs:

1. Split the N-point DFT into two N/2-point DFTs:
   - One for even-indexed points
   - One for odd-indexed points
2. Combine results using the "butterfly" pattern

### Computational Complexity Comparison

| Algorithm | Time Complexity | Space Complexity | Example: N=1,000,000 |
|-----------|-----------------|------------------|----------------------|
| Naive DFT | $O(N^2)$        | $O(N)$           | ~1 trillion operations |
| FFT       | $O(N \log N)$   | $O(N)$           | ~20 million operations |

This dramatic improvement from $O(N^2)$ to $O(N \log N)$ is what made Fourier analysis practically useful for large datasets.

### Recursive Implementation Example

```python
def fft_recursive(x):
    """
    Compute the FFT using the Cooley-Tukey algorithm.
    Works for signal lengths that are powers of 2.
    
    Parameters:
        x (array): Input signal array
        
    Returns:
        array: FFT of x (complex array)
    """
    N = len(x)
    
    # Base case: FFT of a single point is the point itself
    if N == 1:
        return x
    
    # Check if N is a power of 2
    if N & (N-1) != 0:
        raise ValueError("Signal length must be a power of 2")
    
    # Split even and odd indices
    even = fft_recursive(x[0::2])
    odd = fft_recursive(x[1::2])
    
    # Twiddle factors
    twiddle = np.exp(-2j * np.pi * np.arange(N//2) / N)
    
    # Combine using butterfly pattern
    result = np.zeros(N, dtype=complex)
    half_N = N // 2
    
    for k in range(half_N):
        result[k] = even[k] + twiddle[k] * odd[k]
        result[k + half_N] = even[k] - twiddle[k] * odd[k]
    
    return result
```

### Iterative Implementation Considerations

While the recursive implementation is elegant, an iterative implementation with bit-reversal permutation is often more efficient in practice:

```python
def bit_reversal_permutation(N):
    """Generate bit-reversal permutation indices for FFT."""
    num_bits = int(np.log2(N))
    indices = np.zeros(N, dtype=int)
    
    for i in range(N):
        binary = format(i, f'0{num_bits}b')
        reversed_binary = binary[::-1]
        indices[i] = int(reversed_binary, 2)
    
    return indices
```

## Advanced Topics

### Gibbs Phenomenon

The Gibbs phenomenon is an oscillatory behavior that occurs near discontinuities when approximating a function with Fourier series. Key characteristics:

- Overshoot near discontinuities (about 9% of jump magnitude)
- Overshoot doesn't diminish with more terms
- Width of the oscillations decreases with more terms
- Total area of the oscillations approaches zero

### Windowing Functions

Windowing functions are applied to signals before Fourier analysis to reduce spectral leakage:

| Window | Main Lobe Width | Side Lobe Attenuation | Use Case |
|--------|-----------------|------------------------|----------|
| Rectangular | Narrowest | -13 dB | Best frequency resolution |
| Hamming | Medium | -43 dB | General-purpose |
| Hann | Medium | -32 dB | Good balance |
| Blackman | Wide | -58 dB | Excellent side lobe suppression |

#### Implementation Example

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

### Filtering and Smoothing

Frequency domain filtering allows selective manipulation of signal components:

#### Low-Pass Filter Example

```python
def lowpass_filter(signal, cutoff_freq, sample_rate):
    """Apply a low-pass filter in the frequency domain"""
    # Compute FFT
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    # Create filter mask
    mask = abs(frequencies) <= cutoff_freq
    
    # Apply filter
    filtered_fft = fft_result * mask
    
    # Inverse FFT
    return np.real(np.fft.ifft(filtered_fft))
```

### Nyquist-Shannon Sampling Theorem

A fundamental principle in signal processing:

- To accurately represent a signal, sample at a rate greater than twice its highest frequency
- Sampling rate $f_s > 2f_{max}$
- Maximum recoverable frequency: $f_{Nyquist} = \frac{f_s}{2}$
- Aliasing occurs when signals contain frequencies above the Nyquist frequency

### Frequency Resolution and Zero-Padding

Frequency resolution in DFT depends on the observation length:

- Resolution: $\Delta f = \frac{f_s}{N} = \frac{1}{T}$
- Zero-padding increases the number of frequency points but doesn't improve fundamental resolution
- Longer signals provide better frequency resolution

## Applications

### Signal Processing
- Filtering (removing noise or unwanted frequencies)
- Spectral analysis
- Audio processing

### Image Processing
- Image compression (JPEG, etc.)
- Feature extraction
- Edge detection

### Differential Equations
- Solving PDEs using spectral methods
- Fast Poisson solvers
- Heat equation solutions

### Astronomy and Physics
- Analysis of periodic phenomena
- Spectral line identification
- Gravitational wave detection

## Implementation Examples

This repository contains several example implementations:

- `fourier_series.py`: Compute and visualize Fourier series
- `dft_implementation.py`: DFT implementation and analysis
- `fft_implementation.py`: Fast Fourier Transform algorithms
- `signal_processing.py`: Real-world signal processing applications

## References

1. Newman, M. (2013). Computational Physics. CreateSpace Independent Publishing Platform.
2. Feigelson, E. D., & Babu, G. J. (2012). Modern Statistical Methods for Astronomy. Cambridge University Press.
3. Cooley, J. W., & Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. Mathematics of Computation, 19(90), 297-301.
4. Brigham, E. O. (1988). The Fast Fourier Transform and Its Applications. Prentice Hall.