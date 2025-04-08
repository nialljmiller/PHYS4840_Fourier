# Fourier Analysis in Computational Physics

This repository contains implementations and explanations of Fourier analysis techniques for computational physics applications. Fourier methods are fundamental tools for solving problems in physics that involve periodic phenomena, wave propagation, signal processing, and partial differential equations.

## Table of Contents

1. [Fourier Series](#fourier-series)
2. [Gibbs Phenomenon](#gibbs-phenomenon)
3. [Fourier Transform (DFT)](#fourier-transform-dft)
4. [Inverse Fourier Transform](#inverse-fourier-transform)
5. [Fast Fourier Transform](#fast-fourier-transform)
6. [Discrete Cosine and Sine Transforms](#discrete-cosine-and-sine-transforms) (TBA)
7. [Advanced Topics](#advanced-topics) (TBA)
   - [Windowing Functions](#windowing-functions) (TBA)
   - [Filtering and Smoothing](#filtering-and-smoothing) (TBA)
   - [Nyquist-Shannon Sampling Theorem](#nyquist-shannon-sampling-theorem) (TBA)
   - [Frequency Resolution and Zero-Padding](#frequency-resolution-and-zero-padding) (TBA)
8. [Applications](#applications) (TBA)
9. [Implementation Examples](#implementation-examples) (TBA)
10. [References](#references)

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
        an[n-1] = integral of above
        
        # bn coefficients (sine terms)
        integrand = y * np.sin(2 * np.pi * n * x / period)
        bn[n-1] = integral of above
    
    return a0, an, bn
```

## Gibbs Phenomenon

The Gibbs phenomenon is an oscillatory behavior that occurs near discontinuities when approximating a function with Fourier series. Key characteristics:

- Overshoot near discontinuities (about 9% of jump magnitude)
- Overshoot doesn't diminish with more terms
- Width of the oscillations decreases with more terms
- Total area of the oscillations approaches zero

## Fourier Transform (DFT)

### From Continuous to Discrete

While the Fourier series applies to continuous, periodic functions, the Discrete Fourier Transform (DFT) is designed for sampled, finite-length signals. The DFT transforms a sequence of $N$ complex numbers into another sequence of complex numbers representing frequency components.

### Mathematical Formulation

For a sequence $x_0, x_1, ..., x_{N-1}$, the DFT is defined as:

$$X_k = \sum_{n=0}^{N-1} x_n e^{-i\frac{2\pi}{N}kn} \quad \text{for } k = 0, 1, ..., N-1$$

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

## Inverse Fourier Transform

The inverse DFT (IDFT) reverses the transformation, taking frequency components back to the time domain:

$$x_k = \frac{1}{N}\sum_{n=0}^{N-1} X_n e^{i\frac{2\pi}{N}kn} \quad \text{for } k = 0, 1, ..., N-1$$

Implementation example:

```python
def idft_naive(X):
    """
    Compute the Inverse Discrete Fourier Transform (IDFT).
    
    Parameters:
        X (array): Input frequency domain signal array
        
    Returns:
        array: IDFT of X (complex array)
    """
    N = len(X)
    x = np.zeros(N, dtype=complex)
    
    for k in range(N):
        for n in range(N):
            x[k] += X[n] * np.exp(2j * np.pi * k * n / N)
    
    # Normalize by N
    x = x / N
    
    return x
```



## Fast Fourier Transform

### The Breakthrough Algorithm

The Fast Fourier Transform (FFT) is an algorithm that computes the Discrete Fourier Transform (DFT) in O(N log N) time, making Fourier analysis practical for real-world applications. This remarkable improvement over the naive O(N²) DFT implementation represents one of the most significant algorithmic breakthroughs in computational mathematics.

### Understanding Computational Complexity

When analyzing algorithms, we use Big O notation to describe how their performance scales with input size:

- **O(N²)**: The computational cost grows quadratically with input size N (naive DFT)
- **O(N log N)**: The computational cost grows much more slowly (FFT)

To appreciate this difference, consider the computational requirements for a signal with N = 1,000,000 points:
- Naive DFT: ~1,000,000,000,000 (trillion) operations
- FFT: ~20,000,000 (20 million) operations

This approximately 50,000× speedup is what made many modern digital signal processing applications feasible.

### Divide and Conquer Strategy

The Cooley-Tukey FFT algorithm uses an elegant recursive approach:

1. **Insight**: Computing two N/2-point DFTs and combining them is much cheaper than one N-point DFT
2. **Recursive decomposition**: Split an N-point transform into:
   - One N/2-point transform of the even-indexed elements
   - One N/2-point transform of the odd-indexed elements
3. **Recombination**: Combine these results using the "butterfly" pattern

The "butterfly" name comes from the shape of the data flow diagram:
```
A ----> A + W*B
|       |
X       X  ← "Butterfly" shape
|       |
B ----> A - W*B
```
Where W is a "twiddle factor" (complex exponential term).

### Recursive Implementation

```python
def fft(x):
    """
    Compute the Fast Fourier Transform using the Cooley-Tukey algorithm.
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
    even = fft(x[0::2])  # Recursive call on even-indexed elements
    odd = fft(x[1::2])   # Recursive call on odd-indexed elements
    
    # Twiddle factors
    twiddle = np.exp(-2j * np.pi * np.arange(N//2) / N)
    
    # Combine using butterfly pattern
    result = np.zeros(N, dtype=complex)
    half_N = N // 2
    
    for k in range(half_N):
        result[k] = even[k] + twiddle[k] * odd[k]  # First half frequencies
        result[k + half_N] = even[k] - twiddle[k] * odd[k]  # Second half frequencies
    
    return result
```

The mathematical elegance of this approach lies in exploiting the symmetry properties of the DFT to avoid redundant calculations.

### Iterative Implementation

While the recursive implementation illustrates the algorithm clearly, production code typically uses an iterative implementation for better performance:

```python
def bit_reversal_permutation(x):
    """
    Rearrange array using bit-reversal permutation.
    
    Parameters:
        x (array): Input array
        
    Returns:
        array: Permuted array
    """
    N = len(x)
    num_bits = int(np.log2(N))
    permuted = np.zeros_like(x)
    
    for i in range(N):
        # Convert i to binary, reverse bits, convert back to decimal
        binary = format(i, f'0{num_bits}b')
        reversed_binary = binary[::-1]
        j = int(reversed_binary, 2)
        permuted[j] = x[i]
    
    return permuted

def iterative_fft(x):
    """
    Compute FFT using an iterative implementation.
    
    Parameters:
        x (array): Input signal
        
    Returns:
        array: FFT of x
    """
    N = len(x)
    
    # Check if N is a power of 2
    if N & (N-1) != 0:
        raise ValueError("Signal length must be a power of 2")
    
    # Bit-reversal permutation
    x = bit_reversal_permutation(x)
    
    # Main FFT computation
    num_stages = int(np.log2(N))
    
    # For each stage
    for stage in range(1, num_stages + 1):
        butterfly_size = 2 ** stage
        half_size = butterfly_size // 2
        
        # Twiddle factors for this stage
        omega = np.exp(-2j * np.pi / butterfly_size)
        twiddles = [omega ** k for k in range(half_size)]
        
        # Apply butterflies
        for k in range(0, N, butterfly_size):
            for j in range(half_size):
                u = x[k + j]
                v = x[k + j + half_size] * twiddles[j]
                
                x[k + j] = u + v
                x[k + j + half_size] = u - v
    
    return x
```

The iterative implementation offers several advantages:
1. **Efficiency**: Avoids function call overhead and potential stack overflow
2. **Memory locality**: Better cache performance
3. **In-place computation**: Can be implemented to modify the array directly

### Practical Considerations

When implementing or using FFT in practice:

1. **Input length requirements**: 
   - Most efficient when N is a power of 2
   - Use zero-padding or specialized algorithms for other lengths
   
2. **Numerical stability**:
   - Be aware of potential roundoff errors, especially with large N
   - Consider scaled or normalized variants for better precision
   
3. **Performance optimizations**:
   - Memory access patterns can significantly affect performance
   - Modern implementations use SIMD instructions, multi-threading, and GPU acceleration
   - Libraries like FFTW, MKL, and cuFFT provide highly optimized implementations

4. **Applications**:
   - Signal processing: filtering, spectral analysis
   - Audio processing: music analysis, speech recognition
   - Image processing: filtering, feature extraction
   - Numerical methods: fast convolution, solving PDEs
   - Scientific computing: data analysis, simulations

### Advanced FFT Variants

- **Real-valued FFT**: Optimized for real-valued inputs (nearly 2× faster)
- **Pruned FFT**: Optimized when many inputs or outputs are zero
- **Split-radix FFT**: Combines radix-2 and radix-4 butterflies for better efficiency
- **Bluestein's algorithm**: Handles non-power-of-2 sizes efficiently
- **Parallel FFT algorithms**: Designed for multi-core and distributed systems
## References

1. Newman, M. (2013). Computational Physics. CreateSpace Independent Publishing Platform.
2. Feigelson, E. D., & Babu, G. J. (2012). Modern Statistical Methods for Astronomy. Cambridge University Press.
