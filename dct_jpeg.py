import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct

# Create the 2D DCT basis functions (8x8)
def dct2d_basis(n=8):
    basis_images = []
    for u in range(n):
        for v in range(n):
            basis = np.outer(
                np.cos(np.pi * (np.arange(n) + 0.5) * u / n),
                np.cos(np.pi * (np.arange(n) + 0.5) * v / n)
            )
            basis_images.append(basis)
    return basis_images

# Create and plot the DCT basis functions
basis_images = dct2d_basis()
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(basis_images[i], cmap='gray')
    ax.axis('off')
fig.suptitle("8x8 DCT Basis Functions", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()
