from matplotlib import pyplot as plt
import imageio.v2 as imageio
from scipy.fft import fft2, ifft2
import numpy as np

im1 = imageio.imread("khushi.jpg")
im2 = imageio.imread("as.jpg")
i1 = (im1[:, :, 0] + im1[:, :, 1] + im1[:, :, 2]) / 3
i2 = (im2[:, :, 0] + im2[:, :, 1] + im2[:, :, 2]) / 3

A1 = fft2(i1)
A2 = fft2(i2)
amp1 = np.abs(A1)
phase1 = np.angle(A1)
amp2 = np.abs(A2)
phase2 = np.angle(A2)
C1 = (amp1 + amp2) * np.exp(1j * phase2)
C2 = amp2 * np.exp(1j * phase1)
D1 = ifft2(C1)
D2 = ifft2(C2)

plt.imshow(np.abs(D1))  # Use 'gray' for grayscale images
# plt.imshow(D2_norm, cmap='gray')
plt.title("phase of as")
plt.show()
plt.imshow(np.abs(D2), cmap='gray')  # Use 'gray' for grayscale images
# plt.imshow(D2_norm, cmap='gray')
plt.title("phase of khushi")
plt.show()
