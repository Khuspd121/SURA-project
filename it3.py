import numpy as np
import matplotlib.pyplot as plt
import imageio

th = np.random.rand(512, 512)
g = np.exp(1j * th)
x0 = np.linspace(-255.5, 255.5, 512)
# print(x0)
X, Y = np.meshgrid(x0, x0)
# print(X)
h = X**2 + Y**2
# print(h)
S = np.zeros((512, 512))
for i in range(512):
    for j in range(512):
        if h[i][j] <= 128**2:
            S[i][j] = 1
plt.imshow(S,cmap='gray')
plt.title('support function')
plt.show()

g0 = S * g
plt.imshow(g0.real,cmap='gray')
plt.title('phase mask')
plt.show()

im = imageio.imread("star.jpg")

# Convert image to grayscale
gray = np.mean(im, axis=2).astype(np.uint8)

# Apply threshold
threshold = 127
input = (gray > threshold) * 255

plt.imshow(input, cmap='gray')
plt.title("Thresholded Image")
plt.show()

for i in range(100):
    nf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g0)))
    x = np.sqrt(input)
    G = x * np.exp(1j * np.angle(nf))
    g1 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G)))
    if i % 10 == 0:
        plt.imshow(np.abs(G), cmap='gray')
        plt.title(f"Iteration {i}")
        plt.show()
        plt.imshow(np.abs(S*np.angle(g1)), cmap="gray")
        plt.show()
    g0 = S * np.exp(1j * np.angle(g1))