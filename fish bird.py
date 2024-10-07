from matplotlib import pyplot as plt
import imageio.v2 as imageio
import numpy as np

im = imageio.imread("jiya.jpg")
im1 = imageio.imread("khushi.jpg")
# Convert image to grayscale
gray = np.mean(im, axis=2).astype(np.uint8)
gray1 = np.mean(im1, axis=2).astype(np.uint8)

# Apply threshold
threshold = 127
input_ = (gray > threshold) * 255
plt.imshow(input_, cmap='gray')
plt.title("Thresholded input Image")
plt.show()
threshold = 127
output_ = (gray1 > threshold) * 1
plt.imshow(output_, cmap='gray')
plt.title("Thresholded output Image")
plt.show()

x0 = np.linspace(-255.5, 255.5, 512)
X, Y = np.meshgrid(x0, x0)
S = np.zeros((512, 512))
h = X**2 + Y**2

for i in range(512):
    for j in range(512):
        if h[i][j] <= 128**2:
            S[i][j] = 1
plt.imshow(S, cmap='gray')
plt.title('support function')
plt.show()
th = np.random.rand(512, 512)
print(th.dtype)
g = np.exp(1j * th)
g0 = S*g
plt.imshow(g0.real, cmap='gray')
plt.title('phase mask')
plt.show()
for i in range(100):
    G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g0)))
    G1 = np.sqrt(np.abs(output_))*np.exp(1j * np.angle(G))
    g_ = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G1)))
    if (i % 20) == 0:
        plt.imshow(np.abs(G)**2, cmap="gray")
        cbar = plt.colorbar()
        cbar.set_label("values")
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
        plt.imshow(S*np.angle(g_), cmap="gray")
        cbar = plt.colorbar()
        cbar.set_label("values")
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

    g0 = np.sqrt(np.abs(input_))*np.exp(1j * np.angle(g_))
