from matplotlib import pyplot as plt
import imageio.v2 as imageio
import autograd.numpy as anp
import numpy as np
from autograd import grad
from scipy.ndimage import gaussian_filter
from PIL import Image
x0 = np.linspace(-600, 600, 1200)
y0 = np.linspace(-960, 960, 1920)
X, Y = np.meshgrid(x0, x0)
S = np.zeros((512, 512))
h = X ** 2 + Y ** 2
for i in range(512):
    for j in range(512):
        if h[i][j] <= 250 ** 2:
            S[i][j] = 1
th = np.random.rand(512, 512)
g = np.exp(1j * th)
g0_ = S * g
def create_spiral(n_points=512, n_turns=10):
    t = np.linspace(0, 2 * np.pi * n_turns, n_points)
    x = t * np.cos(t)
    y = t * np.sin(t)
    return x, y

n_points = 25
x, y = create_spiral(n_points)
signal = np.zeros((512, 512))
for i in range(n_points):
    signal[int(x[i] + 256), int(y[i] + 256)] = 1

plt.imshow(signal,cmap='gray')
plt.show()
input = signal
def IFTA(g, N,inp):
    g0 = g
    for i in range(N):
        G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g0)))
        G1 = np.sqrt(np.abs(inp)) * np.exp(1j * np.angle(G))
        g_ = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G1)))
        # quantized_phase = np.where(np.angle(g_) > (np.pi / 2), np.pi, 0)
        # quantized_phase = quantize_phase(np.angle(g_))
        # g0 = S * (1 * np.exp(1j * quantized_phase))
        g0 = S * (1 * np.exp(1j * np.angle(g_)))
    return np.angle(g_)
    # return quantized_phase
initial_phase = IFTA(g0_, 100,input)
field1 = np.exp(1j * initial_phase)
output_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field1)))
plt.imshow(np.abs(output_image), cmap="gray")
plt.title("Image before optimization")
plt.show()
def normalize_phase(phase_pattern):
    normalized_phase = (phase_pattern - np.min(phase_pattern)) / (np.max(phase_pattern) - np.min(phase_pattern))
    normalized_phase = normalized_phase*2*np.pi
    return normalized_phase

def quantize_phase(phase_pattern):
    normalized_phase = normalize_phase(phase_pattern)
    binary_phase = np.where(normalized_phase > (np.pi/2), np.pi, 0)
    return binary_phase

quantized_phase =quantize_phase(initial_phase)

plt.imshow(np.abs(quantized_phase), cmap="gray")
plt.title(f"phase mask")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
field1 = np.exp(1j * quantized_phase)
output_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field1)))
plt.imshow(np.abs(output_image), cmap="gray")
plt.title(f"image generated in 8 bit")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
# Normalizing phase values to 8-bit range (0 to 255)
normalized_phase = ((initial_phase + np.pi) / (2 * np.pi)) * 255
rounded_phase = np.round(normalized_phase).astype(np.uint8)
# rounded_phase = np.round(initial_phase)


slm_matrix = np.zeros((1200, 1920), dtype=np.uint8)
start_x = (1200 - 512) / 2
start_y = (1920 - 512) / 2
print(field1.shape)
slm_matrix[start_x:start_x+512, start_y:start_y+512] = np.round(initial_phase)
plt.imshow(slm_matrix,cmap='gray')
plt.show()

# Save the SLM matrix as an image
imageio.imwrite("trial1.png", slm_matrix)


