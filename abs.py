from matplotlib import pyplot as plt
import imageio.v2 as imageio
import autograd.numpy as anp
import numpy as np
from autograd import grad
from scipy.ndimage import gaussian_filter
from PIL import Image


# Load the image
im = imageio.imread(r"A.jpg")
gray = np.mean(im,axis=2)
threshold = 127
input = (gray > threshold) * 255
threshold = 127
input = (im > threshold) * 256
plt.imshow(input, cmap='gray')
plt.title("Target Image")
plt.show()

# Initial setup
# block_size = 25
# x_start, y_start = 200, 200

# Meshgrid
x0 = np.linspace(-255.5, 255.5, 512)
X, Y = np.meshgrid(x0, x0)
S = np.zeros((512, 512))
h = X ** 2 + Y ** 2
for i in range(512):
    for j in range(512):
        if h[i][j] <= 128 ** 2:
            S[i][j] = 1
th = np.random.rand(512, 512)
g = np.exp(1j * th)
g0_ = S * g

# Loss function with regularization
def loss(phase):
    phase = phase.reshape(input.shape)
    g1 = anp.exp(1j * phase)
    g1ft = anp.fft.fftshift(anp.fft.fft2(anp.fft.ifftshift(g1)))
    return anp.sum(anp.abs(anp.abs(g1ft) ** 2 - anp.abs(input)) ** 2)**2

# IFTA function
def IFTA(g, N=100):
    g0 = g
    for i in range(N):
        G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g0)))
        G1 = np.sqrt(np.abs(input)) * np.exp(1j * np.angle(G))
        g_ = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G1)))
        g0 = S * (1 * np.exp(1j * np.angle(g_)))
    return np.angle(g_)

# Optimization function with Adam optimizer
def optimization(f, inphase, num=100, step=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = anp.zeros_like(inphase)
    v = anp.zeros_like(inphase)
    t = 0
    phase = inphase
    loss_grad = grad(f)
    for k in range(num):
        t += 1
        s = loss_grad(phase)
        m = beta1 * m + (1 - beta1) * s
        v = beta2 * v + (1 - beta2) * (s ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        phase = phase - step * m_hat / (anp.sqrt(v_hat) + epsilon)
        gp = S * np.exp(1j * phase)
        Gx = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gp)))
        # if k % 100 == 0:
        #     plt.imshow(np.abs(Gx), cmap='gray')
        #     plt.show()
        #     print(f"Iteration {k}, loss: {f(phase)}")
    return phase

# Initial phase retrieval
initial_phase = IFTA(g0_, 100)
field1 = np.exp(1j * initial_phase)
output_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field1)))
plt.imshow(np.abs(output_image), cmap="gray")
plt.title("Image before optimization")
plt.show()
plt.imshow(np.angle(field1),cmap='gray')
plt.show()
# Optimization
optimized_phase = optimization(loss, initial_phase)
field = np.exp(1j * optimized_phase)*S
output_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
plt.imshow(np.abs(output_image), cmap="gray")
plt.title("Image after optimization")
plt.show()
plt.imshow(optimized_phase,cmap='gray')
plt.show()
output_image_filtered = gaussian_filter(np.abs(output_image), sigma=1)

slm_matrix = np.zeros((1200, 1920))
start_x = (1200 - 512) // 2
start_y = (1920 - 512) // 2

slm_matrix[start_x:start_x+512, start_y:start_y+512] = optimized_phase+2*np.pi
plt.imshow(slm_matrix,cmap='gray')
plt.show()

plt.imshow(output_image_filtered, cmap="gray")
plt.title("Final image after optimization and filtering")
plt.show()
def create_new_image(size, luminance):
    width, height = size
    black_frame = int(luminance) * np.ones((width, height, 1), dtype=np.uint8)
    return black_frame

def save_image(image, output_path):
    imageio.imwrite(output_path, image)

img = create_new_image((256, 256), 125)
save_image(slm_matrix, "test.png")
img1 = imageio.imread("test.png")

slm = np.zeros((1200, 1920))
slm[start_x:start_x+512, start_y:start_y+512] = S
field2 = np.exp(1j * slm_matrix)*slm
output_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field2)))
plt.imshow(np.abs(output_image), cmap="gray")
plt.show()