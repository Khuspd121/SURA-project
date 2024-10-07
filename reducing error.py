from matplotlib import pyplot as plt
import imageio.v2 as imageio
import autograd.numpy as anp
import numpy as np
from autograd import grad
from scipy.ndimage import gaussian_filter

im = imageio.imread("A.jpg")
# Convert image to grayscale
gray = gaussian_filter(np.mean(im, axis=2).astype(np.uint8), 1)
# Apply threshold
threshold = 127
input = (gray > threshold) * 1
plt.imshow(input, cmap='gray')
plt.title("target Image")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
# for k in range(512):
#     for l in range(512):
#         if input[k][l]==1:
#             print(k,l)
# meshgrid
x0 = np.linspace(-255.5, 255.5, 512)
X, Y = np.meshgrid(x0, x0)
S = np.zeros((512, 512))
h = X ** 2 + Y ** 2
for i in range(512):
    for j in range(512):
        if h[i][j] <= 128 ** 2:
            S[i][j] = 1
# plt.imshow(S, cmap='gray')
# plt.title('support function')
# plt.show()
th = np.random.rand(512, 512)
g = np.exp(1j * th)
g0_ = S * g
# plt.imshow(g0_.real, cmap='gray')
# plt.title("g0")
# plt.show()


def loss(phase):
    phase = phase.reshape(input.shape)
    g1 = anp.exp(1j * phase)
    g1ft = anp.fft.fftshift(anp.fft.fft2(anp.fft.ifftshift(g1)))
    return anp.sum(anp.abs(anp.abs(g1ft) ** 2 - anp.abs(input)) ** 2)


def IFTA(g, N = 100):
    g0 = g
    for i in range(N):
        G = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g0)))
        G1 = np.sqrt(np.abs(input)) * np.exp(1j * np.angle(G))
        g_ = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G1)))
        # if (i % 50) == 0:
        #     plt.imshow(np.abs(G) ** 2, cmap="gray")
        #     plt.title(f"iteration {i}, intensity distribution")
        #     plt.show()
        #     plt.imshow(S * np.angle(g_), cmap="gray")
        #     plt.title(f"iteration {i}, phase")
        #     plt.show()
        g0 = S * (1 * np.exp(1j * np.angle(g_)))
    return np.angle(g_)


def optimization(f, inphase, num=1000, step=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = anp.zeros_like(inphase)
    v = anp.zeros_like(inphase)
    t = 0
    phase = inphase
    loss_grad = grad(f)  # Define gradient function outside the loop

    for k in range(num):
        t += 1
        s = loss_grad(phase)  # Compute the gradient

        m = beta1 * m + (1 - beta1) * s
        v = beta2 * v + (1 - beta2) * (s ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        # print(snr(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(S * (1 * np.exp(1j * phase)))))))
        phase = phase - step * m_hat / (anp.sqrt(v_hat) + epsilon)
        # if k % 100 == 0:
        #     print(f"Iteration{k}, loss:{f(phase)}")

    return phase

def snr(output_image):
    I_ = np.abs(output_image)**2
    I0 = np.abs(gray)**2
    size = 25
    x_start, y_start = 200, 200
    speckle = I_[x_start:x_start+size, y_start:y_start+size]
    og = I0[x_start:x_start+size, y_start:y_start+size]
    snr_1 = np.std(speckle)/np.mean(og)
    return snr_1

initial_phase = IFTA(g0_, 100)
round = np.round(initial_phase)
field1 = np.exp(1j * round)*S
output_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field1)))
plt.imshow(np.abs(output_image) ** 2, cmap="gray")
plt.title("image before optimization")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
plt.imshow(np.abs(round), cmap="gray")
plt.title("rounded phase")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
optimized_phase = optimization(loss, initial_phase,100)
#roundphase = ((optimized_phase+np.pi))
# roundphase=((initial_phase+np.pi))
# field = np.exp(1j * roundphase)*S
# output_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
# out = gaussian_filter(output_image,1)
# plt.imshow(np.abs(out) ** 2, cmap="gray")
# plt.title("final image after optimization")
# cbar = plt.colorbar()
# cbar.set_label("values")
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()
# # plt.imshow(np.abs(S * roundphase), cmap="gray")
# plt.title(f"phase mask")
# cbar = plt.colorbar()
# cbar.set_label("values")
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()

# Normalizing phase values to 8-bit range (0 to 255)
normalized_phase = ((optimized_phase + np.pi) / (2 * np.pi)) * 255
rounded_phase = np.round(normalized_phase).astype(np.uint8)

quantized_phase = np.where(optimized_phase+np.pi < np.pi, 0, np.pi)

field2 = np.exp(1j * quantized_phase)*S
output_image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field2)))
plt.imshow(np.abs(output_image), cmap="gray")
plt.title("image quantized")
cbar = plt.colorbar()
cbar.set_label("values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
# # Creating the SLM matrix of size 1200x1920 and placing the phase values
slm_matrix = np.zeros((1200, 1920), dtype=np.uint8)
start_x = (1200 - 512) // 2
start_y = (1920 - 512) // 2
slm_matrix[start_x:start_x+512, start_y:start_y+512] = rounded_phase

# Save the SLM matrix as an image
imageio.imwrite("SLM_Phase_Image.png", slm_matrix)

# Generate the field using the final phase values directly
# Use the normalized phase in the range [0, 255]
field_final = np.exp(1j * (rounded_phase / 255) * 2 * np.pi) * S
output_image_final = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field_final)))
out = gaussian_filter(output_image_final, 1)
# Show the final output image
plt.imshow(np.abs(out) ** 2, cmap="gray")
plt.title("Final Image after Optimization")
cbar = plt.colorbar()
cbar.set_label("Values")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
# Display the final phase mask
plt.imshow(slm_matrix, cmap='gray')
plt.title("SLM Phase Matrix (0 to 255)")
cbar = plt.colorbar()
cbar.set_label("Phase values (0 to 255)")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
