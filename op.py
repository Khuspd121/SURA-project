import autograd.numpy as np
import matplotlib.pyplot as plt
import imageio
from autograd import grad

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
        if(h[i][j] <= 128**2):
            S[i][j] = 1
plt.imshow(S,cmap='grey')
plt.show()

im1 = imageio.imread(r"C:\Users\Win11\Downloads\letterA.jpg")
# gray = np.mean(im1,axis=2)
threshold = 127
# input = (gray > threshold) * 255
input = (im1 > threshold) * 255

input = input.astype(np.uint8)
plt.imshow(input, cmap='gray')
plt.title('Input')
plt.show()

# grid_size = 128
# rect_size = 12
# grid = np.zeros((grid_size, grid_size))

# start_idx = (grid_size - rect_size) // 2
# end_idx = start_idx + rect_size

# grid[start_idx:end_idx, start_idx:end_idx] = 1

# plt.imshow(grid, cmap='gray')
# plt.show()

# th = np.random.rand(128, 128)*2 *np.pi
# g = np.exp(1j * th)

# # Create support function S (circular mask)
# x0 = np.linspace(-64, 64, 128)
# X, Y = np.meshgrid(x0, x0)
# h = X**2 + Y**2
# S = np.zeros((128, 128))
# for i in range(128):
#     for j in range(128):
#         if h[i][j] <= 32**2:
#             S[i][j] = 1

# plt.imshow(S, cmap='gray')
# plt.title('Support Function')
# plt.show()


def getPhase(mat):
    return S*np.exp(1j * mat)

def loss_function(t):
    g0 = getPhase(t)
    nf0 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g0)))
    x0 = np.abs(input)
    return np.mean((np.abs(nf0) - x0) ** 2)

grad_loss = grad(loss_function)
# print(th)
for i in range(100):
    loss = grad_loss(th)
    th = th - 0.001*loss
    gp = S*np.exp(1j*th)
    nf = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(gp)))
    G = np.abs(input)*np.exp(1j*np.angle(nf))
    g_ = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(G)))
    th = np.angle(g_)
    # if i%10 == 0:
    #     plt.imshow(np.abs(nf),cmap='grey')
    #     plt.show()
    #     print(th)




plt.imshow(np.abs(nf), cmap='gray')
plt.show()


