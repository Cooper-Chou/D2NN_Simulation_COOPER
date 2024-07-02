import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using {device} device")

#%%
# ------------------Optical parameters---------------------
M = 250     # sampling count on each axis
N = M       # for preprocessing, N = M
lmbda = 0.5e-6  # wavelength of coherent light
z = 100         # the propagation distance in free space
j = torch.tensor([0+1j], dtype=torch.complex128, device=device)
L = 0.2         # the area of simulated region


#%%
# ---------------------spliting the OUTPUT area--------------------!!!!!!
square_size = round(M / 20)
canvas_size = M

# define the places for each digit
canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

positions = np.array([[0, 1], [0, 4], [0, 7], [3, 0.65], [3, 2.65+0.5], [3, 4.65+1], [3, 6.65+1.5], [6, 1], [6, 4], [6, 7]])

# Calculate the offset of the center of the target squares in the output canvas
offset = (canvas_size - 8 * square_size) / 2

# 十个目标点的坐标
start_col = np.zeros(positions.shape[0], dtype=int)
start_row = np.zeros(positions.shape[0], dtype=int)

for i in range(10):
    row, col = positions[i]
    start_row[i] = round(offset + row * square_size)
    start_col[i] = round(offset + col * square_size)

print(start_row, start_col)

# 计算 **一个** 输出点像素区域的总光强
def count_pixel(img, start_x, start_y, width, height):
    # Slicing the batch of images and summing over the desired region for each image
    res = torch.sum(img[:, start_x:start_x+width, start_y:start_y+height], dim=(1,2))
    return res

# 二范数
def norm(x):
    return torch.sqrt(torch.dot(x, x)) 


#%%
#----------------Define the network------------------!!!!
class propagation_layer(nn.Module):
    def __init__(self, L, lmbda, z):
        super(propagation_layer, self).__init__()
        self.L = L # the area to be illuminated
        self.lmbda = lmbda # wavelength
        self.z = z # propagation distance in free space
    
    def forward(self, u1):
        M, N = u1.shape[-2:]
        dx = self.L / M # sample interval (?)
        k = 2 * np.pi / self.lmbda # angular wavenumber

        fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / self.L, M, device=u1.device, dtype=u1.dtype).to(device) # 暂时还没整明白
        FX, FY = torch.meshgrid(fx, fx, indexing='ij') # x and y coordinate of every point in the canvas
        
        # Transfer function of free space
        H = torch.exp(torch.tensor(-1j) * np.pi * self.lmbda * self.z * (FX**2 + FY**2)).to(device) * torch.exp(torch.tensor(1j) * k * self.z).to(device)
        H = torch.fft.fftshift(H, dim=(-2,-1))
        U1 = torch.fft.fft2(torch.fft.fftshift(u1, dim=(-2,-1))).to(device)
        U2 = H * U1
        u2 = torch.fft.ifftshift(torch.fft.ifft2(U2), dim=(-2,-1))
        return u2

class modulation_layer(nn.Module):
    def __init__(self, M, N):
        super(modulation_layer, self).__init__()

        # INITIALIZE phase values to zeros, but they will be updated during training
        self.phase_values = nn.Parameter(torch.zeros((M, N)))
        # nn.init.kaiming_uniform_(self.phase_values, nonlinearity='relu')
        nn.init.uniform_(self.phase_values, a = 0, b = 2) # uniform random values between 0 and 2PI
            
    def forward(self, input_tensor):
        # Create the complex modulation matrix
        modulation_matrix = torch.exp(1j * 2 * np.pi * self.phase_values)
        # Modulate the input tensor
        # print(input_tensor.shape)
        modulated_tensor = input_tensor * modulation_matrix
        # print(modulated_tensor.shape)
        return modulated_tensor

class imaging_layer(nn.Module):
    def __init__(self):
        super(imaging_layer, self).__init__()
        
    def forward(self, u):
        # Calculate the intensity
        intensity = torch.abs(u)**2
        values = torch.zeros(u.size(0), 10, device=u.device, dtype=intensity.dtype)  # Batch x 10 tensor
        value_output = torch.zeros(u.size(0), 10, device=u.device, dtype=intensity.dtype)  
        for i in range(10):
            values[:, i] = count_pixel(intensity, start_row[i], start_col[i], square_size, square_size)

        for i in range(u.size(0)):
            values_temp = values[i, :] / norm(values[i, :]) # 归一化!
            # to avoid in-place operation
            value_output[i, :] = values_temp
        
        surrounding_intensity = torch.sum(intensity, dim=(1,2)) - torch.sum(values, dim=1) # sum of the intensity of the surrounding area
        return value_output, surrounding_intensity


class OpticalNetwork(nn.Module):
    def __init__(self, M, L, lmbda, z):
        super(OpticalNetwork, self).__init__()
        
        # 5 Propagation and Modulation layers interleaved
        layers = []
        for _ in range(5):
            layers.append(propagation_layer(L, lmbda, z))
            layers.append(modulation_layer(M, M))
        
        self.optical_layers = nn.Sequential(*layers) # 星号表示解包, 将layers中的元素一个个提取出来作为参数传入Sequential
        self.imaging = imaging_layer()
        
    def forward(self, x):
        x = self.optical_layers(x)
        x, sur_int = self.imaging(x)
        return x, sur_int
