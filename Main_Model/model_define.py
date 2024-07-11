import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

if torch.cuda.is_available():
    print("\n \n Using CUDA device:", torch.cuda.get_device_name(torch.cuda.current_device()), "\n \n")
    device = 'cuda'
else:
    device = 'cpu'
    raise Exception("CUDA is not available!!")

#将接下来创建的变量类型均为Float
torch.torch.set_default_dtype(torch.float32)

#%%
# ------------------Optical parameters---------------------
lmbda = 6e-4  # 600um, wavelength of coherent light
L = 0.213         # 30cm, the side length of phase masks
mesh_num = int(1.41*L/lmbda) # 为了保证在光锥内, mesh_num至少是L的平方根的sqrt(2)倍
scale_coe = 2
pixel_num = int(mesh_num/scale_coe)     # number of pixels on each axis, related to the fabrication resolution of phase masks
z = 0.03      # 3cm,  the propagation distance in free space
j = torch.tensor([0+1j], dtype=torch.complex128, device=device)

# 长度太短或者mesh_num太大, 会使得propagation_layer中的 H 矩阵中的值出现nan, 从而导致整个网络的输出为nan!! 可能是因为此时空间频率(波数)超过了光的最大空间波数(处于光锥之下了), 光无法传播!

print("L = ", L, "z = ", z, "wavelength = ", lmbda, "mesh_num = ", mesh_num, "pixel_num = ", pixel_num)

#%%
# ---------------------spliting the OUTPUT area--------------------!!!!!!
square_size = round(mesh_num / 10)

# Calculate the offset of the center of the target squares in the output canvas
middle_coor = mesh_num // 2

# 十个目标点的坐标
start_x = np.array([-2.5, -2.5, -2.5, -0.5, -0.5, -0.5, -0.5, 1.5, 1.5, 1.5])*square_size + middle_coor
start_y = np.array([-2.5, -0.5, 1.5, -3.5, -1.5, 0.5, 2.5, -2.5, -0.5, 1.5])*square_size + middle_coor

start_x = start_x.astype(int)
start_y = start_y.astype(int)

# 计算一个输出点像素区域的总光强
def count_pixel(img, start_x, start_y, width, height):
    # Slicing the batch of images and summing over the desired region for each image
    return torch.sum(img[:, start_x:start_x+width, start_y:start_y+height], dim=(1,2))

def norm(x):
    return torch.sqrt(torch.dot(x, x)) # 二范数


#%%
#----------------Define the network------------------!!!!
class propagation_layer(nn.Module):
    def __init__(self, L, lmbda, z):
        super(propagation_layer, self).__init__()
        self.L = L # the area to be illuminated
        self.lmbda = lmbda # wavelength
        self.z = z # propagation distance in free space
    
    def forward(self, u, test=False):
        if test: # 画一下光场分布图, 方便调试
            print("propagation")
            # plt.figure()
            # plt.imshow((torch.abs(u)**2).detach().cpu().numpy()[0,:,:])
            # plt.show()

        M, N = u.shape[-2:] # [-2:] means the last two dimensions! (pattern 数量, 长, 宽)
        dx = self.L / M # sample interval, 划分的像素大小!
        k = 2 * np.pi / self.lmbda # angular wavenumber

        fx = torch.linspace(-1 / (2 * dx), 1 / (2 * dx) - 1 / self.L, M, device=u.device, dtype=u.dtype).to(device) 
        # 定义了空间频率坐标系, 上下限是离散情况下可以分辨的最大空间频率! 单位是 1/m (如果L的单位是m的话)
        # 数学上可以得到, 频域的分辨率和原始图像的实空间分辨率是一样的, 所以 也取了M个点
        # 减去 1/L 可能是为了让 fx 中全是整数, 并且可以存在 0 频率, 不然全是恶心的小数

        FX, FY = torch.meshgrid(fx, fx, indexing='ij') # x and y coordinate of every point in the canvas
        
        # Transfer function of free space
        H = torch.exp(torch.tensor(1j) * k * self.z * torch.sqrt(1-(lmbda*FX)**2 - (lmbda*FY)**2)).to(device)
        # H = torch.fft.fftshift(H, dim=(-2,-1))
        U1 = torch.fft.fftshift(torch.fft.fft2(u, dim=(-2,-1))).to(device)
        # print('U1.shape = ', U1.shape, 'H.shape = ', H.shape)
        U2 = H * U1
        u2 = torch.fft.ifft2(torch.fft.ifftshift(U2), dim=(-2,-1))

        return u2

class modulation_layer(nn.Module):
    def __init__(self, num=0):
        super(modulation_layer, self).__init__()
        # INITIALIZE phase values to zeros, but they will be updated during training
        self.phase_values = nn.Parameter(torch.zeros((pixel_num, pixel_num),dtype=torch.float32).to(device))
        # nn.init.kaiming_uniform_(self.phase_values, nonlinearity='relu')
        nn.init.uniform_(self.phase_values, a = 0, b = 2) # uniform random values between 0 and 2PI
        self.num = num

        if self.num == 0:
            raise Exception("Invalid modulation layer number!")

        self.inverse_mat = torch.zeros((pixel_num, pixel_num)).to(device)
        for i in range(pixel_num):
            self.inverse_mat[i,-(i+1)] = 1
            
        self.upsamp_mat_left = torch.zeros((mesh_num, pixel_num)).to(device)
        self.upsamp_mat_right = torch.zeros((pixel_num, mesh_num)).to(device)
        for i in range(mesh_num):
            self.upsamp_mat_left[i, i//scale_coe] = 1
            self.upsamp_mat_right[i//scale_coe, i] = 1

    def forward(self, u, test=False, inverse=False):

        if test: # 画一下光场分布图, 方便调试
            print("modulation", self.num)
            plt.figure()
            plt.imshow((torch.abs(u)**2).detach().cpu().numpy()[0,:,:])
            plt.show()

        phase_values_temp = torch.matmul(self.phase_values, self.inverse_mat) if inverse else self.phase_values # 反向处理的时候要把调制层乘上这个inverse_mat镜面转一下, 不然就不符合物理情况

        # Create the complex modulation matrix
        modulation_matrix = torch.exp(1j * 2 * np.pi * torch.matmul(torch.matmul(self.upsamp_mat_left, phase_values_temp),self.upsamp_mat_right)) # torch.complex64

        # Modulate the input tensor
        modulated_tensor = u * modulation_matrix

        return modulated_tensor

class imaging_layer(nn.Module):
    def __init__(self):
        super(imaging_layer, self).__init__()
        
    def forward(self, u, test=False):
        # Calculate the intensity
        intensity = torch.abs(u)**2

        if test: # 画一下光场分布图, 方便调试
            print("image")
            plt.figure()
            plt.imshow(intensity.detach().cpu().numpy()[0,:,:])
            plt.show()

        # 这里使用 u.size(0) 而不直接使用 batch_size 是因为 batch_size 不一定能被数据集的长度整除, 所以最后一个 batch 的大小可能会小于 batch_size, 直接写 batch_size 可能会报错!!!
        output_values_temp = torch.zeros(u.size(0), 11, device=u.device, dtype=intensity.dtype)  # (Batch_size, 10) tensor
        value_output = torch.zeros(u.size(0), 11, device=u.device, dtype=intensity.dtype)  # (Batch_size, 11) tensor
        for i in range(10):
            output_values_temp[:, i] = count_pixel(intensity, start_x[i], start_y[i], square_size, square_size)

        surrounding_intensity = (torch.sum(intensity, dim=(1,2)) - torch.sum(output_values_temp, dim=1)) # sum of the intensity of the surrounding area

        output_values_temp[:, 10] = surrounding_intensity

        for i in range(u.size(0)):
            value_output[i, :] = output_values_temp[i, :] / norm(output_values_temp[i, :]) # 归一化! 为了对应上label中所有位置的元素和为 1 !
        return value_output


class OpticalNetwork(nn.Module):
    def __init__(self, L, lmbda, z):
        super(OpticalNetwork, self).__init__()
        # layers = []
        # for _ in range(5):
        #     layers.append(propagation_layer(L, lmbda, z, test))
        #     layers.append(modulation_layer(M, M, test))
        
        # self.optical_layers = nn.Sequential(*layers) # 星号表示解包, 将layers中的元素一个个提取出来作为参数传入Sequential

        self.propa = propagation_layer(L, lmbda, z)
        self.modu_1 = modulation_layer(num=1)
        self.modu_2 = modulation_layer(num=2)
        self.modu_3 = modulation_layer(num=3)
        self.modu_4 = modulation_layer(num=4)
        self.modu_5 = modulation_layer(num=5)

        self.imaging = imaging_layer()
        
    # def forward(self, x):
    #     x = self.optical_layers(x)
    #     x = self.imaging(x)
    #     return x

#------------------------------------------------------
    def forward(self, x, test=False, inverse=False):
        if inverse:
            x = self.propa(x, test)
            x = self.modu_5(x, test, inverse)
            x = self.propa(x, test)
            x = self.modu_4(x, test, inverse)
            x = self.propa(x, test)
            x = self.modu_3(x, test, inverse)
            x = self.propa(x, test)
            x = self.modu_2(x, test, inverse)
            x = self.propa(x, test)
            x = self.modu_1(x, test, inverse)
            x = self.propa(x, test)
            x = self.imaging(x, test)
        else:
            # x = self.optical_layers(x)
            x = self.propa(x, test)
            x = self.modu_1(x, test, inverse)
            x = self.propa(x, test)
            x = self.modu_2(x, test, inverse)
            x = self.propa(x, test)
            x = self.modu_3(x, test, inverse)
            x = self.propa(x, test)
            x = self.modu_4(x, test, inverse)
            x = self.propa(x, test)
            x = self.modu_5(x, test, inverse)
            x = self.propa(x, test)
            x = self.imaging(x, test)
        return x
#------------------------------------------------------
