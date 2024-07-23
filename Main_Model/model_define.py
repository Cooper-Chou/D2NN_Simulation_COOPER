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
frequency = 0.5e12
lmbda = 3e8/frequency  # 600um, wavelength of coherent light
L = 0.08 # 0.213         # the side length of phase masks
mesh_num = int(1.41*L/lmbda) # 为了保证在光锥内, mesh_num最多是L的平方根的sqrt(2)倍
# 长度太短或者mesh_num太大, 会使得propagation_layer中的 H 矩阵中的值出现nan, 从而导致整个网络的输出为nan!! 可能是因为此时空间频率(波数)超过了光的最大空间波数(处于光锥之下了), 光无法传播!
para_scale_coe = 1
z = 0.03      # 3cm,  the propagation distance in free space
j = torch.tensor([0+1j], dtype=torch.complex128, device=device)

surr_inten_weight = 0.8
surr_inten_num = 100
norm_type = 'MAX' # 'L2' # L1 # 
zooming_coefficient = 0.5

output_for_loss_type = "Clps" # "Full" # 
# FullPlane and Collapsed

loss_type = "Cross" # "MSE" # 
# CrossEntropy and MeanSquaredError

square_size = round(mesh_num / 20)

start_x_pctg = np.array([-2.5, -2.5, -2.5, -0.5, -0.5, -0.5, -0.5, 1.5, 1.5, 1.5])
start_y_pctg = np.array([-2.5, -0.5, 1.5, -3.5, -1.5, 0.5, 2.5, -2.5, -0.5, 1.5])
#%%

print("L = ", L, "z = ", z, "wavelength = ", lmbda, "mesh_num = ", mesh_num, "scale_coe = ", para_scale_coe, "zooming_coefficient = ", zooming_coefficient,"surrounding_weight = ", surr_inten_weight, "surrounding_num = ", surr_inten_num, "square_size = ", square_size, "\n", "norm_type = ", norm_type, "loss_type = ", loss_type, "output_for_loss_type = ", output_for_loss_type)

# 计算一个输出点像素区域的总光强
def count_pixel_and_wipe(intensity, start_x, start_y, end_x, end_y, surrounding_intensity=None):
    if surrounding_intensity is not None:
        surrounding_intensity[:, start_x:end_x, start_y:end_y] = 0
    # Slicing the batch of images and summing over the desired region for each image
    return torch.sum(intensity[:, start_x:end_x, start_y:end_y], dim=(1,2))

def norm(x, type=None):
    output = None
    dim = x.dim()
    if type == 'L2':
        if dim == 2:
            output = torch.sqrt(torch.sum(x*x, dim=1)) # 二范数
        elif dim == 3:
            output = torch.sqrt(torch.sum(x*x, dim=(1,2))) # 二范数
    elif type == 'L1':
        if dim == 2:
            output = torch.sum(torch.abs(x), dim=1) # 一范数
        elif dim == 3:
            output = torch.sum(torch.abs(x), dim=(1,2)) # 一范数
    elif type == 'MAX':
        if dim == 2:
            output = torch.max(x,dim=1).values
        elif dim == 3:
            output = torch.max(torch.max(x,dim=2).values,dim=1).values
    else:
        raise Exception("Invalid norm type!")
    return output



#%%
#----------------Define the network------------------!!!!
class propagation_layer(nn.Module):
    def __init__(self, L, lmbda, z):
        super(propagation_layer, self).__init__()
        self.L = L # the area to be illuminated
        self.lmbda = lmbda # wavelength
        self.z = z # propagation distance in free space
    
    def forward(self, u, test=False):
        # if test: # 画一下光场分布图, 方便调试
        #     print("propagation")
        #     # plt.figure()
        #     # plt.imshow((torch.abs(u)**2).detach().cpu().numpy()[0,:,:])
        #     # plt.show()

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
    def __init__(self, mesh_num, pixel_num, order=0):
        super(modulation_layer, self).__init__()
        # INITIALIZE phase values to zeros, but they will be updated during training
        self.phase_values = nn.Parameter(torch.zeros((pixel_num, pixel_num),dtype=torch.float32).to(device))
        # nn.init.kaiming_uniform_(self.phase_values, nonlinearity='relu')
        nn.init.uniform_(self.phase_values, a = 0, b = 2) # uniform random values between 0 and 2PI
        self.order = order
        self.mesh_num = mesh_num
        self.pixel_num = pixel_num

        if self.order == 0:
            raise Exception("Invalid modulation layer number!")

        self.inverse_mat = torch.zeros((self.pixel_num, self.pixel_num)).to(device)
        for i in range(self.pixel_num):
            self.inverse_mat[i,-(i+1)] = 1
            
        self.upsamp_mat_left = torch.zeros((self.mesh_num, self.pixel_num)).to(device)
        self.upsamp_mat_right = torch.zeros((self.pixel_num, self.mesh_num)).to(device)
        for i in range(self.mesh_num):
            self.upsamp_mat_left[i, i//para_scale_coe] = 1
            self.upsamp_mat_right[i//para_scale_coe, i] = 1

    def forward(self, u, test=False, inverse=False):

        if test: # 画一下光场分布图, 方便调试
            # print("modulation", self.num)
            plt.subplot(2,3,self.order+1)
            plt.imshow((torch.abs(u)**2).detach().cpu().numpy()[0,:,:])

        phase_values_temp = torch.matmul(torch.sigmoid(self.phase_values)*2, self.inverse_mat) if inverse else torch.sigmoid(self.phase_values)*2
        # 反向处理的时候要把调制层乘上这个inverse_mat镜面转一下, 不然就不符合物理情况; 同时把相位值约束到0~2PI之间

        # Create the complex modulation matrix
        modulation_matrix = torch.exp(1j * np.pi * torch.matmul(torch.matmul(self.upsamp_mat_left, phase_values_temp),self.upsamp_mat_right)) # torch.complex64

        # Modulate the input tensor
        modulated_tensor = u * modulation_matrix

        return modulated_tensor

class imaging_layer(nn.Module):
    def __init__(self, output_for_loss_type, square_size, mesh_num):
        super(imaging_layer, self).__init__()
        self.output_for_loss_type = output_for_loss_type
        # Calculate the offset of the center of the target squares in the output canvas
        middle_coor = mesh_num // 2
        # 十个目标点的坐标
        start_x = start_x_pctg*square_size + middle_coor
        start_y = start_y_pctg*square_size + middle_coor

        self.start_x = start_x.astype(int)
        self.start_y = start_y.astype(int)
        self.square_size = square_size
        
    def forward(self, u, test=False):
        # Calculate the intensity
        intensity = torch.abs(u)**2

        if test: # 画一下光场分布图, 方便调试
            # print("image")
            plt.figure()
            plt.imshow(intensity.detach().cpu().numpy()[0,:,:])
            plt.show()

        if self.output_for_loss_type == "Clps":
            surrounding_intensity = intensity.detach().clone()
            # 这里使用 u.size(0) 而不直接使用 batch_size 是因为 batch_size 不一定能被数据集的长度整除, 所以最后一个 batch 的大小可能会小于 batch_size, 直接写 batch_size 可能会报错!!!
            output_temp = torch.zeros((u.size(0), 10 + surr_inten_num), device=u.device, dtype=intensity.dtype)  # (Batch_size, 10) tensor

            for i in range(10):
                output_temp[:, i] = count_pixel_and_wipe(intensity, self.start_x[i], self.start_y[i], self.start_x[i]+self.square_size, self.start_y[i]+self.square_size, surrounding_intensity=surrounding_intensity) # 在这里计算每个目标点的光强的同时会把 surronding_intensity 的对应区域变为零

            surr_inten_sums = torch.zeros((u.size(0), surr_inten_num), device=u.device, dtype=intensity.dtype)
            for i in range(surr_inten_num):
                surr_inten_sums[:, i] = count_pixel_and_wipe(surrounding_intensity, 0, round(i*u.size(2)/surr_inten_num), u.size(2)-1, round((i+1)*u.size(2)/surr_inten_num)) # sum of the intensity of the surrounding area

            output_temp[:, 10:10+surr_inten_num] = surr_inten_sums * surr_inten_weight

            output_for_loss = output_temp / norm(output_temp, type=norm_type).reshape(u.size(0),1) # 归一化! 为了对应上label中所有位置的元素和为 1 !

            _, predicted_labels = torch.max(output_for_loss.detach()[:,0:10], 1)

            return output_for_loss, predicted_labels

        elif self.output_for_loss_type == "Full":
            output_temp = intensity.clone()
            for i in range(10):
                output_temp[:,self.start_x[i]:self.start_x[i]+self.square_size, self.start_y[i]:self.start_y[i]+self.square_size] /= surr_inten_weight
            output_temp *= surr_inten_weight
            output_for_loss = output_temp/norm(output_temp, type=norm_type).reshape(u.size(0),1,1)
            predicted_labels_temp = torch.zeros((u.size(0),10), device=u.device, dtype=torch.int32)
            for i in range(10):
                predicted_labels_temp[:,i] = count_pixel_and_wipe(output_for_loss, self.start_x[i], self.start_y[i], self.start_x[i]+self.square_size, self.start_y[i]+self.square_size)
            _, predicted_labels = torch.max(predicted_labels_temp.detach()[:,0:10], 1)

            return output_for_loss.reshape(u.size(0),u.size(1)*u.size(2)), predicted_labels



class OpticalNetwork(nn.Module):
    def __init__(self, L, lmbda, z, square_size, mesh_num, para_scale_coe, output_for_loss_type="Clps"):
        super(OpticalNetwork, self).__init__()
        # layers = []
        # for _ in range(5):
        #     layers.append(propagation_layer(L, lmbda, z, test))
        #     layers.append(modulation_layer(M, M, test))
        
        # self.optical_layers = nn.Sequential(*layers) # 星号表示解包, 将layers中的元素一个个提取出来作为参数传入Sequential
        pixel_num = int(mesh_num/para_scale_coe)     # number of pixels on each axis, related to the fabrication resolution of phase masks


        self.propa = propagation_layer(L, lmbda, z)
        self.modu_1 = modulation_layer(order=1, mesh_num=mesh_num, pixel_num=pixel_num)
        self.modu_2 = modulation_layer(order=2, mesh_num=mesh_num, pixel_num=pixel_num)
        self.modu_3 = modulation_layer(order=3, mesh_num=mesh_num, pixel_num=pixel_num)
        self.modu_4 = modulation_layer(order=4, mesh_num=mesh_num, pixel_num=pixel_num)
        self.modu_5 = modulation_layer(order=5, mesh_num=mesh_num, pixel_num=pixel_num)
        self.modulation_layers = [self.modu_1, self.modu_2, self.modu_3, self.modu_4, self.modu_5] # 不知对不对!!

        self.imaging = imaging_layer(output_for_loss_type, square_size=square_size, mesh_num=mesh_num)
        
    # def forward(self, x):
    #     x = self.optical_layers(x)
    #     x = self.imaging(x)
    #     return x

#------------------------------------------------------
    def forward(self, x, test=False, inverse=False, order=[1,2,3,4,5]):
        if inverse:
            x = self.propa(x, test)
            x = self.modulation_layers[order[4]-1](x, test, inverse)
            x = self.propa(x, test)
            x = self.modulation_layers[order[3]-1](x, test, inverse)
            x = self.propa(x, test)
            x = self.modulation_layers[order[2]-1](x, test, inverse)
            x = self.propa(x, test)
            x = self.modulation_layers[order[1]-1](x, test, inverse)
            x = self.propa(x, test)
            x = self.modulation_layers[order[0]-1](x, test, inverse)
            x = self.propa(x, test)
            output_for_loss, predicted_labels = self.imaging(x, test)
        else:
            # x = self.optical_layers(x)
            x = self.propa(x, test)
            x = self.modulation_layers[order[0]-1](x, test, inverse)
            x = self.propa(x, test)
            x = self.modulation_layers[order[1]-1](x, test, inverse)
            x = self.propa(x, test)
            x = self.modulation_layers[order[2]-1](x, test, inverse)
            x = self.propa(x, test)
            x = self.modulation_layers[order[3]-1](x, test, inverse)
            x = self.propa(x, test)
            x = self.modulation_layers[order[4]-1](x, test, inverse)
            x = self.propa(x, test)
            output_for_loss, predicted_labels = self.imaging(x, test)
        return output_for_loss, predicted_labels
#------------------------------------------------------
