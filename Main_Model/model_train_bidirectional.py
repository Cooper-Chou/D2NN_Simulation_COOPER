import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import model_define as modef
import os
from datetime import datetime

torch.cuda.empty_cache() # 清空显存!

torch.autograd.set_detect_anomaly(True) # 检测梯度异常


#%%
# -------------------load data----------------------------
class CustomNpyDataset(Dataset):
    def __init__(self, data_dir:str, data_file:str, label_file:str):
        super(CustomNpyDataset, self).__init__()
        self.data_dir = data_dir # 一个字符串!
        self.data_file = data_file # 一个字符串!
        self.label_file = np.load(label_file, mmap_mode='r')

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        # print("__getitem__ called")
        data = np.load(self.data_dir + "/" + self.data_file + "%d.npy"%idx, allow_pickle=True)
        labels = self.label_file[:, idx]
        return torch.tensor(labels).double().to(modef.device), torch.tensor(data).double().to(modef.device)
    
# Hyperparameters
learning_rate = 0.0003
epochs = 4
batch_size = 32

dataset_dir_mnist = "dataset/MNIST_processed_random"
dataset_dir_fashion = "dataset/FASHION_processed_random"

MNIST_train_dataset = CustomNpyDataset(dataset_dir_mnist+"/u0_train", "u0_train_", dataset_dir_mnist+"/label_train_all.npy")
MNIST_train_loader = DataLoader(MNIST_train_dataset, batch_size=batch_size, shuffle=True)

MNIST_test_dataset = CustomNpyDataset(dataset_dir_mnist+"/u0_test", "u0_test_", dataset_dir_mnist+"/label_test_all.npy")
MNIST_test_loader = DataLoader(MNIST_test_dataset, batch_size=batch_size, shuffle=True)

MNIST_validation_dataset = CustomNpyDataset(dataset_dir_mnist+"/u0_validation", "u0_validation_", dataset_dir_mnist+"/label_validation_all.npy")
MNIST_validation_loader = DataLoader(MNIST_validation_dataset, batch_size=batch_size, shuffle=True)
# -----------------------------------------------------------------------------------------------------------------------------
FASHION_train_dataset = CustomNpyDataset(dataset_dir_fashion+"/u0_train", "u0_train_", dataset_dir_fashion+"/label_train_all.npy")
FASHION_train_loader = DataLoader(FASHION_train_dataset, batch_size=batch_size, shuffle=True)

FASHION_test_dataset = CustomNpyDataset(dataset_dir_fashion+"/u0_test", "u0_test_", dataset_dir_fashion+"/label_test_all.npy")
FASHION_test_loader = DataLoader(FASHION_test_dataset, batch_size=batch_size, shuffle=True)

FASHION_validation_dataset = CustomNpyDataset(dataset_dir_fashion+"/u0_validation", "u0_validation_", dataset_dir_fashion+"/label_validation_all.npy")
FASHION_validation_loader = DataLoader(FASHION_validation_dataset, batch_size=batch_size, shuffle=True)
print("successfully loaded!!")


#%%
# Initialize network, loss, and optimizer
model = modef.OpticalNetwork(modef.M, modef.L, modef.lmbda, modef.z).to(modef.device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


print("--------------------------Start Training!!--------------------------")
# Training loop
for epoch in range(epochs):
    print("epoch: ", epoch+1)
    # Training phase
    model.train()
    running_loss_forward = 0.0
    running_loss_inverse = 0.0

    for idx_enum,data_enum in enumerate(zip(FASHION_train_loader, MNIST_train_loader)):
        # zip 可以把两个可迭代对象(这里是 dataloader)一个一个对应起来, 像梳子那样(详情上网查), 然后返回的也是可以迭代的; enumerate 可以在返回可迭代对象的同时返回索引(具体上网查), 反正这两个函数可方便了
        # 在这个 for 循环里, 每次迭代 data_enum 都是一个元组, 元组里面有两个元素, 分别是两个 dataloader 里的一个 batch 的数据

        # -----------------MNIST -> forward-----------------
        batch_labels = data_enum[0][0]
        batch_data = data_enum[0][1]
        optimizer.zero_grad()
        outputs, surrounding_intensity = model(batch_data, inverse=False) # 注意! 这里的surrounding_intensity没有被归一化, 它的值也许很大, 这也可以认为是变相为周围的杂散光场添加了更高的权重(?)
        outputs_for_loss = torch.cat((outputs, surrounding_intensity), dim=1) # 在 outputs 后面加一列周围的杂散光场, 用于计算 loss
        batch_labels_for_loss = torch.cat((batch_labels, torch.zeros(batch_labels.shape[0],1).to(modef.device)), dim=1) # 在 batch_labels 后面加一列 0, 对应于杂散光场的目标值
        loss = loss_function(outputs_for_loss, batch_labels_for_loss)
        loss.backward() # backward() 函数是 tensor 自带的函数, 所有的 lossfunction forward() 以后都返回 tensor 类型的结果! 而且可能返回的 loss 已经是开过 require_grad 和 retain_graph 的了, 后面 optimizer 就会直接根据 loss 的梯度图来迭代
        optimizer.step()
        running_loss_forward += loss.item()
# ------------------------------------------------------------------
        # print("Running Loss Forward: ", loss.item())
# ------------------------------------------------------------------
        
        # -----------------FASHION -> inverse-----------------
        batch_labels = data_enum[1][0]
        batch_data = data_enum[1][1]
        optimizer.zero_grad()
        outputs, surrounding_intensity = model(batch_data, inverse=True)
        outputs_for_loss = torch.cat((outputs, surrounding_intensity), dim=1)
        batch_labels_for_loss = torch.cat((batch_labels, torch.zeros(batch_labels.shape[0],1).to(modef.device)), dim=1)
        loss = loss_function(outputs_for_loss, batch_labels_for_loss)
        loss.backward()
        optimizer.step()
        running_loss_inverse += loss.item()
# ------------------------------------------------------------------
        # print("Running Loss Inverse: ", loss.item())
        # print("------------------------------------------------------------------")
# ------------------------------------------------------------------            

        if idx_enum % 20 == 0:
            print("Finished ", idx_enum, " MNIST batches and ", idx_enum, " FASHION batches, in total ", 2*idx_enum, " batches!!!!!")



    avg_train_loss_forward = running_loss_forward / len(MNIST_train_loader) # 每个 epoch 计算一次
    avg_train_loss_inverse = running_loss_inverse / len(FASHION_train_loader)


#%%
# 每个 epoch 训练完还要 validate 一次
    val_running_loss_forward = 0.0
    val_running_loss_inverse = 0.0
    correct_num_vali_forward = 0
    total_num_vali_forward = 0
    correct_num_vali_inverse = 0
    total_num_vali_inverse = 0


    print("--------------------------Start Validating!!--------------------------")
    with torch.no_grad():
        for idx_enum,data_enum in enumerate(zip(FASHION_validation_loader, MNIST_validation_loader)):
            
            # -----------------MNIST -> forward-----------------
            batch_labels = data_enum[0][0]
            batch_data = data_enum[0][1]
            outputs, surrounding_intensity = model(batch_data, inverse=False)
            outputs_for_loss = torch.cat((outputs, surrounding_intensity), dim=1)
            batch_labels_for_loss = torch.cat((batch_labels, torch.zeros(batch_labels.shape[0],1).to(modef.device)), dim=1)
            loss = loss_function(outputs_for_loss, batch_labels_for_loss)
            val_running_loss_forward += loss.item()
            _, predicted_labels = torch.max(outputs.data, 1)
            _, true_labels = torch.max(batch_labels.data, 1)
            total_num_vali_forward += batch_labels.size(0)
            correct_num_vali_forward += (predicted_labels == true_labels).sum().item()
            
            # -----------------FASHION -> inverse-----------------
            batch_labels = data_enum[1][0]
            batch_data = data_enum[1][1]
            outputs, surrounding_intensity = model(batch_data, inverse=True)
            outputs_for_loss = torch.cat((outputs, surrounding_intensity), dim=1)
            batch_labels_for_loss = torch.cat((batch_labels, torch.zeros(batch_labels.shape[0],1).to(modef.device)), dim=1)
            loss = loss_function(outputs_for_loss, batch_labels_for_loss)
            val_running_loss_inverse += loss.item()
            _, predicted_labels = torch.max(outputs.data, 1)
            _, true_labels = torch.max(batch_labels.data, 1)
            total_num_vali_inverse += batch_labels.size(0)
            correct_num_vali_inverse += (predicted_labels == true_labels).sum().item()
            if idx_enum % 50 == 0:
                print("Validated ", idx_enum, " MNIST batches and ", idx_enum, " FASHION batches, in total ", 2*idx_enum, " batches!!!!!")

    val_accuracy_forward = 100 * correct_num_vali_forward / total_num_vali_forward
    val_accuracy_inverse = 100 * correct_num_vali_inverse / total_num_vali_inverse
    avg_val_loss_forward = 2*val_running_loss_forward / len(FASHION_validation_loader)
    avg_val_loss_inverse = 2*val_running_loss_inverse / len(MNIST_validation_loader)
    
    print(f"Epoch [{epoch+1}/{epochs}], Forward Training Loss: {avg_train_loss_forward:.4f}, Inverse Training Loss: {avg_train_loss_inverse:.4f} \nForward Validation Loss: {avg_val_loss_forward:.4f}, Forward Validation Accuracy: {val_accuracy_forward:.2f}%, Inverse Validation Loss: {avg_val_loss_inverse:.4f}, Inserse Validation Accuracy: {val_accuracy_inverse:.2f}%")
    # 打印当前时间, 用于评估训练时间
    print("Finished training and validation @ ", datetime.now().strftime("%H:%M:%S"))


#%%
# 所有 epoch 完成以后还需要最终 test 一次, 以及保存模型参数
# Compute the correct rate of the model
model.eval()
correct_num_tested_forward = 0
total_num_tested_forward = 0
correct_num_tested_inverse = 0
total_num_tested_inverse = 0


print("--------------------------Start Final Testing!!--------------------------")
with torch.no_grad():
    for idx_enum,data_enum in enumerate(zip(FASHION_test_loader, MNIST_test_loader)):
            
        # -----------------MNIST -> forward-----------------
        batch_labels = data_enum[0][0]
        batch_data = data_enum[0][1]
        outputs, _ = model(batch_data, inverse=False)
        _, predicted_labels = torch.max(outputs.data, 1)
        _, true_labels = torch.max(batch_labels.data, 1)
        total_num_tested_forward += batch_labels.size(0)
        correct_num_tested_forward += (predicted_labels == true_labels).sum().item()
        
        # -----------------FASHION -> inverse-----------------
        batch_labels = data_enum[1][0]
        batch_data = data_enum[1][1]
        outputs, _ = model(batch_data, inverse=True)
        _, predicted_labels = torch.max(outputs.data, 1)
        _, true_labels = torch.max(batch_labels.data, 1)
        total_num_tested_inverse += batch_labels.size(0)
        correct_num_tested_inverse += (predicted_labels == true_labels).sum().item()
        if idx_enum % 50 == 0:
            print("Final Tested ", idx_enum, " MNIST batches and ", idx_enum, " FASHION batches, in total ", 2*idx_enum, " batches!!!!!")

print(f"Forward Test Accuracy: {100 * correct_num_tested_forward / total_num_tested_forward:.2f}%, Inverse Test Accuracy: {100 * correct_num_tested_inverse / total_num_tested_inverse:.2f}%")

# 打印当前时间, 用于评估训练时间
print("*********************************************************************")
print("*********************************************************************")
print("*********************************************************************")
print("    Finished final testing @", datetime.now().strftime("%H:%M:%S"))
print("*********************************************************************")
print("*********************************************************************")
print("*********************************************************************")

torch.save(model.state_dict(), "Main_Model/BD2NN_mnist_random_fashion_random.pt")

