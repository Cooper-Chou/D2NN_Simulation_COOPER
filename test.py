import torch
import torch.nn as nn
import train.model_define as model
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

# Initialize network, loss, and optimizer
D2NN = model.OpticalNetwork(model.M, model.L, model.lmbda, model.z).to(model.device)
D2NN.load_state_dict((torch.load("train\weights_large_uniform.pt")))
D2NN.eval()



# -------------------load data----------------------------
class CustomNpyDataset(Dataset):
    def __init__(self, data_file, label_file):
        self.data_file = np.load(data_file, mmap_mode='r')
        self.label_file = np.load(label_file, mmap_mode='r')
        self.data_len = np.load(data_file, mmap_mode='r').shape[2]  # Assuming data shape is [250, 250, num_samples]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data = self.data_file[:,:,idx]
        labels = self.label_file[:, idx]
        return torch.tensor(data).double().to(model.device), torch.tensor(labels).double().to(model.device)
    

print("Device: ", model.device)
validation_dataset = CustomNpyDataset("processed_dataset/u0_validation_all.npy", "processed_dataset/validation_label_all.npy")
print("successfully loaded")

# print(validation_dataset.__getitem__(100)[0].type())
# fig=plt.figure() #创建画布
# ax=fig.subplots() #创建图表
# ax.imshow(validation_dataset.__getitem__(100)[0].cpu())
# plt.show()

outputs = D2NN(validation_dataset.__getitem__(100)[0].reshape(1, 250, 250))
print(outputs)