import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sys
sys.path.append("D:\\OneDrive\\SUSTech\\MetaSurface_Lab\\Python Project\\D2NN_Simulation_COOPER\\Main_Model") 
print("sys_path = ",sys.path)
import model_define as modef
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

torch.cuda.empty_cache() # 清空显存!

torch.autograd.set_detect_anomaly(True) # 检测梯度异常

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
    
batch_size = 32
dataset_dir_mnist = "dataset/MNIST_processed"
dataset_dir_fashion = "dataset/FASHION_processed"
dir_prefix = "Main_Model/Result_and_post_process/"

model_list = ['BD2NN_mnist_random_fashion_random', 'BD2NN_mnist_random_fashion_random', 'BD2NN_mnist_random_fashion_random', 'BD2NN_mnist_random_fashion_random', 'BD2NN_mnist_0_5_fashion_0_5','BD2NN_mnist_0_8_fashion_0_3','BD2NN_mnist_0_9_fashion_0_9']
dataset_list = [['_random','_random'],['_0_5','_0_5'],['_0_8','_0_3'],['_0_9','_0_9'],['_0_5','_0_5'],['_0_8','_0_3'],['_0_9','_0_9']]
labels_fashion = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
labels_mnist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

for round_idx in range(len(model_list)):
    MNIST_test_dataset = CustomNpyDataset(dataset_dir_mnist+dataset_list[round_idx][0]+"/u0_test", "u0_test_", dataset_dir_mnist+dataset_list[round_idx][0]+"/label_test_all.npy")
    MNIST_test_loader = DataLoader(MNIST_test_dataset, batch_size=batch_size, shuffle=True)
    FASHION_test_dataset = CustomNpyDataset(dataset_dir_fashion+dataset_list[round_idx][1]+"/u0_test", "u0_test_", dataset_dir_fashion+dataset_list[round_idx][1]+"/label_test_all.npy")
    FASHION_test_loader = DataLoader(FASHION_test_dataset, batch_size=batch_size, shuffle=True)
    print("successfully loaded!!")

    m_state_dict = torch.load(dir_prefix+model_list[round_idx]+'.pt')
    D2NN = modef.OpticalNetwork(M=modef.mesh_num, L=modef.L, lmbda=modef.lmbda, z=modef.z).to(modef.device)
    D2NN.load_state_dict(m_state_dict)
    D2NN.eval()
    print(D2NN)

    correct_num_tested_forward = 0
    total_num_tested_forward = 0
    correct_num_tested_inverse = 0
    total_num_tested_inverse = 0
    confusion_matrix_forward = np.zeros((10, 10),dtype=np.int32)
    confusion_matrix_inverse = np.zeros((10, 10),dtype=np.int32)

    print("--------------------------Start Final Testing!!--------------------------")
    with torch.no_grad():
        for idx_enum,data_enum in enumerate(zip(FASHION_test_loader, MNIST_test_loader)):
                
            # -----------------FASHION -> forward-----------------
            batch_labels = data_enum[0][0]
            batch_data = data_enum[0][1]
            outputs = D2NN(batch_data, inverse=False)
            _, predicted_labels = torch.max(outputs.detach(), 1)
            _, true_labels = torch.max(batch_labels.detach(), 1)
            confusion_matrix_forward += confusion_matrix(y_true=true_labels.cpu(), y_pred=predicted_labels.cpu(),labels=np.array(range(10)))
            total_num_tested_forward += batch_labels.size(0)
            correct_num_tested_forward += (predicted_labels == true_labels).sum().item()
            
            # -----------------MNIST -> inverse-----------------
            batch_labels = data_enum[1][0]
            batch_data = data_enum[1][1]
            outputs = D2NN(batch_data, inverse=True)
            _, predicted_labels = torch.max(outputs.detach(), 1)
            _, true_labels = torch.max(batch_labels.detach(), 1)
            confusion_matrix_inverse += confusion_matrix(y_true=true_labels.cpu(), y_pred=predicted_labels.cpu(), labels=np.array(range(10)))
            total_num_tested_inverse += batch_labels.size(0)
            correct_num_tested_inverse += (predicted_labels == true_labels).sum().item()
            if idx_enum % 60 == 0:
                print("Final Tested ", idx_enum, " MNIST batches and ", idx_enum, " FASHION batches, in total ", 2*idx_enum, " batches!!!!!")
    result_msg = f"Model {model_list[round_idx]} + dataset MNIST{dataset_list[round_idx][0]} & FASHION{dataset_list[round_idx][1]}: \n Forward Test Accuracy: {100 * correct_num_tested_forward / total_num_tested_forward:.2f}%, Inverse Test Accuracy: {100 * correct_num_tested_inverse / total_num_tested_inverse:.2f}%\n--------------------------------------------------\n"
    print(result_msg)

    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_forward, display_labels=labels_fashion)
    disp.plot(
        include_values=True,            # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
        ax=None,                        # 同上
        xticks_rotation=30,   # 同上
        values_format="d"               # 显示的数值格式
    )
    plt.savefig('Main_Model/Result_and_post_process/FORWARD_confusion_matrix_'+model_list[round_idx]+'_mnist'+dataset_list[round_idx][0]+'_fashion'+dataset_list[round_idx][1]+'.png')

    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_inverse, display_labels=labels_mnist)
    disp.plot(
        include_values=True,            # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
        ax=None,                        # 同上
        xticks_rotation=0,   # 同上
        values_format="d"               # 显示的数值格式
    )
    plt.savefig('Main_Model/Result_and_post_process/INVERSE_confusion_matrix_'+model_list[round_idx]+'_mnist'+dataset_list[round_idx][0]+'_fashion'+dataset_list[round_idx][1]+'.png')

    with open('Main_Model/Result_and_post_process/Accuracies.txt', 'a', encoding='utf-8') as file:
        file.write(result_msg)
        file.close()