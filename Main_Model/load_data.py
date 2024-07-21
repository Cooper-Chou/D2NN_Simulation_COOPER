import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import model_define as modef

prepro_inline_flag = True # False # 

print(f"********** Trained in {'PREPRO_INLINE' if prepro_inline_flag else 'SEPARATE'} mode! *********\n")

batch_size = 32

# -------------------load data----------------------------

class CustomNpyDataset_Inline(Dataset):
    def __init__(self, data_np_array, type_tag:str):
        super(CustomNpyDataset_Inline, self).__init__()
        self.type_tag = type_tag

        # print(data_np_array[0][0].shape)
        # print(data_np_array[0][1].shape)

        if self.type_tag == "train":
            self.u0_np_array = data_np_array[0][0]
            self.label_np_array = data_np_array[0][1]
        elif self.type_tag == "test":
            self.u0_np_array = data_np_array[1][0]
            self.label_np_array = data_np_array[1][1]
        else:
            print("Invalid type_tag!!")
            
        # [0][0] for train_data, [0][1] for train_label
        # [1][0] for test_data, [1][1] for test_label

        # u0.shape = (batch_size, 28, 28)! label.shape = (batch_size)!

    def __len__(self):
        return self.u0_np_array.shape[0]

    def __getitem__(self, idx):
        u0 = self.u0_np_array[idx]
        labels = self.label_np_array[idx]
        return torch.tensor(labels).int().to(modef.device), torch.tensor(u0).double().to(modef.device)
    

class CustomNpyDataset_Separate(Dataset):
    def __init__(self, data_dir:str, data_file:str, label_np_array):
        super(CustomNpyDataset_Separate, self).__init__()
        self.data_dir = data_dir # 一个字符串!
        self.data_file = data_file # 一个字符串!
        self.label_np_array = label_np_array

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        # print("__getitem__ called")
        u0 = np.load(self.data_dir + "/" + self.data_file + "%d.npy"%idx, allow_pickle=True)
        labels = self.label_np_array[idx]
        return torch.tensor(labels).int().to(modef.device), torch.tensor(u0).double().to(modef.device)
    
# labels 在两个数据集中都是以 numpy 数组的形式输出, 不管 u0 是 tensor 还是 numpy! 而且 labels.shape 都是 (batch_size,) 的一维形式!

MNIST_train_loader = None
MNIST_test_loader = None
MNIST_validation_loader = None
FASHION_train_loader = None
FASHION_test_loader = None
FASHION_validation_loader = None

dataset_mnist = np.load("dataset/MNIST/mnist.npy", allow_pickle=True)
dataset_fashion = np.load("dataset/FASHION/fashion.npy", allow_pickle=True)

# ***************************
# ***************************
# DataLoader 会把输出的数据类型自动转化为 tensor! 我草他妈的
# ***************************
# ***************************

if prepro_inline_flag:
    MNIST_train_dataset = CustomNpyDataset_Inline(dataset_mnist, "train")
    MNIST_train_loader = DataLoader(MNIST_train_dataset, batch_size=batch_size, shuffle=True)

    MNIST_test_dataset = CustomNpyDataset_Inline(dataset_mnist, "test")
    MNIST_test_loader = DataLoader(MNIST_test_dataset, batch_size=batch_size, shuffle=True)
    # -----------------------------------------------------------------------------------------------------------------------------
    FASHION_train_dataset = CustomNpyDataset_Inline(dataset_fashion, "train")
    FASHION_train_loader = DataLoader(FASHION_train_dataset, batch_size=batch_size, shuffle=True)

    FASHION_test_dataset = CustomNpyDataset_Inline(dataset_fashion, "test")
    FASHION_test_loader = DataLoader(FASHION_test_dataset, batch_size=batch_size, shuffle=True)


else:
    dataset_dir_mnist = "dataset/MNIST_processed_0_5"
    dataset_dir_fashion = "dataset/FASHION_processed_0_5"

    MNIST_train_dataset = CustomNpyDataset_Separate(dataset_dir_mnist+"/u0_train", "u0_train_", dataset_mnist[0][1])
    MNIST_train_loader = DataLoader(MNIST_train_dataset, batch_size=batch_size, shuffle=True)

    MNIST_validation_dataset = CustomNpyDataset_Separate(dataset_dir_mnist+"/u0_validation", "u0_validation_", dataset_mnist[1][1])
    MNIST_validation_loader = DataLoader(MNIST_validation_dataset, batch_size=batch_size, shuffle=True)

    MNIST_test_dataset = CustomNpyDataset_Separate(dataset_dir_mnist+"/u0_test", "u0_test_", dataset_mnist[2][1])
    MNIST_test_loader = DataLoader(MNIST_test_dataset, batch_size=batch_size, shuffle=True)
    # -----------------------------------------------------------------------------------------------------------------------------
    FASHION_train_dataset = CustomNpyDataset_Separate(dataset_dir_fashion+"/u0_train", "u0_train_", dataset_fashion[0][1])
    FASHION_train_loader = DataLoader(FASHION_train_dataset, batch_size=batch_size, shuffle=True)

    FASHION_validation_dataset = CustomNpyDataset_Separate(dataset_dir_fashion+"/u0_validation", "u0_validation_", dataset_fashion[1][1])
    FASHION_validation_loader = DataLoader(FASHION_validation_dataset, batch_size=batch_size, shuffle=True)

    FASHION_test_dataset = CustomNpyDataset_Separate(dataset_dir_fashion+"/u0_test", "u0_test_", dataset_fashion[2][1])
    FASHION_test_loader = DataLoader(FASHION_test_dataset, batch_size=batch_size, shuffle=True)

print("************** SUCCESSFULLY LOADED!! ************** \n")