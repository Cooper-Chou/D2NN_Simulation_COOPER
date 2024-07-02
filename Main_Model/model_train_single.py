import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import model_define as modef

torch.cuda.empty_cache() # 清空显存!

torch.autograd.set_detect_anomaly(True) # 检测梯度异常


#%%
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
        return torch.tensor(data).double().to(modef.device), torch.tensor(labels).double().to(modef.device)
    
batch_size = 128

train_dataset = CustomNpyDataset("processed_dataset/u0_train_all.npy", "processed_dataset/train_label_all.npy")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = CustomNpyDataset("processed_dataset/u0_test_all.npy", "processed_dataset/test_label_all.npy")
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = CustomNpyDataset("processed_dataset/u0_validation_all.npy", "processed_dataset/validation_label_all.npy")
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
print("successfully loaded")


#%%
# Hyperparameters
learning_rate = 0.003
epochs = 6
batch_size = 128

# Initialize network, loss, and optimizer
model = modef.OpticalNetwork(modef.M, modef.L, modef.lmbda, modef.z).to(modef.device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    print("epoch: ", epoch+1)
    # Training phase
    model.train()
    running_loss = 0.0
    turns = 0
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = loss_function(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        turns += 1
        if turns % 100 == 0:
            print("Finished ", turns, "batches")
            turns = 0

    avg_train_loss = running_loss / len(train_loader) # 每个 epoch 计算一次


#%%
# 每个 epoch 训练完测试一次
    # Evaluation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in train_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1) # "_" 下划线变量表示不需要这个值, 随便取名
            _, true_labels = torch.max(labels.data, 1) # 此处的 1 表示 dim; 取的值其实是 index, 只是恰好与所需的真实值相等, 不建议这么做!
            total += labels.size(0)

            correct += (predicted == true_labels).sum().item()
            # 布尔运算以后得到的是一个 tensor, 保存每一次预测的是正确还是错误; .sum() 返回的是一个 tensor, .item() 取出 tensor 中的值

    accuracy = 100 * correct / total
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for data, labels in validation_loader:
            outputs = model(data)
            loss = loss_function(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            _, true_labels = torch.max(labels.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == true_labels).sum().item()

    val_accuracy = 100 * correct_val / total_val
    avg_val_loss = val_running_loss / len(validation_loader)
    
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}, Training Accuracy: {accuracy:.2f}%, \nValidation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, labels in test_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        _, true_labels = torch.max(labels.data, 1)
        total += true_labels.size(0)
        correct += (predicted == true_labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "train/weights_large_uniform.pt")

