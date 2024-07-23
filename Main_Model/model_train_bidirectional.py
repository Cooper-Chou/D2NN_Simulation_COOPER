import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import model_define as modef
from datetime import datetime
from tqdm import tqdm
import dataset_prepro_inline as dpi
from load_data import FASHION_train_loader, FASHION_test_loader, MNIST_train_loader, MNIST_test_loader, prepro_inline_flag
import itertools

torch.cuda.empty_cache() # 清空显存!

# *********************改了代码的话一定要改成True!!!! 
# 严重影响性能，仅在Debug中使用!!!!!!
torch.autograd.set_detect_anomaly(False) # 检测梯度异常  
# *********************

# Hyperparameters
learning_rate = 0.001
epochs = 20

#%%
# para_combinations = list(itertools.product(modef.surr_inten_weight, modef.norm_type, modef.zooming_coefficient, modef.output_for_loss_type))
# for surr_inten_weight, norm_type, zooming_coefficient, output_for_loss_type in para_combinations:

# Initialize network, loss, and optimizer
model = modef.OpticalNetwork(L=modef.L, lmbda=modef.lmbda, z=modef.z, square_size=modef.square_size, mesh_num=modef.mesh_num, para_scale_coe=modef.para_scale_coe, output_for_loss_type=modef.output_for_loss_type).to(modef.device)
loss_function = nn.MSELoss() if modef.loss_type == "MSE" else nn.CrossEntropyLoss() # X.Lin 本人说可以换成交叉熵, 效果更好
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
prepro = dpi.data_prepro_inline(modef.mesh_num, modef.start_x_pctg, modef.start_y_pctg, modef.square_size, modef.zooming_coefficient)

best_avrg_accuracy = 0.0
best_dict = model.state_dict()
best_epoch = 0

saved_name = f"BD2NN_z_{modef.z}_L_{modef.L}_PSC_{modef.para_scale_coe}_mesh_{modef.mesh_num}_SW_{modef.surr_inten_weight}_SN_{modef.surr_inten_num}_SS_{modef.square_size}_Norm_{modef.norm_type}_ZC_{modef.zooming_coefficient}_LT_{modef.loss_type}_OT_{modef.output_for_loss_type}_PH-FA.pt"

# Training loop
for epoch in range(epochs):
    print(f"-------------------------- [EPOCH: {epoch+1}/{epochs}] --------------------------")
    # Training phase
    model.train()
    running_loss_forward = 0.0
    running_loss_inverse = 0.0

    process_bar = tqdm(enumerate(zip(FASHION_train_loader, MNIST_train_loader)),desc=f"EPOCH {epoch+1} --- TRAINING -> ", total=len(FASHION_train_loader), ncols=160, position=0)
    for idx_enum,data_enum in process_bar:
        # zip 可以把两个可迭代对象(这里是 dataloader)一个一个对应起来, 像梳子那样(详情上网查), 然后返回的也是可以迭代的; enumerate 可以在返回可迭代对象的同时返回索引(具体上网查), 反正这两个函数可方便了
        # 在这个 for 循环里, 每次迭代 data_enum 都是一个元组, 元组里面有两个元素, 分别是两个 dataloader 里的一个 batch 的数据

        # -----------------FASHION -> forward-----------------
        batch_labels = data_enum[0][0]
        batch_u0 = prepro.batch_u0_modify(data_enum[0][1]) if prepro_inline_flag else data_enum[0][1]
        # u0 和 label 都是 tensor double 类型的!!
        optimizer.zero_grad()
        output_for_loss, _ = model(batch_u0, inverse=False)
        batch_labels_for_loss = prepro.get_batch_labels_for_loss(batch_labels, modef.output_for_loss_type, modef.surr_inten_num)
        loss = loss_function(output_for_loss, batch_labels_for_loss)
        loss.backward() # backward() 函数是 tensor 自带的函数, 所有的 lossfunction forward() 以后都返回 tensor 类型的结果! 而且可能返回的 loss 已经是开过 require_grad 和 retain_graph 的了, 后面 optimizer 就会直接根据 loss 的梯度图来迭代
        optimizer.step()
        running_loss_forward += loss.item()
        
        # -----------------MNIST -> inverse-----------------
        batch_labels = data_enum[1][0]
        batch_u0 = prepro.batch_u0_modify(data_enum[1][1]) if prepro_inline_flag else data_enum[1][1]
        optimizer.zero_grad()
        output_for_loss, _ = model(batch_u0, inverse=True)
        batch_labels_for_loss = prepro.get_batch_labels_for_loss(batch_labels, modef.output_for_loss_type, modef.surr_inten_num)
        loss = loss_function(output_for_loss, batch_labels_for_loss)
        loss.backward()
        optimizer.step()
        running_loss_inverse += loss.item() 

        process_bar.set_postfix(Loss_Forward = running_loss_forward, Loss_Inverse = running_loss_inverse)
        process_bar.update()  # 默认参数n=1，每update一次，进度+n

    avg_train_loss_forward = running_loss_forward / len(MNIST_train_loader) # 每个 epoch 计算一次
    avg_train_loss_inverse = running_loss_inverse / len(FASHION_train_loader)


#%%
# 每个 epoch 训练完还要 validate 一次
    model.eval()
    val_running_loss_forward = 0.0
    val_running_loss_inverse = 0.0
    correct_num_vali_forward = 0
    total_num_vali_forward = 0
    correct_num_vali_inverse = 0
    total_num_vali_inverse = 0

    # ---------------------Start Validating!!---------------------
    with torch.no_grad():
        process_bar = tqdm(enumerate(zip(FASHION_test_loader, MNIST_test_loader)),desc=f"EPOCH {epoch+1} --- TESTING -> ", total=len(FASHION_test_loader), ncols=150, position=0)
        for idx_enum,data_enum in process_bar:
            
            # -----------------FASHION -> forward-----------------
            batch_labels = data_enum[0][0]
            batch_u0 = prepro.batch_u0_modify(data_enum[0][1]) if prepro_inline_flag else data_enum[0][1]
            output_for_loss, predicted_labels = model(batch_u0, inverse=False)
            batch_labels_for_loss = prepro.get_batch_labels_for_loss(batch_labels, modef.output_for_loss_type, modef.surr_inten_num)
            loss = loss_function(output_for_loss, batch_labels_for_loss)
            val_running_loss_forward += loss.item()
            true_labels = batch_labels.clone().to(modef.device)
            total_num_vali_forward += batch_labels.size(0)
            correct_num_vali_forward += (predicted_labels == true_labels).sum().item()
            
            # -----------------MNIST -> inverse-----------------
            batch_labels = data_enum[1][0]
            batch_u0 = prepro.batch_u0_modify(data_enum[1][1]) if prepro_inline_flag else data_enum[1][1]
            output_for_loss, predicted_labels = model(batch_u0, inverse=True)
            batch_labels_for_loss = prepro.get_batch_labels_for_loss(batch_labels, modef.output_for_loss_type, modef.surr_inten_num)
            loss = loss_function(output_for_loss, batch_labels_for_loss)
            val_running_loss_inverse += loss.item()
            true_labels = batch_labels.clone().to(modef.device)
            total_num_vali_inverse += batch_labels.size(0)
            correct_num_vali_inverse += (predicted_labels == true_labels).sum().item()

            process_bar.set_postfix(Loss_Forward = val_running_loss_forward, Loss_Inverse = val_running_loss_inverse)
            process_bar.update()  # 默认参数n=1，每update一次，进度+n

    val_accuracy_forward = 100 * correct_num_vali_forward / total_num_vali_forward
    val_accuracy_inverse = 100 * correct_num_vali_inverse / total_num_vali_inverse
    avg_val_loss_forward = 2*val_running_loss_forward / len(FASHION_test_loader)
    avg_val_loss_inverse = 2*val_running_loss_inverse / len(MNIST_test_loader)

    if (val_accuracy_forward+val_accuracy_inverse)/2 > best_avrg_accuracy:
        best_dict = model.state_dict()
        torch.save(best_dict, "Main_Model/Result_and_post_process/"+saved_name)
        best_avrg_accuracy = (val_accuracy_forward+val_accuracy_inverse)/2
        best_epoch = epoch
    
    print(f"Epoch [{epoch+1}/{epochs}], Forward Training Loss: {avg_train_loss_forward:.4f}, Inverse Training Loss: {avg_train_loss_inverse:.4f} \nForward Validation Loss: {avg_val_loss_forward:.4f}, Forward Validation Accuracy: {val_accuracy_forward:.2f}%, Inverse Validation Loss: {avg_val_loss_inverse:.4f}, Inserse Validation Accuracy: {val_accuracy_inverse:.2f}%")
    # 打印当前时间, 用于评估训练时间

print("Finished training and validation @ ", datetime.now().strftime("%H:%M:%S"))


#%%
# 所有 epoch 完成以后还需要最终 test 一次, 以及保存模型参数
# Compute the correct rate of the model
model.load_state_dict(best_dict )
model.eval()
correct_num_tested_forward = 0
total_num_tested_forward = 0
correct_num_tested_inverse = 0
total_num_tested_inverse = 0


print("--------------------------Start Final Testing!!--------------------------")
with torch.no_grad():
    process_bar = tqdm(enumerate(zip(FASHION_test_loader, MNIST_test_loader)),desc="FINAL-TESTING -> ", total=len(FASHION_test_loader), ncols=160, position=0)
    for idx_enum,data_enum in process_bar:
            
        # -----------------FASHION -> forward-----------------
        batch_labels = data_enum[0][0]
        batch_u0 = prepro.batch_u0_modify(data_enum[0][1]) if prepro_inline_flag else data_enum[0][1]
        output_for_loss, predicted_labels = model(batch_u0, inverse=False)
        true_labels = batch_labels.clone().to(modef.device)
        total_num_tested_forward += batch_labels.size(0)
        correct_num_tested_forward += (predicted_labels == true_labels).sum().item()
        
        # -----------------MNIST -> inverse-----------------
        batch_labels = data_enum[1][0]
        batch_u0 = prepro.batch_u0_modify(data_enum[1][1]) if prepro_inline_flag else data_enum[1][1]
        output_for_loss, predicted_labels = model(batch_u0, inverse=True)
        true_labels = batch_labels.clone().to(modef.device)
        total_num_tested_inverse += batch_labels.size(0)
        correct_num_tested_inverse += (predicted_labels == true_labels).sum().item()

        process_bar.set_postfix(Finished_Batches = idx_enum+1)
        process_bar.update()  # 默认参数n=1，每update一次，进度+n
        


print(f"\nForward Test Accuracy: {100 * correct_num_tested_forward / total_num_tested_forward:.2f}%, Inverse Test Accuracy: {100 * correct_num_tested_inverse / total_num_tested_inverse:.2f}%")

# 打印当前时间, 用于评估训练时间
print("\n*********************************************************************")
print("    Finished final testing @", datetime.now().strftime("%H:%M:%S"))
print("*********************************************************************\n")



print(f"Best accuracy in [epoch {best_epoch+1}], saved as {saved_name}!")

