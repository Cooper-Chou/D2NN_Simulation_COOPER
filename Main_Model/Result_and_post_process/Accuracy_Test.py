import sys
sys.path.append("D:\\OneDrive\\SUSTech\\MetaSurface_Lab\\Python Project\\D2NN_Simulation_COOPER\\Main_Model") 
# print("sys_path = ",sys.path)
import torch
import torch.nn as nn
import numpy as np
import dataset_prepro_inline as dpi
from tqdm import tqdm
import glob
from load_data import FASHION_train_loader, FASHION_test_loader, MNIST_train_loader, MNIST_test_loader, prepro_inline_flag


import model_define as modef
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

torch.cuda.empty_cache() # 清空显存!

torch.autograd.set_detect_anomaly(False) # 测试就不检测梯度异常了, 不然好慢的


parent_dir = 'Main_Model\\Result_and_post_process\\'
model_name_list = glob.glob(parent_dir+'BD2NN*.pt')
for idx in range(len(model_name_list)):
    model_name_list[idx] = model_name_list[idx][len(parent_dir):-3]

labels_fashion = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
labels_mnist = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

with open('Main_Model/Result_and_post_process/Accuracies.txt', 'a', encoding='utf-8') as file:
    file.write("\n\n\nGenerated at: "+str(datetime.now())+'\n')
    file.close()

def get_paras_from_name(model_name:str):
    _,_,z,_,L,_,PSC,_,mesh,_,SW,_,SN,_,SS,_,Norm,_,ZC,_,LT,_,OT = model_name.split('_', maxsplit=-1)
    return float(z), float(L), float(PSC), int(mesh), float(SW), int(SN), int(SS), Norm, float(ZC), LT, OT



for round_idx in range(len(model_name_list)):
    print(f"Testing [{round_idx}/{len(model_name_list)}]: [{model_name_list[round_idx]}]")

    m_state_dict = torch.load(parent_dir+model_name_list[round_idx]+'.pt')
    z,L,para_scale_coe,mesh_num,surr_inten_weight,surr_inten_num,square_size,norm_type,zooming_coefficient,loss_type,output_for_loss_type = get_paras_from_name(model_name_list[round_idx])
    print("mesh_num = ",mesh_num, "para_scale_coe = ",para_scale_coe)

    prepro = dpi.data_prepro_inline(mesh_num=mesh_num, zooming_coefficient=zooming_coefficient)
    D2NN = modef.OpticalNetwork(L=L, lmbda=modef.lmbda, z=z, square_size=square_size, mesh_num=mesh_num, para_scale_coe=para_scale_coe, output_for_loss_type=output_for_loss_type).to(modef.device)
    D2NN.load_state_dict(m_state_dict)
    D2NN.eval()
    # print(D2NN)

    correct_num_tested_forward = 0
    total_num_tested_forward = 0
    correct_num_tested_inverse = 0
    total_num_tested_inverse = 0
    confusion_matrix_forward = np.zeros((10, 10),dtype=np.int32)
    confusion_matrix_inverse = np.zeros((10, 10),dtype=np.int32)


    with torch.no_grad():
        process_bar = tqdm(enumerate(zip(FASHION_test_loader, MNIST_test_loader)), total=len(FASHION_test_loader), ncols=160, position=0)
        for idx_enum,data_enum in process_bar:
                
            # -----------------FASHION -> forward-----------------
            batch_labels = data_enum[0][0]
            batch_u0 = prepro.batch_u0_modify(data_enum[0][1]) if prepro_inline_flag else data_enum[0][1]
            _, predicted_labels = D2NN(batch_u0, inverse=False)
            true_labels = batch_labels.clone().to(modef.device)
            total_num_tested_forward += batch_labels.size(0)
            correct_num_tested_forward += (predicted_labels == true_labels).sum().item()

            confusion_matrix_forward += confusion_matrix(true_labels.cpu().numpy(), predicted_labels.cpu().numpy(), labels=np.array(range(len(labels_fashion))))
            
            # -----------------MNIST -> inverse-----------------
            batch_labels = data_enum[1][0]
            batch_u0 = prepro.batch_u0_modify(data_enum[1][1]) if prepro_inline_flag else data_enum[1][1]
            _, predicted_labels = D2NN(batch_u0, inverse=True)
            true_labels = batch_labels.clone().to(modef.device)
            total_num_tested_inverse += batch_labels.size(0)
            correct_num_tested_inverse += (predicted_labels == true_labels).sum().item()

            confusion_matrix_inverse += confusion_matrix(true_labels.cpu().numpy(), predicted_labels.cpu().numpy(), labels=np.array(range(len(labels_mnist))))

            process_bar.set_postfix(Finished_Batches = idx_enum+1)
            process_bar.update()  # 默认参数n=1，每update一次，进度+n

    result_msg = f"Model [{model_name_list[round_idx]}]: \n Forward Test Accuracy: [{100 * correct_num_tested_forward / total_num_tested_forward:.2f}%], Inverse Test Accuracy: [{100 * correct_num_tested_inverse / total_num_tested_inverse:.2f}%]\n--------------------------------------------------\n"
    print('\n'+result_msg)

    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_forward, display_labels=labels_fashion)
    disp.plot(
        include_values=True,            # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
        ax=None,                        # 同上
        xticks_rotation=30,   # 同上
        values_format="d"               # 显示的数值格式
    )
    plt.savefig('Main_Model/Result_and_post_process/FORWARD_CM_'+model_name_list[round_idx]+'.png')

    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_inverse, display_labels=labels_mnist)
    disp.plot(
        include_values=True,            # 混淆矩阵每个单元格上显示具体数值
        cmap="viridis",                 # 不清楚啥意思，没研究，使用的sklearn中的默认值
        ax=None,                        # 同上
        xticks_rotation=0,   # 同上
        values_format="d"               # 显示的数值格式
    )
    plt.savefig('Main_Model/Result_and_post_process/INVERSE_CM_'+model_name_list[round_idx]+'.png')

    with open('Main_Model/Result_and_post_process/Accuracies.txt', 'a', encoding='utf-8') as file:
        file.write(result_msg)
        file.close()