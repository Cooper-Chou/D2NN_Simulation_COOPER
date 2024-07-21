import numpy as np
import matplotlib.pyplot as plt
from model_define import device
import torch

class data_prepro_inline():
    def __init__(self, mesh_num, start_x_pctg, start_y_pctg, square_size, zooming_coefficient, origin_size=28):
        self.origin_size = origin_size
        self.mesh_num = mesh_num
        cropped_size = round(mesh_num*zooming_coefficient)

        self.zoom_upsamp_mat_left = torch.zeros((mesh_num, origin_size), dtype=torch.double, device=device)
        self.zoom_upsamp_mat_right = torch.zeros((origin_size, mesh_num), dtype=torch.double, device=device)
        for i in range(cropped_size):
            self.zoom_upsamp_mat_left[i+(mesh_num-cropped_size)//2, round(i*(origin_size-1)/(cropped_size-1))] = 1
            self.zoom_upsamp_mat_right[round(i*(origin_size-1)/(cropped_size-1)), i+(mesh_num-cropped_size)//2] = 1

        middle_coor = mesh_num // 2
        # 十个目标点的坐标
        self.start_x = start_x_pctg*square_size + middle_coor
        self.start_y = start_y_pctg*square_size + middle_coor

        self.start_x = self.start_x.astype(int)
        self.start_y = self.start_y.astype(int)
        self.square_size = square_size

    # 一次就处理一个batch的数据的!
    def batch_u0_modify(self, batch_u0: torch.tensor) -> torch.tensor:
        batch_u0_mod = torch.matmul(torch.matmul(self.zoom_upsamp_mat_left, batch_u0), self.zoom_upsamp_mat_right)
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(batch_u0[0], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(bath_u0_mod[0], cmap='gray')
        # plt.show()
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        return batch_u0_mod


    def get_batch_labels_for_loss(self, batch_labels, output_type, surr_inten_num) -> torch.tensor:
        if output_type == "Full":
            labels_for_loss = torch.zeros((batch_labels.shape[0],self.mesh_num,self.mesh_num), dtype=float).to(device)
            for i in range(batch_labels.shape[0]):
                labels_for_loss[i, self.start_x[batch_labels[i]]:self.start_x[batch_labels[i]]+self.square_size, self.start_y[batch_labels[i]]:self.start_y[batch_labels[i]]+self.square_size] = 1
                # batch_labels[i] 既是标签值, 也作为索引!
            return labels_for_loss.reshape(labels_for_loss.shape[0], labels_for_loss.shape[1]*labels_for_loss.shape[2])
        
        elif output_type == "Clps":
            labels_for_loss = torch.zeros((batch_labels.shape[0],10+surr_inten_num), dtype=float).to(device)
            # 在 batch_labels 后面加一列 0, 对应于杂散光场的目标值
            for i in range(batch_labels.shape[0]):
                labels_for_loss[i, batch_labels[i]] = 1
            return labels_for_loss
        else:
            return None

