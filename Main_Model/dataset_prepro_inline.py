import numpy as np
import matplotlib.pyplot as plt
from model_define import device, N
import torch

def zoom_and_upsamp(origin_img_array, zooming_coe, canvas_width, canvas_height):
    result_canvas = np.zeros((canvas_width, canvas_height), dtype=float)
    origin_height, origin_width = origin_img_array.shape
    croped_canvas_height = round(canvas_height * zooming_coe)
    croped_canvas_width = round(canvas_width * zooming_coe)

    height_blank = (canvas_height-croped_canvas_height)//2
    width_blank = (canvas_width-croped_canvas_width)//2

    for i in range(croped_canvas_height):
        for j in range(croped_canvas_width):
            result_canvas[(height_blank-1)+i][(width_blank-1)+j] = origin_img_array[round(i*(origin_height-1)/croped_canvas_height)][round(j*(origin_width-1)/croped_canvas_width)]

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(origin_img_array, cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(result_canvas, cmap='gray')
    # plt.show()
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    return result_canvas


# 一次就处理一个batch的数据的!

def label_modify(batch_label: np.ndarray) -> torch.tensor:
    # "mod" stands for modified
    batch_label_mod = np.zeros([batch_label.shape[0], 10])
    for i in range(batch_label.shape[0]):
        batch_label_mod[i, batch_label[i]] = 1
    return torch.tensor(batch_label_mod).double().to(device)

def u0_modify(batch_u0: np.ndarray, zooming_coefficient) -> torch.tensor:
    batch_u0_mod = np.zeros((batch_u0.shape[0], N, N), dtype=float)
    for t in range(batch_u0.shape[0]):
        # change (28,28) this into the ideal shape of image, which is (28,28) for MNIST
        img_array = batch_u0[t,:,:]
        batch_u0_mod[t,:,:] = zoom_and_upsamp(img_array, zooming_coefficient, N, N)
    return torch.tensor(batch_u0_mod).double().to(device)