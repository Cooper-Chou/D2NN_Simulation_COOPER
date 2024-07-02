import numpy as np
import matplotlib.pyplot as plt
from Main_Model.model_define import L, w, N, lmbda, z

def pixel_value_center(x, y, imgArray):
    # Calculate the center of the image
    rows, cols = imgArray.shape # (theoretically) 28, 28
    center_x = rows // 2 #(theoretically) 14
    center_y = cols // 2 #(theoretically) 14
    x = x * rows
    y = y * cols
    
    # Adjust the coordinates so that (0,0) is at the center
    x_adjusted = center_x + round(x)
    y_adjusted = center_y - round(y)

    # Check if the adjusted coordinates are within the image bounds
    if 0 <= x_adjusted < rows and 0 <= y_adjusted < cols:
        # Return 1 if pixel value is 255, otherwise 0
        value = imgArray[x_adjusted, cols - y_adjusted - 1]
    else:
        # If the adjusted coordinates are outside the image bounds, return 0
        value = 0

    return value


def label_modify(batch_label):
    # "mod" stands for modified
    batch_label_mod = np.zeros([10, batch_label.shape[0]])
    for i in range(batch_label.shape[0]):
        batch_label_mod[batch_label[i], i] = 1
    return batch_label_mod

def u0_modify(batch_u0):
    # print(batch_u0.shape)
    x_axis = np.linspace(-L/2, L/2, N)
    y_axis = np.linspace(-L/2, L/2, N)
    batch_u0_mod = np.zeros((N, N, batch_u0.shape[0]),dtype=float)
    for t in range(batch_u0.shape[0]):
        ex = batch_u0[t]
        # change (28,28) this into the ideal shape of image, which is (28,28) for MNIST
        img_array = np.reshape(ex, (28, 28))
        for i in range(0, N):
            for j in range(0, N):
                batch_u0_mod[i][j][t] = pixel_value_center(x_axis[i] / (2*w), y_axis[j] / (2*w), img_array) # 此处x_axis[]和y_axis[]的值是在[-L/2, L/2]之间的, 2*w的值其实就是L/2, 所以 x_axis[i] / (2*w) 是处于[-1, 1]之间的, 也就是说这个值是归一化的, 表示图片的坐标相对位置
    return batch_u0_mod