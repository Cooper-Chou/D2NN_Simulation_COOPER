import numpy as np
import matplotlib.pyplot as plt

from Main_Model.model_define import L, w, N, lmbda, z

# mnist.npy 中的数据是二维数组, (50000, 28x28), 这里把它转化为了 (50000, 28, 28), 还改变了一下大小

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




# load the data here
mnist_data = np.load('dataset/MNIST/raw/mnist.npy', allow_pickle=True)

# [0][0] for train_data, [0][1] for train_label
# [1][0] for validation_data, [1][1] for validation_label
# [2][0] for test_data, [2][1] for test_label

data = mnist_data[0][0]

print(data.shape)
x_axis = np.linspace(-L/2, L/2, N)
y_axis = np.linspace(-L/2, L/2, N)
u0 = np.zeros((N, N, data.shape[0]),dtype=float)
for t in range(data.shape[0]):
    if t % 100 == 0: # visualize the progress, making me FEEL GOOD!!
        print("t = ", t,"in ", data.shape[0])

    ex = data[t]
    # change (28,28) this into the ideal shape of image, which is (28,28) for MNIST
    img_array = np.reshape(ex, (28, 28))
    for i in range(0, N):
        for j in range(0, N):
            u0[i][j][t] = pixel_value_center(x_axis[i] / (2*w), y_axis[j] / (2*w), img_array) # 此处x_axis[]和y_axis[]的值是在[-L/2, L/2]之间的, 2*w的值其实就是L/2, 所以 x_axis[i] / (2*w) 是处于[-1, 1]之间的, 也就是说这个值是归一化的, 表示图片的坐标相对位置

# change this accordingly
np.save('processed_dataset/u0_train_all.npy', u0)