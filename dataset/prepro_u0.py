import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("C:\\Users\\25460\\OneDrive\\SUSTech\\MetaSurface_Lab\\Python Project\\D2NN_Simulation_COOPER\\Main_Model") 
from model_define import L, mesh_num, lmbda, z

dataset_name = ['MNIST','mnist'] # ['FASHION','fashion'] # 

save_dir_1 = 'dataset/'+dataset_name[0]+'_processed_0_5/'
# save_dir_1 = 'dataset/'+dataset_name[0]+'_processed/'
save_dir_2 = [['u0_train/u0_train_', 0], ['u0_validation/u0_validation_', 1], ['u0_test/u0_test_', 2]] # 这个顺序很重要!

print("The dataset is: ", dataset_name[0])

zooming_coefficient = 0.5


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

    # # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(origin_img_array, cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(result_canvas, cmap='gray')
    # plt.show()
    # # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    return result_canvas



# load the data here
origin_datasetS_28_28 = np.load('dataset/'+dataset_name[0]+'/'+dataset_name[1]+'.npy', allow_pickle=True)

# [0][0] for train_data, [0][1] for train_label
# [1][0] for validation_data, [1][1] for validation_label
# [2][0] for test_data, [2][1] for test_label

# data_train = mnist_data[0][0]
# data_vali = mnist_data[1][0]
# data_test = mnist_data[2][0]

for dir,idx in save_dir_2: # 依次处理 train, validation, test 数据集
    origin_dataset_28_28 = origin_datasetS_28_28[idx][0]
    for t in range(origin_dataset_28_28.shape[0]):
        img_array = origin_dataset_28_28[t,:,:]

        # random_covered_pctg = np.random.rand() * 0.8 + 0.2

        # dataset_250_250 = img_embedding(N,N, random_covered_pctg,random_covered_pctg, img_array) 
        dataset_250_250 = zoom_and_upsamp(img_array, zooming_coefficient, mesh_num,mesh_num) 
        np.save(save_dir_1+dir+'%d.npy'%t, dataset_250_250)
        
        if t % 100 == 0: # visualize the progress, making me FEEL GOOD!!
            print("t = ", t,"in ", origin_dataset_28_28.shape[0])