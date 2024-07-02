import gzip
import numpy as np
import matplotlib.pyplot as plt 
import random

dataset_name = ['MNIST','mnist']  # ['FASHION','fashion'] # 

dir_src = 'dataset/'+dataset_name[0]+'/' 

with gzip.open(dir_src+'train-images-idx3-ubyte.gz') as all_data_ubyte_1:
    all_data_ubyte_1 = all_data_ubyte_1.read()

with gzip.open(dir_src+'t10k-images-idx3-ubyte.gz') as all_data_ubyte_2:
    all_data_ubyte_2 = all_data_ubyte_2.read()

with gzip.open(dir_src+'train-labels-idx1-ubyte.gz') as all_label_ubyte_1:
    all_label_ubyte_1 = all_label_ubyte_1.read()

with gzip.open(dir_src+'t10k-labels-idx1-ubyte.gz') as all_label_ubyte_2:
    all_label_ubyte_2 = all_label_ubyte_2.read()

all_data_ubyte = all_data_ubyte_1[16:] + all_data_ubyte_2[16:] # 这里已经把开头的内容去掉了, 所以数组里完全就是数据本身, 长度为 28*28*60000+28*28*10000
all_label_ubyte = all_label_ubyte_1[8:] + all_label_ubyte_2[8:]
dataset_length = int.from_bytes(all_data_ubyte_1[4:4+4], byteorder='big', signed=False) + int.from_bytes(all_data_ubyte_2[4:4+4], byteorder='big', signed=False)

# print((len(all_data_ubyte_1)-16)/784, (len(all_data_ubyte_2)-16)/784, len(all_label_ubyte_1)-8, len(all_label_ubyte_2)-8)
# print((len(all_data_ubyte_1+all_data_ubyte_2)-2*16)/784)
# print(int.from_bytes(all_data_ubyte_1[4:4+4], byteorder='big', signed=True), int.from_bytes(all_data_ubyte_2[4:4+4], byteorder='big', signed=False), int.from_bytes(all_label_ubyte_1[4:4+4], byteorder='big', signed=False), int.from_bytes(all_label_ubyte_2[4:4+4], byteorder='big', signed=False))


all_data_float32 = np.zeros(shape=(dataset_length,28,28), dtype=np.float32)
all_label_int32 = np.zeros(shape=(dataset_length), dtype=np.int32)

for i in range(dataset_length):
    temp_data_ubyte = all_data_ubyte[i*784:(i+1)*784]
    temp_data_float32 = []
    for j in range(28*28):
            temp_data_float32.append(temp_data_ubyte[j]) # 此处 all_data_ubyte 本来是 bytes 类型, 但是 append 之后就变成了 int 类型, 我也不知道为啥
    all_data_float32[i,:,:] = np.array(temp_data_float32, dtype=np.float32).reshape(28, 28)
    all_label_int32[i] = np.array(all_label_ubyte[i], dtype=np.int32)


# 打乱原数据集的顺序, 并且保存为 (3,2) 的 npy 文件
# {其实没必要打乱? 因为训练的时候可以在 dataloader 里面设置 shuffle}
# (不过为了 train,vali,test 里面的数据都分布得更加均匀, 还是打乱了)
rand_indice = np.array(range(dataset_length))
random.shuffle(rand_indice)

data_train = np.zeros((50000, 28, 28), dtype=np.float32)
label_train = np.zeros((50000), dtype=np.int32)
data_vali = np.zeros((10000, 28, 28), dtype=np.float32)
label_vali = np.zeros((10000), dtype=np.int32)
data_test = np.zeros((10000, 28, 28), dtype=np.float32)
label_test = np.zeros((10000), dtype=np.int32)

for idx,rand_idx in enumerate(rand_indice):
    if idx < 50000:
         data_train[idx,:,:] = all_data_float32[rand_idx,:,:]
         label_train[idx] = all_label_int32[rand_idx]
    elif idx < 60000:
        data_vali[idx-50000,:,:] = all_data_float32[rand_idx,:,:]
        label_vali[idx-50000] = all_label_int32[rand_idx]
    else:
        data_test[idx-60000,:,:] = all_data_float32[rand_idx,:,:]
        label_test[idx-60000] = all_label_int32[rand_idx]

# plt.imshow(data_train[1530,:,:])
# plt.show()

final_npy = np.array([[data_train, label_train], [data_vali, label_vali], [data_test, label_test]],dtype=object)

np.save(dir_src+dataset_name[1]+'.npy', final_npy)

print(final_npy.shape, final_npy[0][0].shape, final_npy[0][1].shape, final_npy[1][0].shape, final_npy[1][1].shape, final_npy[2][0].shape, final_npy[2][1].shape)

"""
There are 4 files:

train-images-idx3-ubyte: training set images 
train-labels-idx1-ubyte: training set labels 
t10k-images-idx3-ubyte:  test set images 
t10k-labels-idx1-ubyte:  test set labels

The training set contains 60000 examples, and the test set 10000 examples.

The first 5000 examples of the test set are taken from the original NIST training set. The last 5000 are taken from the original NIST test set. The first 5000 are cleaner and easier than the last 5000.
-------------------------------------------------------------------------------------------------
TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  60000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
-------------------------------------------------------------------------------------------------
TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  60000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
-------------------------------------------------------------------------------------------------
TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
0004     32 bit integer  10000            number of items 
0008     unsigned byte   ??               label 
0009     unsigned byte   ??               label 
........ 
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
-------------------------------------------------------------------------------------------------
TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
[offset] [type]          [value]          [description] 
0000     32 bit integer  0x00000803(2051) magic number 
0004     32 bit integer  10000            number of images 
0008     32 bit integer  28               number of rows 
0012     32 bit integer  28               number of columns 
0016     unsigned byte   ??               pixel 
0017     unsigned byte   ??               pixel 
........ 
xxxx     unsigned byte   ??               pixel
Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 
"""
