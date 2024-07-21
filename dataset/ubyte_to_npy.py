import gzip
import numpy as np
import matplotlib.pyplot as plt

dataset_name = ['FASHION','fashion'] # ['MNIST','mnist']  # 

dir_src = 'dataset/'+dataset_name[0]+'/' 

with gzip.open(dir_src+'train-images-idx3-ubyte.gz') as train_data_file:
    train_data_file = train_data_file.read()

with gzip.open(dir_src+'t10k-images-idx3-ubyte.gz') as test_data_file:
    test_data_file = test_data_file.read()

with gzip.open(dir_src+'train-labels-idx1-ubyte.gz') as train_label_file:
    train_label_file = train_label_file.read()

with gzip.open(dir_src+'t10k-labels-idx1-ubyte.gz') as test_label_file:
    test_label_file = test_label_file.read()



# 这里已经把开头的内容去掉了, 所以数组里完全就是数据本身, 长度为 28*28*60000 和 28*28*10000 
train_data_ubyte = train_data_file[16:] 
train_dataset_length = int.from_bytes(train_data_file[4:4+4], byteorder='big', signed=False)
train_label_ubyte = train_label_file[8:]

test_data_ubyte = test_data_file[16:]
test_dataset_length = int.from_bytes(test_data_file[4:4+4], byteorder='big', signed=False)
test_label_ubyte = test_label_file[8:]

# print((len(all_data_ubyte_1)-16)/784, (len(all_data_ubyte_2)-16)/784, len(all_label_ubyte_1)-8, len(all_label_ubyte_2)-8)
# print((len(all_data_ubyte_1+all_data_ubyte_2)-2*16)/784)
# print(int.from_bytes(all_data_ubyte_1[4:4+4], byteorder='big', signed=True), int.from_bytes(all_data_ubyte_2[4:4+4], byteorder='big', signed=False), int.from_bytes(all_label_ubyte_1[4:4+4], byteorder='big', signed=False), int.from_bytes(all_label_ubyte_2[4:4+4], byteorder='big', signed=False))


train_data_float32 = np.zeros(shape=(train_dataset_length,28,28), dtype=np.float32)
train_label_int32 = np.zeros(shape=(train_dataset_length), dtype=np.int32)

for i in range(train_dataset_length):
    temp_data_ubyte = train_data_ubyte[i*784:(i+1)*784]
    temp_data_float32 = []
    for j in range(28*28):
        temp_data_float32.append(temp_data_ubyte[j]) # 此处 all_data_ubyte 本来是 bytes 类型, 但是 append 之后就变成了 int 类型, 我也不知道为啥
    train_data_float32[i,:,:] = np.array(temp_data_float32, dtype=np.float32).reshape(28, 28)
    train_label_int32[i] = np.array(train_label_ubyte[i], dtype=np.int32)


test_data_float32 = np.zeros(shape=(test_dataset_length,28,28), dtype=np.float32)
test_label_int32 = np.zeros(shape=(test_dataset_length), dtype=np.int32)

for i in range(test_dataset_length):
    temp_data_ubyte = test_data_ubyte[i*784:(i+1)*784]
    temp_data_float32 = []
    for j in range(28*28):
        temp_data_float32.append(temp_data_ubyte[j]) # 此处 all_data_ubyte 本来是 bytes 类型, 但是 append 之后就变成了 int 类型, 我也不知道为啥
    test_data_float32[i,:,:] = np.array(temp_data_float32, dtype=np.float32).reshape(28, 28)
    test_label_int32[i] = np.array(test_label_ubyte[i], dtype=np.int32)

# plt.imshow(data_train[1530,:,:])
# plt.show()

final_npy = np.array([[train_data_float32, train_label_int32], [test_data_float32, test_label_int32]], dtype=object)

np.save(dir_src+dataset_name[1]+'.npy', final_npy)

print(final_npy.shape, final_npy[0][0].shape, final_npy[0][1].shape, final_npy[1][0].shape, final_npy[1][1].shape)

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
