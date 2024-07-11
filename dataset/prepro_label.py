import numpy as np

dataset_name = ['MNIST','mnist'] # ['FASHION','fashion'] # 
dataset_dir = 'dataset/'+dataset_name[0]+'_processed_0_5/'

dataset_npy = np.load('dataset/'+dataset_name[0]+'/'+dataset_name[1]+'.npy', allow_pickle=True)

# [0][0] for train_data, [0][1] for train_label
# [1][0] for validation_data, [1][1] for validation_label
# [2][0] for test_data, [2][1] for test_label

train_label = dataset_npy[0][1]
validation_label = dataset_npy[1][1]
test_label = dataset_npy[2][1]

# train_label = np.load('./data2/train_label.npy', allow_pickle=True)
# validation_label = np.load('./data2/validation_label.npy', allow_pickle=True)
# test_label = np.load('./data2/test_label.npy', allow_pickle=True)

print("train_label.shape =", train_label.shape)
print("validation_label.shape =", validation_label.shape)
print("test_label.shape =", test_label.shape)

# ****_mod stands for modified
train_label_modified = np.zeros([10,train_label.shape[0]])
validation_label_modified = np.zeros([10,validation_label.shape[0]])
test_label_modified = np.zeros([10,test_label.shape[0]])

for i in range(train_label.shape[0]):
    train_label_modified[train_label[i],i] = 1

for i in range((test_label.shape[0])):
    test_label_modified[test_label[i],i] = 1

for i in range(validation_label.shape[0]):
    validation_label_modified[validation_label[i],i] = 1

np.save(dataset_dir+'label_train_all.npy', train_label_modified)
np.save(dataset_dir+'label_validation_all.npy', validation_label_modified)
np.save(dataset_dir+'label_test_all.npy', test_label_modified)