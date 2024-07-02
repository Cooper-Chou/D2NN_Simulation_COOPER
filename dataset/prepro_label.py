import numpy as np

# mnist_data = np.load('mnist.npy', allow_pickle=True)
mnist_data = np.load('dataset/MNIST/raw/mnist.npy', allow_pickle=True)

# [0][0] for train_data, [0][1] for train_label
# [1][0] for validation_data, [1][1] for validation_label
# [2][0] for test_data, [2][1] for test_label

train_label = mnist_data[0][1]
validation_label = mnist_data[1][1]
test_label = mnist_data[2][1]

# train_label = np.load('./data2/train_label.npy', allow_pickle=True)
# validation_label = np.load('./data2/validation_label.npy', allow_pickle=True)
# test_label = np.load('./data2/test_label.npy', allow_pickle=True)

print("train_label.shape = ", train_label.shape)
print("validation_label.shape = ", validation_label.shape)
print("test_label.shape = ", test_label.shape)

# ****_mod stands for modified
train_label_mod = np.zeros([10,train_label.shape[0]])
validation_label_mod = np.zeros([10,validation_label.shape[0]])
test_label_mod = np.zeros([10,test_label.shape[0]])

for i in range(train_label.shape[0]):
    train_label_mod[train_label[i],i] = 1

for i in range((test_label.shape[0])):
    test_label_mod[test_label[i],i] = 1

for i in range(validation_label.shape[0]):
    validation_label_mod[validation_label[i],i] = 1

np.save('processed_dataset/train_label_all.npy', train_label_mod)
np.save('processed_dataset/validation_label_all.npy', validation_label_mod)
np.save('processed_dataset/test_label_all.npy', test_label_mod)