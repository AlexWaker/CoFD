import torchvision
from torch.utils.data import Dataset, DataLoader
import gzip
import os
import numpy as np
from torchvision import datasets
# import torchvision.transforms as transform
from torch.utils.data.sampler import SubsetRandomSampler
# import tensorflow as tf
# import numpy as np
# import math
# # from six.moves import cPickle as pickle
# import os
# import platform
# from subprocess import check_output
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# def load_pickle(f):
#     version = platform.python_version_tuple()
#     if version[0] == '2':
#         return  pickle.load(f)
#     elif version[0] == '3':
#         return  pickle.load(f, encoding='latin1')
#     raise ValueError("invalid python version: {}".format(version))

# def load_CIFAR_batch(filename):
#     """ load single batch of cifar """
#     with open(filename, 'rb') as f:
#         datadict = load_pickle(f)
#         X = datadict['data']
#         Y = datadict['labels']
#         X = X.reshape(10000,3072)
#         Y = np.array(Y)
#         return X, Y

# def load_CIFAR10(ROOT):
#     """ load all of cifar """
#     xs = []
#     ys = []
#     for b in range(1,6):
#         f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
#         X, Y = load_CIFAR_batch(f)
#         xs.append(X)
#         ys.append(Y)
#     Xtr = np.concatenate(xs)
#     Ytr = np.concatenate(ys)
#     del X, Y
#     Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
#     return Xtr, Ytr, Xte, Yte
    
# def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
#     # Load the raw CIFAR-10 data
#     cifar10_dir = '../input/cifar-10-batches-py/'
#     X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

#     # Subsample the data
#     mask = range(num_training, num_training + num_validation)
#     X_val = X_train[mask]
#     y_val = y_train[mask]
#     mask = range(num_training)
#     X_train = X_train[mask]
#     y_train = y_train[mask]
#     mask = range(num_test)
#     X_test = X_test[mask]
#     y_test = y_test[mask]

#     x_train = X_train.astype('float32')
#     x_test = X_test.astype('float32')

#     x_train /= 255
#     x_test /= 255

#     return x_train, y_train, X_val, y_val, x_test, y_test


# # Invoke the above function to get our data.
# x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()


# print('Train data shape: ', x_train.shape)
# print('Train labels shape: ', y_train.shape)
# print('Validation data shape: ', x_val.shape)
# print('Validation labels shape: ', y_val.shape)
# print('Test data shape: ', x_test.shape)
# print('Test labels shape: ', y_test.shape)
def load_data(folder, transform):
    train_data = datasets.CIFAR10(folder, train=True, download=False, transform=transform)
    return train_data

class DealCifarDataset(Dataset):
    """
        读取数据、初始化数据
    """
    def __init__(self, folder, transform=None):
        
        self.transform = transform
        self.train_data = load_data(folder, transform)  
        self.cifar10_path = folder
        self.train_batchs = [
            self.cifar10_path + 'data_batch_1',
            self.cifar10_path + 'data_batch_2',
            self.cifar10_path + 'data_batch_3',
            self.cifar10_path + 'data_batch_4',
            self.cifar10_path + 'data_batch_5'
        ]
        self.test_batchs = [self.cifar10_path + 'test_batch']

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

print(torchvision.__file__)