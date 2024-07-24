import torchvision
from torch.utils.data import Dataset, DataLoader
import gzip
import os
import numpy as np

##mnist和emnist处理方法一样一样的

def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)

##第一步：重写dataset类
class DealMnistDataset(Dataset):
    """
        读取数据、初始化数据
    """
    #这个类就是
    def __init__(self, folder, data_name, label_name, transform=None):
        (train_set, train_labels) = load_data(folder, data_name,
                                              label_name)  # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)


# #************************************a2torchloadlocalminist*********************************************************
if __name__ == '__main__':
# train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
# test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    train_dataset = DealMnistDataset(r'/home/gaopeijie/project/FD/myfd/dataset/CVdataset/mnist/gz', "train-images-idx3-ubyte.gz",
                           "train-labels-idx1-ubyte.gz", transform=torchvision.transforms.ToTensor())
    test_dataset = DealMnistDataset(r'/home/gaopeijie/project/FD/myfd/dataset/CVdataset/mnist/gz', "t10k-images-idx3-ubyte.gz",
                           "t10k-labels-idx1-ubyte.gz", transform=torchvision.transforms.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)
    print(1)
    images, labels = next(iter(train_loader))
    img = torchvision.utils.make_grid(images)

    img = img.numpy().transpose(1, 2, 0)
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img * std + mean
    print(labels)