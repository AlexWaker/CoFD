import numpy as np
import torch
from torchvision import datasets, transforms

def tiqu_common_ziji(dataset, num_samples):
    # 加载MNIST数据集
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # 从数据加载器中获取所有数据
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # 随机抽取num_samples个样本
    indices = np.random.choice(len(images), num_samples, replace=False) #这个提取索引的本质跟dilikelei文件中的方法一样
    
    sampled_images = images[indices]
    sampled_labels = labels[indices]

    return indices.tolist()

# 调用函数并获取抽取的样本
# sampled_images, sampled_labels = load_mnist_and_sample()
