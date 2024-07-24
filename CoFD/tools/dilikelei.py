import numpy as np
import torch
from torchvision import datasets, transforms
from collections import Counter
import matplotlib.pyplot as plt
'''
提取索引的逻辑是什么？
本质还是顺序，原数据集将数据转化为tensor之后，提取顺序作为数据的索引作为数据的索引
'''
def load_mnist_data(train_set):
    # 加载MNIST数据集
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    
    # 正确的获取数据方式
    for images, labels in train_loader:
        return images.numpy(), labels.numpy() # 头一次见循环return的，C++不是return直接跳出函数吗

def generate_dirichlet_distribution(labels, num_clients, concentration, num_classes, common, logger, name):
    # 根据狄利克雷分布生成每个客户端的数据索引
    # num_classes = 10
    client_indices = [[] for _ in range(num_clients)]

    # 计算每个类别的索引
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    # 删掉公共集部分
    for i in range(len(class_indices)):
        # print(len(class_indices[i]))
        result = [item for item in class_indices[i] if item not in common]
        class_indices[i] = result
        # print(len(class_indices[i]))

    for i in range(num_classes):
        # 为每个类别生成狄利克雷分布
        distributions = np.random.dirichlet(np.ones(num_clients) * concentration) #每个客户端在当前类别样本中所占有的份额
        distributions *= len(class_indices[i])
        distributions = np.round(distributions).astype(int)

        # 修正可能因四舍五入出现的数量不匹配
        while np.sum(distributions) > len(class_indices[i]):
            distributions[np.argmax(distributions)] -= 1
        while np.sum(distributions) < len(class_indices[i]):
            distributions[np.argmin(distributions)] += 1

        # 分配索引到客户端
        np.random.shuffle(class_indices[i])
        start = 0
        for idx, amount in enumerate(distributions):
            client_indices[idx].extend(class_indices[i][start:start + amount])
            start += amount
            # print(len(client_indices[idx]))
    for j in range(len(client_indices)):
        lll = labels[client_indices[j]].tolist()
        element_counts = Counter(lll)
        logger.debug('In %s client %s has %s datas and their labels are %s', name, j, len(client_indices[j]), element_counts)
    return client_indices

def allocate_data_to_clients(images, labels, client_indices):
    # 根据索引分配数据和标签到客户端
    client_data = []
    for indices in client_indices:
        client_images = images[indices]
        client_labels = labels[indices]
        client_data.append((client_images, client_labels))
    return client_data

def plot_data_distribution(client_data_indices, labels, num, alpha, name):
    # 绘制每个客户端的数据分布图
    plt.figure(figsize=(14, 7))
    for i, indices in enumerate(client_data_indices):
        plt.subplot(4, 5, i + 1)
        plt.hist(labels[indices], bins=np.arange(11) - 0.5, rwidth=0.8)
        plt.title(f'Client {i + 1}')
        plt.xticks(np.arange(10))
    plt.tight_layout()
    plt.savefig('fenbu_{}_{}_{}.png'.format(num, alpha, name))
    plt.show()

def noniid(dataset_train, num_clients, concentration, name, num_classes, common, logger):

    # 加载数据
    images, labels = load_mnist_data(dataset_train)
    # 生成每个客户端的数据分布
    client_indices = generate_dirichlet_distribution(labels, num_clients, concentration, num_classes, common, logger, name)
    # plot_data_distribution(client_indices, labels, num_clients, concentration, name)
    # client_indices:20个客户端，每个客户端包含哪些数据，20个一维数组
    return client_indices
    # 分配数据到客户端
    #client_data = allocate_data_to_clients(images, labels, client_indices)
# 正儿八经的每个客户端所包含的图片的tensor

# 绘制分布图（根据标签分布）
#plot_data_distribution(client_indices, labels)
