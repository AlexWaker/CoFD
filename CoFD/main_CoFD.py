import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from tools.mnist_gz_dealdataset import DealMnistDataset
from torch.utils.data import DataLoader # type: ignore
from tools.dilikelei import noniid
from tools.option_win5 import args_parser
from tools.fed import FedAvg
# from models.Update import LocalUpdate
from model.CVmodel.simplemodel import MLP, CNNMnist, DeepNet_Cifar, resnet18, resnet34
from train_model.update import LocalUpdate
from tools.suijichouqu import tiqu_common_ziji
from tools.gouzi import jiagouzi
from train_model.zhengliutrain import zhengliutrainer
import tools.logits_agg as loag
import logging
from server.tpsserver import mytpsserver
# from models.Fed import FedAvg
# from models.test import test_img




def initial_model(num_users, set_name, logger):
    user_model = []
    
    if set_name == 'mnist' or set_name == 'emnist':
        for i in range(num_users):
            suijishu = np.random.random()
            if suijishu > 0.5:
                len_in = 1
                for x in img_size:
                    len_in *= x
                user_model.append(MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device))
                logger.debug('In %s client %s model is MLP', set_name, i)
            else:
                user_model.append(CNNMnist(args=args).to(args.device))
                logger.debug('In %s client %s model is CNNMnist', set_name, i)
    
    elif set_name == 'cifar10':
        for i in range(num_users):
            suijishu = np.random.random()
            if suijishu > 0.5:
                user_model.append(DeepNet_Cifar(10))
                logger.debug('In %s client %s model is deepcifar', set_name, i)
            else:
                user_model.append(resnet18(10))
                logger.debug('In %s client %s model is resnet18', set_name, i)
    elif set_name == 'cifar100':
        for i in range(num_users):
            suijishu = np.random.random()
            if suijishu > 0.5:
                user_model.append(DeepNet_Cifar(100))
                logger.debug('In %s client %s model is DeepNet_Cifar', set_name, i)
            else:
                user_model.append(resnet18(100))
                logger.debug('In %s client %s model is resnet18', set_name, i)
    
    return user_model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # print(f'Accuracy of the model on the test images: {100 * correct / total:.4f}%')
    return 100 * correct / total


if __name__ == '__main__':
    # 公共集比例，客户端，alpha，localepoch，数据集 轮数
    logger = logging.getLogger('cofd_log')
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建日志处理器
    file_handler = logging.FileHandler('cofd_log.log')
    stream_handler = logging.StreamHandler()

    # 创建并设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # parse args
    args = args_parser()
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # args.device = torch.device('cpu')
    args.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    logger.debug(args.device)
    # load dataset and split users
    # if args.dataset == 'all':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset_mnist_train = DealMnistDataset(r'/home/gaopeijie/project/FD/myfd/dataset/CVdataset/mnist/gz', "train-images-idx3-ubyte.gz",
    #                     "train-labels-idx1-ubyte.gz", transform=trans_mnist)
    # dataset_mnist_test = DealMnistDataset(r'/home/gaopeijie/project/FD/myfd/dataset/CVdataset/mnist/gz', "t10k-images-idx3-ubyte.gz",
    #                     "t10k-labels-idx1-ubyte.gz", transform=trans_mnist)
    # dataset_cifar10_train = DealMnistDataset(r'/home/gaopeijie/project/FD/myfd/dataset/CVdataset/mnist/gz', "train-images-idx3-ubyte.gz",
    #                     "train-labels-idx1-ubyte.gz", transform=trans_mnist)
    # dataset_cifar10_test = DealMnistDataset(r'/home/gaopeijie/project/FD/myfd/dataset/CVdataset/mnist/gz', "t10k-images-idx3-ubyte.gz",
    #                     "t10k-labels-idx1-ubyte.gz", transform=trans_mnist)
    # dataset_mnist_train = DealMnistDataset(r'/home/gaopeijie/project/FD/myfd/dataset/CVdataset/mnist/gz', "train-images-idx3-ubyte.gz",
    #                     "train-labels-idx1-ubyte.gz", transform=trans_mnist)
    # dataset_mnist_test = DealMnistDataset(r'/home/gaopeijie/project/FD/myfd/dataset/CVdataset/mnist/gz', "t10k-images-idx3-ubyte.gz",
    #                     "t10k-labels-idx1-ubyte.gz", transform=trans_mnist)
    
    dataset_cifar10_train = datasets.CIFAR10('./dataset/CVdataset/cifar10', train=True, download=False, transform=trans_cifar)
    dataset_cifar10_test = datasets.CIFAR10('./dataset/CVdataset/cifar10', train=False, download=False, transform=trans_cifar)
    
    # dataset_cifar100_train = datasets.CIFAR100('/home/gaopeijie/project/FD/myfd/dataset/CVdataset/cifar100', train=True, download=False, transform=trans_cifar)
    # dataset_cifar100_test = datasets.CIFAR100('/home/gaopeijie/project/FD/myfd/dataset/CVdataset/cifar100', train=False, download=False, transform=trans_cifar)

    img_size = dataset_cifar10_train[0][0].shape

    logger.info("Complete data set loading")
    #以上将数据集划分为non-iid的

    # 接下来客户端进行本地训练
    # 先初始化各个客户端上的异构模型
    # 可以初始化多种模型，此时先初始化两种
    mnist_user_model = initial_model(args.num_users, 'cifar10', logger)
    # mnist_user_model = initial_model(args.num_users, 'mnist', logger)
    # cifar10_user_model = initial_model(args.num_users, 'cifar10', logger)
    # cifar100_user_model = initial_model(args.num_users, 'cifar100', logger)
    
    logger.info("Complete initialization of all models")

    common_mnist_indices = tiqu_common_ziji(dataset_cifar10_train, int(args.common_proportion * len(dataset_cifar10_train))) # common_dataset才是真数据集
    logger.debug('mnist common number %s', len(common_mnist_indices))
    # common_mnist_indices = tiqu_common_ziji(dataset_mnist_train, int(args.common_proportion * len(dataset_mnist_train)))
    # logger.debug('mnist common number %s', len(common_mnist_indices))
    # common_cifar10_indices = tiqu_common_ziji(dataset_cifar10_train, int(args.common_proportion * len(dataset_cifar10_train)))
    # logger.debug('cifar10 common number %s', len(common_cifar10_indices))
    # common_cifar100_indices = tiqu_common_ziji(dataset_cifar100_train, int(args.common_proportion * len(dataset_cifar100_train)))
    # logger.debug('cifar100 common number %s', len(common_cifar100_indices))
    # logger.info("Extracting the common subset is completed")

    if args.iid:
        logger.debug('IID')
        mnist_users = noniid(dataset_cifar10_train, args.num_users, 999999, 'cifar10', 10, common_mnist_indices, logger)
        # mnist_users = noniid(dataset_mnist_train, args.num_users, 999999, 'mnist', 10, common_mnist_indices, logger)
        # cifar10_users = noniid(dataset_mnist_train, args.num_users, 999999, 'cifar10', 10, common_mnist_indices, logger)
        # cifar100_users = noniid(dataset_mnist_train, args.num_users, 999999, 'cifar100', 10, common_mnist_indices, logger)
    else:
        logger.debug('Non-iid and alpha is %s', args.alpha)
        mnist_users = noniid(dataset_cifar10_train, args.num_users, args.alpha, 'cifar10', 10, common_mnist_indices, logger)
        # logger.debug('emnist common number ', len(emnist_users))
        # mnist_users = noniid(dataset_mnist_train, args.num_users, args.alpha, 'mnist', 10, common_mnist_indices, logger)
        # cifar10_users = noniid(dataset_mnist_train, args.num_users, args.alpha, 'mnist', 10, common_mnist_indices, logger)
        # cifar100_users = noniid(dataset_mnist_train, args.num_users, args.alpha, 'mnist', 100, common_mnist_indices, logger)
    
    logger.error('Begin federated learning!')
    logger.info('First MNIST')
    round_num = 50
    cifar10server = mytpsserver(args, dataset_cifar10_train, common_mnist_indices)
        # id_logitsmatrix = [None] * args.num_users
    zhengliuqi = [zhengliutrainer(args, u, mnist_user_model[u], dataset_cifar10_train, common_mnist_indices) for u in range(args.num_users)]
    # emnistserver = mytpsserver()
    for r in range(round_num):

        logger.debug('%s round', r)
        # 初始化服务器和矩阵
        # emnistserver = mytpsserver(args, r, dataset_emnist_train, common_emnist_indices)
        # id_logitsmatrix = [None] * args.num_users
        zhengliuqi = [zhengliutrainer(args, u, mnist_user_model[u], dataset_cifar10_train, 
        common_mnist_indices) for u in range(args.num_users)]
        for u in range(args.num_users):
            local = LocalUpdate(args, dataset_cifar10_train, mnist_users[u], common_mnist_indices) #idxs是私有数据及的索引
            w, loss = local.train(net=mnist_user_model[u].to(args.device), logger=logger)
            logger.info('%s client finished %s epoch local training and loss is %s', u, args.local_ep, loss)
            # zhengliu = zhengliutrainer(args, u, emnist_user_model[u], dataset_emnist_train, common_emnist_indices)
            logitsmatrix = zhengliuqi[u].predict_comset_logits()
            rightnum, rightbutwrong = cifar10server.receive_logits_matrix(u, logitsmatrix)
            # rightnum = emnistserver.fenbu[u].alpha
            logger.info('%s client finished predicting common logits and send to server and right number is %s/%s, and rightbutwrong is %s', u, rightnum, len(common_mnist_indices), rightbutwrong)
    
        # 上传服务器做聚合
        teacher_logits, final_right_number = cifar10server.send_teacher_matrix()

        logger.debug('round %s server finish aggregation and final right number is %s', r, final_right_number)

        # for u in range(args.num_users):
        #     zhengliuqi[u].localzhengliu(teacher_matrix = teacher_logits)
        #     logger.info('%s client finish local distillation', u)
            
        #     #测试准确率
        #     loader = DataLoader(dataset_cifar10_test, batch_size=32, shuffle=True)
        #     test_acc = evaluate_model(mnist_user_model[u], loader, args.device)
        #     logger.info('In round %s client %s acc is %s', r, u, test_acc)

        acc = []
        for u in range(args.num_users):
            zhengliuqi[u].localzhengliu(teacher_matrix = teacher_logits)
            logger.info('%s client finish local distillation', u)
            
            #测试准确率
            loader = DataLoader(dataset_cifar10_test, batch_size=32, shuffle=True)
            test_acc = evaluate_model(mnist_user_model[u], loader, args.device)
            logger.info('In round %s client %s acc is %s', r, u, test_acc)
            acc.append(test_acc)
        avgacc = sum(acc)/args.num_users
        logger.info('In round %s avgacc is %s', r, avgacc)







    #完成客户端各自训练，进行联邦蒸馏
    #提取出公共子集
    # zijishuliang = 5000
    # common_ziji_indices = tiqu_common_ziji(common_dataset, zijishuliang) # common_dataset才是真数据集
    # id_logitsmatrix = empty_array = [None] * args.num_users
    
    
    '''
    开始聚合
    '''
    # res = loag.avg_agg(id_logitsmatrix, common_ziji_indices, 10, args.device)
    # # print(res)

    # '''
    # 聚合后开始知识蒸馏
    # '''
    # zhengliu.localzhengliu(res)

    # print('finish')





