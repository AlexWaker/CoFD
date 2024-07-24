import numpy as np
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

def calculate_entropy(tensor):
    # 将张量转换为概率分布
    tensor = tensor.float()  # 转换为浮点数类型
    value_counts = torch.bincount(tensor, minlength=tensor.max().int() + 1).float()
    probabilities = value_counts / len(tensor)

    # 过滤零概率
    probabilities = probabilities[probabilities > 0]

    # 计算熵
    entropy = -torch.sum(probabilities * torch.log(probabilities))
    return entropy.item()

def rightornot(logits, real_label):
    np_array = np.argmax(np.array(logits.tolist()))
    first_value = real_label.flatten()[0]
    if np_array == first_value:
        return True
    else:
        return False

class everyfenbu():
    def __init__(self, alpha, beta):
        # self.n_arms = n_arms
        self.alpha = alpha  # 初始化所有臂的 alpha 参数 成功
        self.beta = beta   # 初始化所有臂的 beta 参数 失败
    
    def update(self, reward):
        # 根据实际奖励更新 alpha 和 beta 参数
        self.alpha += reward
        self.beta += (1 - reward)
        

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class mytpsserver():
    def __init__(self, args, common_set, common_indices):
        self.args = args
        self.fenbu = [[everyfenbu(1, 9) for _ in range(len(common_set))] for _ in range(args.num_users)]
        self.now_round = 0
        self.common_set = common_set
        self.common_indices = common_indices
        self.set_loader = DataLoader(DatasetSplit(self.common_set, self.common_indices), batch_size=1, shuffle=False)
        # self.hiweight = [[0 for _ in range(len(common_set))] for _ in range(args.num_users)]
        self.hiweight = np.array([[0. for _ in range(args.num_users)] for _ in range(len(common_set))])
        self.enweight = [[0 for _ in range(len(common_set))] for _ in range(args.num_users)]
        self.flag = [[False for _ in range(len(common_set))] for _ in range(args.num_users)]
        self.all_logits = {}
        self.this_round_users = []
        self.all_round_users = {}

    def roundadd(self):
        self.now_round += 1

    def calculate_entropy_1d(self, tensor):
    # 将张量转换为概率分布（归一化）
        # probabilities = tensor / tensor.sum()

    # 计算熵，添加1e-9以避免log(0)
        entropy = -torch.sum(tensor * torch.log(tensor + 1e-9))

        return entropy.item()    

    def receive_logits_matrix(self, user_id, logits):
        rightnum = 0
        self.this_round_users.append(user_id)
        self.all_logits[user_id] = logits
        rightbutwrong = 0
        for i, (images, labels) in enumerate(self.set_loader):
            # print(logits[self.common_indices[i]])
            # print(labels)
            if rightornot(logits[self.common_indices[i]], labels):
                self.fenbu[user_id][self.common_indices[i]].update(1)
                self.flag[user_id][self.common_indices[i]] = True
                rightnum += 1
            else:
                self.fenbu[user_id][self.common_indices[i]].update(0)
                if self.flag[user_id][self.common_indices[i]]: #之前对但这次错了
                    rightbutwrong += 1
                self.flag[user_id][self.common_indices[i]] = False
        return rightnum, rightbutwrong
    
    def calculate_entropy_from_logits(self, logits):
    # 将logits转换为概率分布
        probabilities = F.softmax(logits, dim=-1)
    
    # 计算每个元素的负对数
        log_probabilities = F.log_softmax(logits, dim=-1)
    
    # 计算熵
        entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
    
        return entropy

    def weight_agg(self):
        final_right_number = 0
        avg_shang = 0.0
        teacher_logits = {}
        for i, (images, labels) in enumerate(self.set_loader):
            teacher = torch.zeros(1, self.args.num_classes).to(self.args.device)
            for j in range(len(self.this_round_users)):
                # teacher = teacher + (self.hiweight[self.this_round_users[j]][self.common_indices[i]] + 1 / self.enweight[self.this_round_users[j]][self.common_indices[i]]) * self.all_logits[self.this_round_users[j]][self.common_indices[i]]
                # print(self.hiweight[self.this_round_users[j]][self.common_indices[i]])
                # print(self.all_logits[self.this_round_users[j]][self.common_indices[i]])
                # teacher = teacher + (self.hiweight[self.this_round_users[j]][self.common_indices[i]]) * self.all_logits[self.this_round_users[j]][self.common_indices[i]]
                t = F.softmax(self.all_logits[self.this_round_users[j]][self.common_indices[i]], dim=1)
                teacher = teacher + (self.hiweight[self.common_indices[i]][self.this_round_users[j]]) * t
            # teacher = teacher / len(self.this_round_users)
            # teacher = F.softmax(teacher, dim=-1)
            teacher_logits[self.common_indices[i]] = teacher
            entropy = self.calculate_entropy_1d(teacher)
            avg_shang += entropy
            if rightornot(teacher, labels):
                final_right_number += 1
    
        return teacher_logits, final_right_number, avg_shang / (len(self.this_round_users) * len(self.common_indices))



    def send_teacher_matrix(self):
        '''
        先抽样，两个权重，然后发送
        '''
        self.hiweight.fill(0.)
        self.all_round_users[self.now_round] = self.this_round_users
        for i in range(len(self.common_indices)):
            for j in range(len(self.this_round_users)):
                #抽五次
                samples = np.random.beta(self.fenbu[self.this_round_users[j]][self.common_indices[i]].alpha, self.fenbu[self.this_round_users[j]][self.common_indices[i]].beta, size=10)
                hiwei = sum(samples)/len(samples)
                self.hiweight[self.common_indices[i]][self.this_round_users[j]] = hiwei
                # if self.flag[i][self.common_indices[j]]:
                #     self.enweight[i][self.common_indices[j]] = self.calculate_entropy_from_logits(self.all_logits[i][self.common_indices[j]])
                # else:
                #     self.enweight[i][self.common_indices[j]] = 99999
            # print(self.hiweight[self.common_indices[i]])
            temp = sum(self.hiweight[self.common_indices[i]])
            # print(temp)
            for k in range(len(self.this_round_users)):
                self.hiweight[self.common_indices[i]][self.this_round_users[k]] /= temp
        teacher_logits, final_right_number, avg_shang = self.weight_agg()
        self.roundadd()
        self.this_round_users = []
        return teacher_logits, final_right_number, avg_shang

