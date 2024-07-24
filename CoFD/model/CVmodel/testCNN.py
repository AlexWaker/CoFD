import torch.nn as nn
#from sklearn.model_selection import train_test_split

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__() ##父类self参数初始化
        # Convolution 1 , input_shape=(1,28,28), output_shape=(16,24,24)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        # activation
        self.relu1 = nn.ReLU()
        # Max pool 1, output_shape=(16,12,12)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) 
        # Convolution 2, output_shape=(32,8,8)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        # activation
        self.relu2 = nn.ReLU() 
        # Max pool 2, output_shape=(32,4,4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1, input_shape=(32*4*4)
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2 
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        return out