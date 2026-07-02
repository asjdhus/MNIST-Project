import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    """
    MNIST手写数字识别模型
    简单的卷积神经网络架构
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        # 第一层卷积：输入通道1（灰度图），输出通道32，卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 第二层卷积：输入通道32，输出通道64，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 第一层池化：2x2最大池化
        self.pool = nn.MaxPool2d(2, 2)
        # 第一层全连接：输入特征数64*7*7，输出128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 第二层全连接：输入128，输出10（10个数字类别）
        self.fc2 = nn.Linear(128, 10)
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 卷积层1 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积层2 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))
        # 将特征图展平为一维向量
        x = x.view(-1, 64 * 7 * 7)
        # 全连接层1 + ReLU + Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # 全连接层2输出
        x = self.fc2(x)
        return x
