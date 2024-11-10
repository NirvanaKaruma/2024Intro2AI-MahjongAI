"""
示例用神经网络，包括三层卷积和两层全连接。

提示：需要修改这个类，从而实现：
1. 表达能力更强的神经网络。
"""
import torch
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv_identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if identity.shape[1] != out.shape[1]:
            identity = self.conv_identity(identity)
        out += identity
        out = self.relu(out)
        return out
    

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.features = nn.Sequential(
            ResidualBlock(147, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.Flatten()
        )

        self.fc_adjust = nn.Linear(64 * 4 * 9, 1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc_hidden1 = nn.Linear(1024, 512)
        self.policy = nn.Linear(512, 235)

    def forward(self, x):
        x = self.features(x)
        x = self.fc_adjust(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc_hidden1(x)
        policy_logits = self.policy(x)
        return policy_logits

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.value_net = ValueNet()

    def forward(self, input_dict):
        self.train(mode=input_dict.get("is_training", False))
        obs = input_dict["obs"]["observation"].float()
        policy_logits= self.value_net(obs)
        action_mask = input_dict["obs"]["action_mask"].float()
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        return policy_logits + inf_mask