# -*- coding: utf-8 -*-
# @Author  : Dengxun
# @Time    : 2023/3/22 20:58
# @Function:AlexNetModel
import torch
from torch import nn
import torch.nn.functional as F

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4,
                            padding=2)  # (224 - 11 + 2*2) / 4 + 1 = 55
        self.ReLU = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1,
                            padding=2)  # (55 - 5 + 2*2) / 1 + 1 = 55
        self.s2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1,
                            padding=1)  # (27 - 3 + 2*1) / 1 + 1 = 27
        self.s3 = nn.MaxPool2d(2)
        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1,
                            padding=1)  # (13 - 3 + 2*1) / 1 + 1 = 13
        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1,
                            padding=1)  # (13 - 3 + 2*1) / 1 + 1 = 13
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(4608, 2048)
        self.f7 = nn.Linear(2048, 2048)
        self.f8 = nn.Linear(2048, 1000)
        # 为满足该实例另加 ↓
        self.f9 = nn.Linear(1000, 10)

    def forward(self, x):  # 输入shape: torch.Size([1, 3, 224, 224])
        x = self.ReLU(self.c1(x))  # shape: torch.Size([1, 48, 55, 55])
        x = self.ReLU(self.c2(x))  # shape: torch.Size([1, 128, 55, 55])
        x = self.s2(x)  # shape: torch.Size([1, 128, 27, 27])
        x = self.ReLU(self.c3(x))  # shape: torch.Size([1, 192, 27, 27])
        x = self.s3(x)  # shape: torch.Size([1, 192, 13, 13])
        x = self.ReLU(self.c4(x))  # shape: torch.Size([1, 192, 13, 13])
        x = self.ReLU(self.c5(x))  # shape: torch.Size([1, 128, 13, 13])
        x = self.s5(x)  # shape: torch.Size([1, 128, 6, 6])
        x = self.flatten(x)  # shape: torch.Size([1, 4608])
        x = self.f6(x)  # shape: torch.Size([1, 2048])
        x = F.dropout(x, p=0.5)  # shape: torch.Size([1, 2048])
        x = self.f7(x)  # shape: torch.Size([1, 2048])
        #x = F.dropout(x, p=0.5)  # shape: torch.Size([1, 2048])
        x = self.f8(x)  # shape: torch.Size([1, 1000])
        x = F.dropout(x, p=0.5)  # shape: torch.Size([1, 1000])
        # 为满足该实例另加 ↓（仍然使用 dropout 防止过拟合）
        x = self.f9(x)  # shape: torch.Size([1, 10])
        #x = F.dropout(x, p=0.5)  # shape: torch.Size([1, 10])
        return x

