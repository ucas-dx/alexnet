# -*- coding: utf-8 -*-
# @Author  : Dengxun
# @Time    : 2023/3/22 21:32
# @Function:
import torch
from torch import nn
from AlexnetModel import MyAlexNet
import numpy as np
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from tqdm import tqdm
data_transform = transforms.Compose([
    transforms.Resize(224),  # 缩放图像大小为 224*224
    transforms.ToTensor()  # 仅对数据做转换为 tensor 格式操作
])

import torch
# # 限制0号设备的显存的使用量为0.5，就是半张卡那么多，比如12G卡，设置0.5就是6G。
torch.cuda.set_per_process_memory_fraction(1.0, 0)
torch.cuda.empty_cache()
# # 计算一下总内存有多少。
total_memory = torch.cuda.get_device_properties(0).total_memory
# # 使用0.9的显存:
tmp_tensor = torch.empty(int(total_memory * 0.8), dtype=torch.int8, device='cuda')





# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 给训练集创建一个数据集加载器
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
# 给测试集创建一个数据集加载器
test_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 如果显卡可用，则用显卡进行训练
device = "cuda" if torch.cuda.is_available() else 'cpu'

# 调用 net 里定义的模型，如果 GPU 可用则将模型转到 GPU
model = MyAlexNet().to(device)
# 定义损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()
# 定义优化器（SGD：随机梯度下降）
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
# 学习率每隔 10 个 epoch 变为原来的 0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)


# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in (enumerate(dataloader)):
        # 单通道转为三通道
        X = np.array(X)
        X = X.transpose((1, 0, 2, 3))  # array 转置
        image = np.concatenate((X, X, X), axis=0)
        image = image.transpose((1, 0, 2, 3))  # array 转置回来
        image = torch.tensor(image)  # 将 numpy 数据格式转为 tensor
        # 前向传播
        image, y = image.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]
        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
    print('train_loss：' + str(loss / n))
    print('train_acc：' + str(current / n))


# 定义测试函数
def test(dataloader, model, loss_fn):
    # 将模型转换为验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    # 非训练，推理期用到（测试时模型参数不用更新，所以 no_grad）
    with torch.no_grad():
        for batch, (X, y) in tqdm(enumerate(dataloader),desc='Processing'):
            # 单通道转为三通道
            X = np.array(X)
            X = X.transpose((1, 0, 2, 3))  # array 转置
            image = np.concatenate((X, X, X), axis=0)
            image = image.transpose((1, 0, 2, 3))  # array 转置回来
            image = torch.tensor(image)  # 将 numpy 数据格式转为 tensor
            image, y = image.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print('test_loss：' + str(loss / n))
        print('test_acc：' + str(current / n))



# 开始训练
epoch = 12
for t in tqdm(range(epoch)):
    #lr_scheduler.step()
    print(f"Epoch {t + 1}\n----------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    lr_scheduler.step()
    torch.save(model.state_dict(), r"E:\alexnet\model.pth")  # 模型保存
print("Done!")
