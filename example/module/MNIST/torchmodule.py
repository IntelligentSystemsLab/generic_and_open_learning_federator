# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/1/1 19:28
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/1/1 19:28

import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=torch.nn.Flatten()
        self.ln1=torch.nn.Linear(6400, 256)
        self.ln2 =torch.nn.Linear(256, 10)


    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.flatten(x)
        x=self.ln1(x)
        output=self.ln2(x)
        return output



model = 'Net'
optimizer = torch.optim.SGD
learning_rate=0.03
loss = torch.nn.CrossEntropyLoss()
batch_size = 128
train_epoch = 1
library = 'torch'
metrics = ["accuracy"]
