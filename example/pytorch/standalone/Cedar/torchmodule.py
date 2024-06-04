# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/1/1 19:28
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/1/1 19:28

import torch

optimizer = torch.optim.Adam
inner_learning_rate=0.001
outer_learning_rate=0.001
loss = torch.nn.CrossEntropyLoss()
batch_size = 10
train_epoch = 1
deploy_epoch = 3
divide_rate = 0.5
library = 'pytorch'
metrics = ["accuracy"]
