# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2024/6/4 22:17
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2024/6/4 22:17
import os

import torch

from golf_federated.server.process.config.device.standalone import StandAloneCedarServer
from golf_federated.server.process.config.model.torchmodel import CedarServerModel
from golf_federated.server.process.strategy.selection.nonprobbased import AllSelect

from golf_federated.server.process.strategy.evaluation.classification import Accuracy
from torchvision import models

import torchmodule as module

from golf_federated.client.process.config.device.standalone import CedarClient
from golf_federated.client.process.config.model.torchmodel import CedarModel
from golf_federated.client.process.config.trainer.direct import CedarTrainer
from golf_federated.server.process.config.task.synchronous import CedarTask
from golf_federated.server.process.strategy.aggregation.synchronous import Cedar_syn

if __name__ == '__main__':
    train_path = '../../../data/non_iid_data/sfddd/' + 'train_client'
    test_path = '../../../data/non_iid_data/sfddd/' + 'test_client'
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path, i) for i in train_file_set]
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path, i) for i in test_file_set]
    train_clients = []
    test_clients = []
    client_id = []
    for index, path in enumerate(train_path_set):
        client_name = path.split(".")[-2][-4:]
        client_id.append(client_name)
        client_n = CedarClient(client_name=client_name)
        model_n = CedarModel(
            module=module,
            model=models.resnet18(pretrained=False, num_classes=10),
            dataset=torch.load(path),
            stimulus_x=torch.load('../../../data/stimuli_example/farm_x_stimulus.pt'),
            layer_fea={'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'},
            process_unit='cuda'
        )
        train_n = CedarTrainer(model=model_n)
        client_n.init_trainer(trainer=train_n)
        train_clients.append(client_n)

    for index, path in enumerate(test_path_set):
        client_name = path.split(".")[-2][-4:]
        client_n = CedarClient(client_name=client_name)
        model_n = CedarModel(
            module=module,
            model=models.resnet18(pretrained=False, num_classes=10),
            dataset=torch.load(path),
            process_unit='cuda'
        )
        model_n.model = models.resnet18(pretrained=False, num_classes=10)
        train_n = CedarTrainer(model=model_n)
        client_n.init_trainer(trainer=train_n)
        test_clients.append(client_n)

    server = StandAloneCedarServer(
        server_name='server1',
        evaluation_client=test_clients,
        client_pool=client_id,
    )
    task = CedarTask(
        task_name='task1',
        maxround=100,
        model=CedarServerModel(
            module=module,
            model=models.resnet18(pretrained=False, num_classes=10),
            process_unit="/cpu:0"
        ),
        evaluation=Accuracy(target=0.9),
        aggregation=Cedar_syn(
            detect=False,
            num_class=10,
            dataset_path='../../../data/non_iid_data/sfddd/test_client/p016.pt',
        ),
        select=AllSelect(
            client_list=train_clients,
            select_num=len(train_clients)
        ),
        module_path='../../module/MNIST/tfmodule.py',
        last_path='/test_sfddd',
        path_now='.',
        dataset='sfddd',
    )
    server.start_task(
        task=task,
        client_objects=train_clients
    )
    result_localize = task.run_localization(20)
    result_localize.to_excel('./result_save_test/localize_result.xlsx', index=False)
