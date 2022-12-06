# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/12/6 12:54
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/12/6 12:54

import warnings

warnings.filterwarnings("ignore", category=Warning)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import sys

sys.path.append('../../../module/MNIST/')

import tfmodule as module

from golf_federated.server.process.strategy.evaluation.classification import Accuracy
from golf_federated.server.process.strategy.aggregation.asynchronous import FedAsync
from golf_federated.server.process.strategy.selection.nonprobbased import AllSelect
from golf_federated.utils.data import CustomFederatedDataset
from golf_federated.server.process.config.device.standalone import StandAloneServer
from golf_federated.server.process.config.task.synchronous import SyncTask
from golf_federated.server.process.config.model.tfmodel import TensorflowModel as TFMserver
from golf_federated.client.process.config.device.standalone import StandAloneClient
from golf_federated.client.process.config.trainer.direct import DirectTrainer
from golf_federated.client.process.config.model.tfmodel import TensorflowModel as TFMclient

if __name__ == '__main__':
    data_dir = '../../../data/non_iid_data_mnist_range5_label_client3/'
    x_train = [data_dir + 'x_train_%s.npy' % (str(i + 1)) for i in range(3)]
    y_train = [data_dir + 'y_train_%s.npy' % (str(i + 1)) for i in range(3)]
    x_test = [data_dir + 'x_test_%s.npy' % (str(i + 1)) for i in range(3)]
    y_test = [data_dir + 'y_test_%s.npy' % (str(i + 1)) for i in range(3)]
    client_id = ['Client%s' % (str(i + 1)) for i in range(3)]
    mnist_fl_data = CustomFederatedDataset(
        train_data=x_train,
        train_label=y_train,
        test_data=x_test,
        test_label=y_test,
        part_num=3,
        part_id=client_id,
        split_data=True,
    )
    clients = []
    for c_id in client_id:
        data_client_n = mnist_fl_data.get_part_train(c_id)
        client_n = StandAloneClient(client_name=c_id)
        model_n = TFMclient(
            module=module,
            train_data=data_client_n[0],
            train_label=data_client_n[1],
            process_unit="/cpu:0"
        )
        train_n = DirectTrainer(model=model_n)
        client_n.init_trainer(trainer=train_n)
        clients.append(client_n)
    server = StandAloneServer(
        server_name='server1',
        client_pool=client_id,
    )
    task = SyncTask(
        task_name='task1',
        maxround=5,
        aggregation=FedAsync(),
        evaluation=Accuracy(target=0.9),
        model=TFMserver(
            module=module,
            test_data=mnist_fl_data.test_data,
            test_label=mnist_fl_data.test_label,
            process_unit="/cpu:0"
        ),
        select=AllSelect(
            client_list=clients,
            select_num=len(clients)
        ),
        module_path='../../../module/MNIST/tfmodule.py'
    )
    server.start_task(
        task=task,
        client_objects=clients
    )
