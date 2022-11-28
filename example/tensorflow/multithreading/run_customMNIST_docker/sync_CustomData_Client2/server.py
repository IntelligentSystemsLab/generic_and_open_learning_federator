# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/14 12:33
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/14 12:33
import warnings

warnings.filterwarnings("ignore", category=Warning)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from golf_federated.server.process.strategy.evaluation.classification import Accuracy

from golf_federated.server.process.strategy.aggregation.synchronous import FedAVG
from golf_federated.server.process.strategy.selection.nonprobbased import AllSelect

import sys

sys.path.append('../../../../module/MNIST/')

import tfmodule as module
from golf_federated.utils.data import CustomFederatedDataset
from golf_federated.server.process.config.task.synchronous import SyncTask
from golf_federated.server.process.config.model.tfmodel import TensorflowModel as TFMserver
from golf_federated.server.process.config.device.multidevice import MultiDeviceServer
import threading


def start_task():
    while True:
        if len(server.client_pool) >= 2:
            server.start_task(
                task=task,
            )
            break


if __name__ == '__main__':
    data_dir = '../../../../data/non_iid_data_mnist_range5_label_client3/'
    x_test = [data_dir + 'x_test_%s.npy' % (str(i + 1)) for i in range(3)]
    y_test = [data_dir + 'y_test_%s.npy' % (str(i + 1)) for i in range(3)]
    mnist_fl_data = CustomFederatedDataset(
        test_data=x_test,
        test_label=y_test,
    )
    server = MultiDeviceServer(
        server_name='server1',
        api_host='127.0.0.1',
        api_port='7788',
        sse_host='127.0.0.1',
        sse_port='6379',
        sse_db=6,
    )
    task = SyncTask(
        task_name='task1',
        maxround=5,
        aggregation=FedAVG(min_to_start=1),
        evaluation=Accuracy(target=0.9),
        model=TFMserver(
            module=module,
            test_data=mnist_fl_data.test_data,
            test_label=mnist_fl_data.test_label,
            process_unit="/cpu:0"
        ),
        select=AllSelect(
            client_list=[],
            select_num=2
        ),
        module_path='../../../../docker/MNIST/tensorflow/image.tar',
        isdocker=True,
        image_name='tfmnist'
    )

    start_thread = threading.Thread(target=start_task)
    start_thread.start()

    server.start_server()
