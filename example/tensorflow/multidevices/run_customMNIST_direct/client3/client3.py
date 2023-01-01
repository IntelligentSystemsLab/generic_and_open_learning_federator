# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/1/1 18:02
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/1/1 18:02

import warnings

warnings.filterwarnings("ignore", category=Warning)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from golf_federated.utils.data import CustomFederatedDataset
from golf_federated.client.process.config.device.multidevice import MultiDeviceClient

if __name__ == '__main__':
    data_dir = './data/'
    x_train = [data_dir + 'x_train_3.npy']
    y_train = [data_dir + 'y_train_3.npy']
    x_test = [data_dir + 'x_test_3.npy']
    y_test = [data_dir + 'y_test_3.npy']
    client_id = ['Client3']

    mnist_fl_data = CustomFederatedDataset(
        train_data=x_train,
        train_label=y_train,
        test_data=x_test,
        test_label=y_test,
        part_num=1,
        part_id=client_id,
        split_data=True,
    )

    data_client_n = mnist_fl_data.get_part_train(client_id[0])
    client_3 = MultiDeviceClient(
        client_name=client_id[0],
        api_host='172.25.151.114',
        api_port='7788',
        train_data=data_client_n[0],
        train_label=data_client_n[1],
    )
