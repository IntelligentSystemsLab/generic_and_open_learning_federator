# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/1/1 19:11
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/1/1 19:11

import warnings

warnings.filterwarnings("ignore", category=Warning)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from golf_federated.utils.config import load_yaml_config

if __name__ == '__main__':

    load_yaml_config('./server.yaml')
