# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/6/4 23:28
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/10/25 15:26

import warnings

warnings.filterwarnings("ignore", category=Warning)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yaml
import importlib
import sys

from golf_federated.utils.log import loggerhear
from golf_federated.utils.data import CustomFederatedDataset
import golf_federated


def load_yaml_config(file: str) -> None:
    """

    Get the configuration and start by reading the yaml file.
    Leave empty value for unused fields when modifying yaml file.

    Args:
        file (str): Path where the yaml file is stored

    """

    loggerhear.log("Common Info  ", 'Loading Config from %s' % file)

    # Open yaml file and extract the corresponding data values of some fields.
    file = open(file, 'r', encoding="utf-8")
    content = yaml.load(file)
    data_content = content['data']
    model_content = content['model']
    device_content = content['device']
    task_content = content['task']

    # Create a federated dataset object.
    fl_data = CustomFederatedDataset(
        train_data=data_content['x_train'],
        train_label=data_content['y_train'],
        test_data=data_content['x_test'],
        test_label=data_content['y_test'],
        part_num=data_content['part_num'],
        part_id=data_content['client_id'],
        split_data=data_content['split_data'],
    )

    # Judge whether a server or clients are there.
    have_server = False
    have_client = False
    for key, value in device_content.items():
        # Judge device type, i.e., server or client.
        if value['type'] == 'server':
            # Judge whether the server object has been created, currently only supports a unique server object.
            if have_server:
                # Create more than one will report an error and terminate the program.
                loggerhear.log("Error Message", 'Server already defined')
                exit(1)

            else:
                have_server = True
                server_device = key

        elif value['type'] == 'client':
            have_client = True

    # Define model module.
    sys.path.append(model_content['filepath'])
    module = importlib.import_module(model_content['module'])

    # Judge whether to create Client objects.
    client_list = []
    if have_client:
        # Judge the library of Model.
        if model_content['type'] == 'tensorflow':
            # Tensorflow.
            model_client = golf_federated.client.process.config.model.tfmodel.TensorflowModel

        elif model_content['type'] == 'torch':
            # PyTorch.
            model_client = golf_federated.client.process.config.model.torchmodel.TorchModel

        # Create Client objects
        for key, value in device_content.items():
            if value['type'] == 'client':
                # Extract field values.
                client_name = value['client_name']
                process_unit = value['process_unit']
                execution = value['execution']
                trainer = value['trainer']
                data_client_n = fl_data.get_part_train(client_name)

                # Judge the type of device execution.
                if execution == 'StandAlone':
                    # StandAlone.
                    # Client object.
                    client_n = golf_federated.client.process.config.device.standalone.StandAloneClient(
                        client_name=client_name,
                    )

                    # Model object.
                    model_n = model_client(
                        module=module,
                        train_data=data_client_n[0],
                        train_label=data_client_n[1],
                        process_unit=process_unit
                    )

                    # Judge the type of Trainer.
                    if trainer == 'Direct':
                        # Direct.
                        train_n = golf_federated.client.process.config.trainer.direct.DirectTrainer(model=model_n)

                    elif trainer == 'Docker':
                        # Docker.
                        train_n = golf_federated.client.process.config.trainer.docker.DockerTrainer(
                            train_data=data_client_n[0],
                            train_label=data_client_n[1],
                            image_path=value['image_path'],
                            docker_name=value['docker_name'],
                            map_port=value['map_port'],
                            docker_port=value['docker_port']
                        )

                    # Initialize Trainer.
                    client_n.init_trainer(trainer=train_n)

                elif execution == 'MultiDevice':
                    # MultiDevice.
                    # Client object.
                    client_n = golf_federated.client.process.config.device.multidevice.MultiDeviceClient(
                        client_name=client_name,
                        api_host=value['api_host'],
                        api_port=value['api_port'],
                        train_data=data_client_n[0],
                        train_label=data_client_n[1],
                    )

                    # Judge the type of Trainer.
                    if trainer == 'Docker':
                        client_n.docker_port = value['docker_port']

                # Add Client object.
                client_list.append(client_n)

    # Judge whether to create a Server object.
    if have_server:
        # Judge the library of Model.
        if model_content['type'] == 'tensorflow':
            # Tensorflow.
            model_server = golf_federated.server.process.config.model.tfmodel.TensorflowModel

        elif model_content['type'] == 'torch':
            # PyTorch.
            model_server = golf_federated.server.process.config.model.torchmodel.TorchModel

        # Extract field values.
        task_name = task_content['task_name']
        task_type = task_content['task_type']
        maxround = task_content['maxround']
        aggregation = task_content['aggregation']
        evaluation = task_content['evaluation']
        target = task_content['target']
        select = task_content['select']
        process_unit = task_content['process_unit']

        # Judge the type of Task.
        if task_type == 'synchronous':
            # Synchronous.
            # Task object.
            task = golf_federated.server.process.config.task.synchronous.SyncTask(
                task_name=task_name,
                maxround=maxround,
                aggregation=getattr(getattr(golf_federated.server.process.strategy.aggregation, aggregation['type']),
                                    aggregation['name'])(),
                evaluation=getattr(getattr(golf_federated.server.process.strategy.evaluation, evaluation['type']),
                                   evaluation['name'])(target=target),
                model=model_server(
                    module=module,
                    test_data=fl_data.test_data,
                    test_label=fl_data.test_label,
                    process_unit=process_unit
                ),
                select=getattr(getattr(golf_federated.server.process.strategy.selection, select['type']),
                               select['name'])(
                    client_list=client_list,
                    select_num=len(client_list)
                ),
                module_path=model_content['filepath'] + model_content['module'] + '.py',
                isdocker=task_content['isdocker'],
                image_name=task_content['image_name'],
            )

        elif task_type == 'timing_asynchronous':
            # Timing asynchronous.
            # Task object.
            task = golf_federated.server.process.config.task.asynchronous.TimingAsyncTask(
                task_name=task_name,
                maxround=maxround,
                aggregation=getattr(getattr(golf_federated.server.process.strategy.aggregation, aggregation['type']),
                                    aggregation['name'])(),
                evaluation=getattr(getattr(golf_federated.server.process.strategy.evaluation, evaluation['type']),
                                   evaluation['name'])(target=target),
                model=model_server(
                    module=module,
                    test_data=fl_data.test_data,
                    test_label=fl_data.test_label,
                    process_unit=process_unit
                ),
                select=getattr(getattr(golf_federated.server.process.strategy.selection, select['type']),
                               select['name'])(
                    client_list=client_list,
                    select_num=len(client_list)
                ),
                timing=task_content['timing'],
                module_path=model_content['filepath'] + model_content['module'] + '.py',
                isdocker=task_content['isdocker'],
                image_name=task_content['image_name'],
            )

        elif task_type == 'ration_asynchronous':
            # Rational asynchronous.
            # Task object.
            task = golf_federated.server.process.config.task.asynchronous.RationAsyncTask(
                task_name=task_name,
                maxround=maxround,
                aggregation=getattr(getattr(golf_federated.server.process.strategy.aggregation, aggregation['type']),
                                    aggregation['name'])(),
                evaluation=getattr(getattr(golf_federated.server.process.strategy.evaluation, evaluation['type']),
                                   evaluation['name'])(target=target),
                model=model_server(
                    module=module,
                    test_data=fl_data.test_data,
                    test_label=fl_data.test_label,
                    process_unit=process_unit
                ),
                select=getattr(getattr(golf_federated.server.process.strategy.selection, select['type']),
                               select['name'])(
                    client_list=client_list,
                    select_num=len(client_list)
                ),
                ration=task_content['ration'],
                module_path=model_content['filepath'] + model_content['module'] + '.py',
                isdocker=task_content['isdocker'],
                image_name=task_content['image_name'],
            )

        # Extract field values.
        server_content = device_content[server_device]
        server_name = server_content['server_name']
        execution = server_content['execution']

        # Judge the type of device execution.
        if execution == 'StandAlone':
            # StandAlone.
            # Server object.
            server = golf_federated.server.process.config.device.standalone.StandAloneServer(
                server_name=server_name,
                client_pool=data_content['client_id'],
            )

            # Start Task.
            server.start_task(
                task=task,
                client_objects=client_list
            )

        elif execution == 'MultiDevice':
            # MultiDevice.
            # Method of task start.
            def start_task():
                while True:
                    if len(server.client_pool) >= 2:
                        server.start_task(
                            task=task,
                        )
                        break

            # Server object.
            server = golf_federated.server.process.config.device.multidevice.MultiDeviceServer(
                server_name=server_name,
                api_host=server_content['api_host'],
                api_port=server_content['api_port'],
                sse_host=server_content['sse_host'],
                sse_port=server_content['sse_port'],
                sse_db=server_content['sse_db'],
            )

            # Open an extra thread.
            import threading
            start_thread = threading.Thread(target=start_task)
            start_thread.start()

            # Start Task.
            server.start_server()
