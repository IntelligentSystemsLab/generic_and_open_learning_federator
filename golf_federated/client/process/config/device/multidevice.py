# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/10 13:30
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/10 13:30

from importlib import import_module
import numpy as np
from numpy import ndarray

from golf_federated.client.process.config.trainer.direct import DirectTrainer
from golf_federated.client.process.config.trainer.docker import DockerTrainer
from golf_federated.client.communication.api.download import download_model, download_info
from golf_federated.client.communication.api.upload import upload_model
from golf_federated.client.communication.api.interact import client_register
from golf_federated.client.communication.sse.monitor import monitor
from golf_federated.client.process.config.device.base import BaseClient
from golf_federated.utils.log import loggerhear


class MultiDeviceClient(BaseClient):
    """

    Multi-Device Client object class, inheriting from Client class.

    """

    def __init__(
            self,
            client_name: str,
            api_host: str,
            api_port: str,
            train_data: ndarray,
            train_label: ndarray,
            process_unit: str = 'cpu',
            docker_port: str = '7789'
    ) -> None:
        """

        Initialize the Multi-Device Client object.

        Args:
            client_name (str): Name of the Client object.
            api_host (str): Host name to connect to the API host.
            api_port (str): Port number to connect to the API host.
            train_data (numpy.ndarray): Data values for training.
            train_label (numpy.ndarray): Data labels for training.
            process_unit (str): Processing unit to perform local training. Default as 'cpu'.
            docker_port (str): Port number for Docker. Default as '7789'.
        """

        # Super class init.
        super().__init__(client_name=client_name)

        # Initialize object properties.
        self.api_host = api_host
        self.api_port = api_port
        self.train_data = train_data
        self.train_label = train_label
        self.process_unit = process_unit
        self.docker_port = docker_port

        # Register this Client with the Server
        client_register(
            host=self.api_host,
            port=self.api_port,
            client_name=self.client_name
        )

        # Create an information channel and listen to Server pushes.
        monitor(client=self, host=api_host, port=api_port, thread_name=self.client_name + '_monitor')

    def update_model(self) -> None:
        """

        Update local model weight.

        """

        # Download the current global model.
        model_file = download_model(
            host=self.api_host,
            port=self.api_port,
            client_name=self.client_name
        )

        # Read model weight.
        new_weight = np.load(
            model_file,
            allow_pickle=True
        )

        # Update local model.
        loggerhear.log("Client Info  ", "Updating model on %s!" % self.client_name)
        self.trainer.update_model(new_weight=new_weight)

    def init_trainer(self) -> None:
        """

        Initialize the Trainer.

        """

        # Download task info.
        isdocker, filename, self.field = download_info(
            host=self.api_host,
            port=self.api_port,
            client_name=self.client_name
        )

        # Judge whether the task requires Docker.
        if isdocker:
            # With Docker.
            # Create Docker trainer.
            self.trainer = DockerTrainer(
                train_data=self.train_data,
                train_label=self.train_label,
                image_path=filename,
                docker_name='docker_' + self.client_name,
                map_port=self.docker_port,
                docker_port=self.docker_port
            )

            # Initialize model weight.
            self.update_model()

        else:
            # Without Docker.
            # Import model module.
            module = import_module(filename)

            # Judge model library and create Model object.
            if module.library == 'tensorflow':
                # Tensorflow.
                from golf_federated.client.process.config.model.tfmodel import TensorflowModel
                model = TensorflowModel(
                    module=module,
                    train_data=self.train_data,
                    train_label=self.train_label,
                    process_unit=self.process_unit
                )

            elif module.library == 'torch':
                # PyTorch.
                from golf_federated.client.process.config.model.torchmodel import TorchModel
                model = TorchModel(
                    module=module,
                    train_data=self.train_data,
                    train_label=self.train_label,
                    process_unit=self.process_unit
                )

            # Create Direct trainer.
            self.trainer = DirectTrainer(model=model)

    def upload_local_weight(self) -> None:
        """

        Upload local model weight.

        """

        upload_model(
            host=self.api_host,
            port=self.api_port,
            client_name=self.client_name,
            model=self.trainer.get_model(),
            aggregation_field=self.get_field()

        )
