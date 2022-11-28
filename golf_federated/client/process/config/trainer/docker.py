# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/10 17:57
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/10 17:57

import os
import numpy as np
from numpy import ndarray

from golf_federated.utils.data import deepcopy_list
from golf_federated.client.process.config.trainer.base import BaseTrainer


class DockerTrainer(BaseTrainer):
    """

    Docker Trainer object class, inheriting from Trainer class.

    """

    def __init__(
            self,
            train_data: ndarray,
            train_label: ndarray,
            image_path: str,
            docker_name: str,
            map_port: str = '7789',
            docker_port: str = '7789'
    ) -> None:
        """

        Args:
            train_data (numpy.ndarray): Data values for training.
            train_label (numpy.ndarray): Data labels for training.
            image_path (str):  File path of the Docker image.
            docker_name (str): Name of Docker containers.
            map_port (str): Port of local device mapped by Docker.
            docker_port (str): Corresponding port inside Docker.

        """

        # Super class init.
        super().__init__('docker')

        # Initialize object properties.
        self.train_data = train_data
        self.train_label = train_label
        self.docker_name = docker_name
        self.weight = []
        image_name = image_path.split('/')[-1].split('.')[0]

        # Load Docker image.
        os.system('docker load -i %s' % image_path)

        # Run Docker container.
        os.system('docker run --network=bridge -p %s:%s  -dit --name=%s %s' % (
            map_port, docker_port, self.docker_name, image_name))

        # Save data.
        np.save('./temp/' + self.docker_name + '_train_data.npy', self.train_data)
        np.save('./temp/' + self.docker_name + '_train_label.npy', self.train_label)

        # Copy data into the container.
        os.system('docker cp ./temp/' + self.docker_name + '_train_data.npy ' + self.docker_name + ':/train_data.npy')
        os.system('docker cp ./temp/' + self.docker_name + '_train_label.npy ' + self.docker_name + ':/train_label.npy')

        # Delete temporary data files.
        os.remove('./temp/' + self.docker_name + '_train_data.npy')
        os.remove('./temp/' + self.docker_name + '_train_label.npy')

    def train(self) -> None:
        """

        Training.

        """

        # Trainer counts the training.
        self.trained_num += 1

        # Save model weight.
        np.save('./temp/%s_global_model.npy' % self.docker_name, self.weight)

        # Transfer model weight file into the container.
        os.system('docker cp ./temp/%s_global_model.npy %s:/' % (self.docker_name, self.docker_name))

        # Perform model training.
        os.system(
            'docker exec %s python train.py --w_download=%s_global_model.npy --train_data=./train_data.npy '
            '--train_label=./train_label.npy --w_upload=./w_upload.npy --is_train=True' % (
                self.docker_name, self.docker_name))

        # Export the trained model weight.
        os.system('docker cp  %s:/w_upload.npy ./temp/%s_global_model.npy' % (
            self.docker_name, self.docker_name))

    def predict(
            self,
            data: ndarray
    ) -> ndarray:
        """

        Prediction.

        Args:
            data (numpy.ndarray): Data values for prediction.

        Returns:
            Numpy.ndarray: Prediction result.

        """

        # Save data.
        np.save('./temp/' + self.docker_name + '_cal_data.npy', data)

        # Save model weight.
        np.save('./temp/%s_global_model.npy' % self.docker_name)

        # Copy data into the container.
        os.system('docker cp ./temp/' + self.docker_name + '_cal_data.npy ' + self.docker_name + ':/cal_data.npy')

        # Copy model weight into the container.
        os.system('docker cp ./temp/%s_global_model.npy %s:/' % (self.docker_name, self.docker_name))

        # Perform model prediction.
        os.system(
            'docker exec %s python train.py --w_download=%s_global_model.npy --cal_data=./cal_data.npy  --is_cal=True '
            '--cal_result=./cal_result.npy' % (self.docker_name, self.docker_name))

        # Export the prediction result.
        os.system('docker cp  %s:/cal_result.npy ./temp/%s_cal_result.npy' % (
            self.docker_name, self.docker_name))

        # Load prediction result.
        return np.load('./temp/%s_cal_result.npy' % self.docker_name)

    def update_model(
            self,
            new_weight: list
    ) -> None:
        """

        Model weight update.

        Args:
            new_weight (list): Model weight for update.

        """

        self.weight = deepcopy_list(new_weight)

    def get_model(self) -> list:
        """

        Get model weight.

        Returns:
            List: Model weight.

        """

        return deepcopy_list(self.weight)

    def get_train_data(self) -> ndarray:
        """

        Get data values.

        Returns:
            Numpy.ndarray: Data values.

        """

        return self.train_data

    def get_train_label(self) -> ndarray:
        """

        Get data labels.

        Returns:
            Numpy.ndarray: Data labels.

        """

        return self.train_label

    def stop_trainer(self) -> None:
        """

        Stop trainer.

        """

        # Remove container.
        os.system('docker rm %s -f' % self.docker_name)
