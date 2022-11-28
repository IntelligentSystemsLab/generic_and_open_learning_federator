# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/14 16:00
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/14 16:00

import tensorflow as tf
from numpy import ndarray

from golf_federated.client.process.config.model.base import BaseModel


class TensorflowModel(BaseModel):
    """

    Tensorflow Model object class, inheriting from Model class.

    """

    def __init__(
            self,
            module: object,
            train_data: ndarray,
            train_label: ndarray,
            process_unit: str = "/cpu:0"
    ) -> None:
        """

        Initialize the Tensorflow Model object.

        Args:
            module (object): Model module, including predefined model structure, loss function, optimizer, etc.
            train_data (numpy.ndarray): Data values for training.
            train_label (numpy.ndarray): Data labels for training.
            process_unit (str): Processing unit to perform local training. Default as "/cpu:0".

        """

        # Super class init.
        super().__init__(module, train_data, train_label, process_unit)

        # Initialize object properties.
        self.file_ext = '.h5'
        self.metrics = module.metrics
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def train(self) -> None:
        """

        Model training.

        """

        with tf.device(self.process_unit):
            self.model.fit(
                self.train_data,
                self.train_label,
                batch_size=self.batch_size,
                epochs=self.train_epoch,
                verbose=1
            )

    def predict(
            self,
            data: ndarray
    ) -> ndarray:
        """

        Model prediction.

        Args:
            data (numpy.ndarray): Data values for prediction.

        Returns:
            Numpy.ndarray: Prediction result.

        """

        with tf.device(self.process_unit):
            result = self.model(data).numpy()

        return result
