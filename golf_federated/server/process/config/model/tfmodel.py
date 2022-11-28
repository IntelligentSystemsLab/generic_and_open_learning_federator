# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:28
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:28

from numpy import ndarray
import tensorflow as tf

from golf_federated.server.process.config.model.base import BaseModel


class TensorflowModel(BaseModel):
    """

    Tensorflow Model object class, inheriting from Model class.

    """

    def __init__(
            self,
            module: object,
            test_data: ndarray,
            test_label: ndarray,
            process_unit: str = "/cpu:0"
    ) -> None:
        """

        Initialize the Tensorflow Model object.

        Args:
            module (object): Model module, including predefined model structure, loss function, optimizer, etc.
            test_data (numpy.ndarray): Data values for evaluation.
            test_label (numpy.ndarray): Data labels for evaluation.
            process_unit: Processing unit to perform evaluation. Default as "/cpu:0".

        """

        # Super class init.
        super().__init__(module, test_data, test_label, process_unit)

        # Initialize object properties.
        self.file_ext = '.h5'

    def predict(self) -> ndarray:
        """

        Model prediction.

        Returns:
            Numpy.ndarray: Prediction result.

        """

        with tf.device(self.process_unit):
            result = self.model(self.test_data).numpy()
        return result
