# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:28
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:28

import torch
from numpy import ndarray

from golf_federated.server.process.config.model.base import BaseModel


class TorchModel(BaseModel):
    """

    PyTorch Model object class, inheriting from Model class.

    """

    def __init__(
            self,
            module: object,
            test_data: ndarray,
            test_label: ndarray,
            process_unit: str = "cpu"
    ) -> None:
        """

        Initialize the PyTorch Model object.

        Args:
            module (object): Model module, including predefined model structure, loss function, optimizer, etc.
            test_data (numpy.ndarray): Data values for evaluation.
            test_label (numpy.ndarray): Data labels for evaluation.
            process_unit: Processing unit to perform evaluation. Default as "cpu".

        """

        # Super class init.
        super().__init__(module, test_data, test_label, process_unit)

        # Initialize object properties.
        if test_data.shape[-1] == 1:
            self.test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
        self.file_ext = '.pt'

    def predict(self) -> ndarray:
        """

        Model prediction.

        Returns:
            Numpy.ndarray: Prediction result.

        """

        with torch.no_grad():
            imput = self.test_data.to(self.process_unit)
            result = self.model(imput).cpu().numpy()
        return result
