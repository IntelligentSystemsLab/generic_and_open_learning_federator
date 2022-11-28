# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/14 16:00
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/14 16:00

import random
from abc import abstractmethod
from numpy import ndarray

from golf_federated.utils.model import get_model_parameter, set_model_parameter
from golf_federated.utils.data import deepcopy_list


class BaseModel(object):
    """

    Model object class, the class function supports the main operation of model on Client.

    """

    def __init__(
            self,
            module: object,
            train_data: ndarray,
            train_label: ndarray,
            process_unit: str
    ) -> None:
        """

        Initialize the Model object.

        Args:
            module (object): Model module, including predefined model structure, loss function, optimizer, etc.
            train_data (numpy.ndarray): Data values for training.
            train_label (numpy.ndarray): Data labels for training.
            process_unit (str): Processing unit to perform local training.
        """

        # Initialize object properties.
        self.model = getattr(module, module.model)()
        self.library = module.library
        self.optimizer = module.optimizer
        self.loss = module.loss
        self.batch_size = module.batch_size
        self.train_epoch = module.train_epoch
        self.train_data = train_data
        self.train_label = train_label
        self.process_unit = process_unit

    @abstractmethod
    def train(self) -> None:
        """

        Abstract method for model training.

        """

        pass

    @abstractmethod
    def predict(
            self,
            data: ndarray
    ) -> ndarray:
        """

        Abstract method for model prediction.

        Args:
            data (numpy.ndarray): Data values for prediction.

        Returns:
            Numpy.ndarray: Prediction result.

        """

        pass

    def get_weight(self) -> list:
        """

        Get model weight.

        Returns:
            List: Model weight.

        """

        return get_model_parameter(
            model=self.model,
            library=self.library,
        )

    def update_weight(
            self,
            new_weight: list,
    ) -> None:
        """

        Update model weight.

        Args:
            new_weight (list): Model weight for update.

        """

        self.model = set_model_parameter(
            model=self.model,
            w=new_weight,
            library=self.library
        )

    def choose_layer(
            self,
            prob_list: list
    ) -> list:
        """

        Get the model parameter and set some layers to None based on the specified probability, i.e. some layers are not uploaded.

        Args:
            prob_list (list): Probability list, which corresponds to the parameter layers individually.

        Returns:
            List: Model parameters after adjustment.

        """

        # Deep copy to create a temporary variable.
        return_w = deepcopy_list(self.get_weight())

        # Set some layers to None based on established rules
        temp = True
        for i in range(len(return_w)):
            if prob_list[i] == 999:
                if not temp:
                    return_w[i] = None
            else:
                p = random.random()
                if p > prob_list[i]:
                    return_w[i] = None
                    temp = False
                else:
                    temp = True

        return return_w
