# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/10 17:56
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/10 17:56

from numpy import ndarray

from golf_federated.client.process.config.trainer.base import BaseTrainer
from golf_federated.client.process.config.model.base import BaseModel


class DirectTrainer(BaseTrainer):
    """

    Direct Trainer object class, inheriting from Trainer class.

    """

    def __init__(
            self,
            model: BaseModel
    ) -> None:
        """

        Args:
            model (golf_federated.client.process.config.model.base.BaseModel): Predefined Model object.

        """

        # Super class init.
        super().__init__('direct')

        # Initialize object properties.
        self.model = model

    def train(self) -> None:
        """

        Training.

        """

        # Trainer counts the training.
        self.trained_num += 1

        # Call model objects to perform training.
        self.model.train()

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

        # Call model objects to perform prediction.
        return self.model.predict(data)

    def update_model(
            self,
            new_weight: list
    ) -> None:
        """

        Model weight update.

        Args:
            new_weight (list): Model weight for update.

        """

        # Call model objects to update model weight.
        self.model.update_weight(new_weight=new_weight)

    def get_model(self) -> list:
        """

        Get model weight.

        Returns:
            List: Model weight.

        """

        # Call model objects to get model weight.
        return self.model.get_weight()

    def get_train_data(self) -> ndarray:
        """

        Get data values.

        Returns:
            Numpy.ndarray: Data values.

        """

        # Call model objects to get data values.
        return self.model.train_data

    def get_train_label(self) -> ndarray:
        """

        Get data labels.

        Returns:
            Numpy.ndarray: Data labels.

        """

        # Call model objects to get data labels.
        return self.model.train_label

    def stop_trainer(self) -> None:
        """

        Stop trainer. Temporarily, the method does not perform the operation.

        """

        pass
