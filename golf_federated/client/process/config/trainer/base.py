# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/10 17:55
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/10 17:55

from abc import abstractmethod
from numpy import ndarray


class BaseTrainer(object):
    """

    Trainer object class, the class function supports the main operation of trainer on Client.

    """

    def __init__(
            self,
            mode: str
    ) -> None:
        """

        Initialize the Trainer object.

        Args:
            mode (str): Mode of Trainer. Now support 'direct' and 'docker'.

        """

        # Initialize object properties.
        self.mode = mode
        self.trained_num = 0

    @abstractmethod
    def train(self) -> None:
        """

        Abstract method for training.

        """

        pass

    @abstractmethod
    def predict(
            self,
            data: ndarray
    ) -> ndarray:
        """

        Abstract method for prediction.

        Args:
            data (numpy.ndarray): Data values for prediction.

        Returns:
            Numpy.ndarray: Prediction result.

        """

        pass

    @abstractmethod
    def update_model(
            self,
            new_weight: list
    ):
        """

        Abstract method for model weight update.

        Args:
            new_weight (list): Model weight for update.

        """

        pass

    @abstractmethod
    def get_model(self) -> list:
        """

        Abstract method for model weight getting.

        Returns:
            List: Model weight.

        """

        pass

    @abstractmethod
    def get_train_data(self) -> ndarray:
        """

        Abstract method for data values getting.

        Returns:
            Numpy.ndarray: Data values.

        """

        pass

    @abstractmethod
    def get_train_label(self) -> ndarray:
        """

        Abstract method for data labels getting.

        Returns:
            Numpy.ndarray: Data labels.

        """

        pass

    @abstractmethod
    def stop_trainer(self) -> None:
        """

        Abstract method for trainer stopping.

        """

        pass
