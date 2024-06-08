# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/10 13:30
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/10 13:30

from abc import abstractmethod
from typing import List

from numpy import ndarray

from golf_federated.utils.log import loggerhear
from golf_federated.utils.data import calculate_IW


class BaseClient(object):
    """

    Client object class, the class function supports the main operation of the client.

    """

    def __init__(
        self,
        client_name: str,
    ) -> None:
        """

        Initialize the Client object.

        Args:
            client_name (str): Name of the Client object.

        """

        # Initialize object properties.
        self.client_name = client_name
        self.trainer = None
        self.field = []

    @abstractmethod
    def init_trainer(
        self,
        *args
    ) -> None:
        """

        Abstract method for initializing the Trainer.

        Args:
            *args: Variable number of parameters, see instantiation methods for details.

        """

        pass

    def train(self) -> None:
        """

        Perform local training.

        """

        # Call the trainer to perform local training.
        loggerhear.log("Client Info  ", "Training Round %d on %s!" % (self.trainer.trained_num + 1, self.client_name))
        self.trainer.train()

    def predict(self) -> ndarray:
        """

        Perform model prediction.

        Returns:
            Numpy.ndarray: Prediction results

        """

        # Call the trainer to perform model prediction.
        loggerhear.log("Client Info  ", "Model prediction on %s!" % self.client_name)
        return self.trainer.predict()

    def get_model(self) -> List:
        """

        Get the current local model weight.

        Returns:
            List: Current local model weight

        """

        # Call the trainer to get the current local model weight.
        return self.trainer.get_model()

    @abstractmethod
    def update_model(
        self,
        *args
    ) -> None:
        """

        Abstract method for updating local model weight.

        Args:
            *args: Variable number of parameters, see instantiation methods for details.

        """

        pass

    def get_field(self) -> dict:
        """

        Get data of specified fields for model aggregation.

        Returns:
            Dict: Data of specified fields for model aggregation.

        """

        # Initialize dictionary.
        field = dict()

        # Fill data to the dictionary.
        for f in self.field:
            # Judge field name.
            if f == 'clientRound':
                # Number of local training rounds.
                field['clientRound'] = self.trainer.trained_num

            elif f == 'informationRichness':
                # Information richness of local data.
                field['informationRichness'] = calculate_IW(self.trainer.get_train_label())

            elif f == 'dataSize':
                # Size of local data.
                field['dataSize'] = self.trainer.get_train_data().shape[0]

        return field

    def stop(self) -> None:
        """

        Stop local training.

        """

        # Call the trainer to stop local training.
        loggerhear.log("Client Info  ", "Stop training on %s!" % self.client_name)
        self.trainer.stop()
