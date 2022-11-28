# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:43
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:43

from queue import Queue

from golf_federated.server.process.strategy.aggregation.function import fedavg, SLMFedsyn
from golf_federated.server.process.strategy.aggregation.base import BaseFed
from golf_federated.server.process.strategy.selection.function import softmax_prob_from_indicators
from golf_federated.utils.log import loggerhear


class FedAVG(BaseFed):
    """

    Synchronous FL with FedAVG, inheriting from BaseFed class.
    From: "Communication-Efficient Learning of Deep Networks from Decentralized Data"
            (http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf)

    """

    def __init__(
            self,
            min_to_start: int = 2
    ) -> None:
        """

        Initialize the FedAVG object.

        Args:
            min_to_start (int): Minimum number of received local model parameters for global model aggregation. Default as 2.

        """

        # Super class init.
        super().__init__(
            name='fedavg',
            synchronous=True,
            min_to_start=min_to_start
        )
        loggerhear.log("Server Info  ", "Being Adopting FedAVG")

    def aggregate(
            self,
            datadict: {
                'current_w': list,
                'parameter': Queue,
                'record'   : list
            }
    ) -> list:
        """

        Abstract method for aggregation.

        Args:
            datadict (dict): Data that will be input into the aggregation function, including current global model weights, client uploaded parameters and evaluation records.

        Returns:
            List: The model generated after aggregation. And use a list to store the parameters of different layers.

        """

        # Create temporary variables
        weight = []
        data_size = []

        # Get the specified data.
        parameter = datadict['parameter']
        while not parameter.empty():
            temp = parameter.get()
            weight.append(temp['model'])
            data_size.append(temp['aggregation_field']['dataSize'])

        # Calling aggregation function.
        current_global_w = fedavg(weight, data_size)

        # Counter plus one.
        self.aggregation_version += 1

        return current_global_w

    def get_field(self) -> list:
        """

        Get the fields needed for aggregation.

        Returns:
            List: Fields needed for aggregation

        """

        # Data size of client
        return ['dataSize']


class FedProx(BaseFed):
    """

    Synchronous FL with FedProx, inheriting from BaseFed class.

    """

    def __init__(
            self,
            miu: float = 1,
            min_to_start: int = 2
    ) -> None:
        """

        Initialize the FedProx object.

        Args:
            miu (float): Corresponds to the parameter μ defined in FedProx. Default as 1.
            min_to_start (int): Minimum number of received local model parameters for global model aggregation. Default as 2.

        """

        # Super class init.
        super().__init__(
            name='fedprox',
            synchronous=True,
            min_to_start=min_to_start
        )

        # Initialize object properties.
        self.miu = miu
        loggerhear.log("Server Info  ", "Being Adopting FedProx")

    def aggregate(
            self,
            datadict: {
                'current_w': list,
                'parameter': Queue,
                'record'   : list
            }
    ) -> list:
        """

        Abstract method for aggregation.

        Args:
            datadict (dict): Data that will be input into the aggregation function, including current global model weights, client uploaded parameters and evaluation records.

        Returns:
            List: The model generated after aggregation. And use a list to store the parameters of different layers.

        """

        # Create temporary variables
        weight = []
        data_size = []

        # Get the specified data.
        parameter = datadict['parameter']
        while not parameter.empty():
            temp = parameter.get()
            weight.append(temp['model'])
            data_size.append(temp['aggregation_field']['dataSize'])

        # Calling aggregation function.
        current_global_w = fedavg(weight, data_size)

        # Counter plus one.
        self.aggregation_version += 1

        return current_global_w

    def get_field(self) -> list:
        """

        Get the fields needed for aggregation.

        Returns:
            List: Fields needed for aggregation

        """

        # Data size of client
        return ['dataSize']


class SLMFed_syn(BaseFed):
    """

        Synchronous FL with SLMFed, inheriting from BaseFed class.

    """

    def __init__(
            self,
            min_to_start: int = 2
    ) -> None:
        """

        Initialize the SLMFed_syn object.

        Args:
            min_to_start (int): Minimum number of received local model parameters for global model aggregation. Default as 2.

        """

        # Super class init.
        super().__init__(
            name='SLMFed_syn',
            synchronous=True,
            min_to_start=min_to_start
        )
        loggerhear.log("Server Info  ", "Being Adopting SLMFed_syn")

    def aggregate(
            self,
            datadict: {
                'current_w': list,
                'parameter': Queue,
                'record'   : list
            }
    ) -> list:
        """

        Abstract method for aggregation.

        Args:
            datadict (dict): Data that will be input into the aggregation function, including current global model weights, client uploaded parameters and evaluation records.

        Returns:
            List: The model generated after aggregation. And use a list to store the parameters of different layers.

        """

        # Create temporary variables
        client_information_richness = []
        client_datasize = []
        weight = []

        # Get the specified data.
        current_weight = datadict['current_w']
        parameter = datadict['parameter']
        while not parameter.empty():
            temp = parameter.get()
            weight.append(temp['model'])
            weight[temp['name']] = temp['model']
            client_information_richness[temp['name']] = temp['aggregation_field']['informationRichness']
            client_datasize[temp['name']] = temp['aggregation_field']['dataSize']

        # Get weight of clients for aggregation.
        aggregate_percentage = softmax_prob_from_indicators([client_information_richness, client_datasize])

        # Calling aggregation function.
        current_global_w = SLMFedsyn(weight=weight, aggregate_percentage=aggregate_percentage,
                                     current_weight=current_weight)

        # Counter plus one.
        self.aggregation_version += 1

        return current_global_w

    def get_field(self) -> list:
        """

        Get the fields needed for aggregation.

        Returns:
            List: Fields needed for aggregation

        """

        # Information richness and data size of client.
        return ['informationRichness', 'dataSize']
