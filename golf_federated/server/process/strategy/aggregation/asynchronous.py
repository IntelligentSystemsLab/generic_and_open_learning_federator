# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:43
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:43

from queue import Queue

from golf_federated.server.process.strategy.aggregation.base import BaseFed
from golf_federated.server.process.strategy.aggregation.function import fedfd, SLMFedasyn, fedasync
from golf_federated.server.process.strategy.selection.function import softmax_prob_from_indicators
from golf_federated.utils.log import loggerhear


class FedFD(BaseFed):
    """

    Asynchronous FL with FedFD, inheriting from BaseFed class.

    """

    def __init__(
            self,
            min_to_start: int = 2
    ) -> None:
        """

        Initialize the FedFD object.

        Args:
            min_to_start (int): Minimum number of received local model parameters for global model aggregation. Default as 2.

        """

        # Super class init.
        super().__init__(
            name='fedfd',
            synchronous=False,
            min_to_start=min_to_start
        )
        loggerhear.log("Server Info  ", "Being Adopting FedFD")

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
        client_id = []
        weight = dict()
        client_round = dict()

        # Get the specified data.
        version_latest = self.aggregation_version
        parameter = datadict['parameter']
        while not parameter.empty():
            temp = parameter.get()
            client_id.append(temp['name'])
            weight[temp['name']] = temp['model']
            client_round[temp['name']] = temp['aggregation_field']['clientRound']

        # Calling aggregation function.
        current_global_w = fedfd(
            client_id=client_id,
            weight=weight,
            client_round=client_round,
            version_latest=version_latest
        )

        # Counter plus one.
        self.aggregation_version += 1

        return current_global_w

    def get_field(self) -> list:
        """

        Get the fields needed for aggregation.

        Returns:
            List: Fields needed for aggregation

        """

        # Rounds of local training.
        return ['clientRound']


class FedAsync(BaseFed):
    """

    Asynchronous FL with FedAsync, inheriting from BaseFed class.
    From: "Asynchronous Federated Optimization"
            (https://opt-ml.org/oldopt/papers/2020/paper_28.pdf)

    """

    def __init__(
            self,
            alpha: float = 0.5,
            beta: float = 0.0,
            staleness: str = 'Polynomial',
            min_to_start: int = 2
    ) -> None:
        """

        Initialize the FedAsync object.

        Args:
            alpha (float): Corresponds to the parameter α defined in FedAsync. Default as 0.5.
            beta (float): Corresponds to the parameter β defined in FedAsync. Default as 0.0.
            staleness (str): Corresponds to the name of the function defined in FedAsync. Default as 'Polynomial'.
            min_to_start (int): Minimum number of received local model parameters for global model aggregation. Default as 2.

        """

        # Super class init.
        super().__init__(
            name='fedasync',
            synchronous=False,
            min_to_start=min_to_start
        )

        # Initialize object properties.
        self.alpha = alpha
        self.beta = beta
        self.staleness = staleness
        loggerhear.log("Server Info  ", "Being Adopting FedAsync")

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
        client_id = []
        weight = dict()

        # Get the specified data.
        current_weight = datadict['current_w']
        current_round = self.aggregation_version
        client_round = dict()
        parameter = datadict['parameter']
        while not parameter.empty():
            temp = parameter.get()
            client_id.append(temp['name'])
            weight[temp['name']] = temp['model']
            client_round[temp['name']] = temp['aggregation_field']['clientRound']

        # Calling aggregation function.
        current_global_w = fedasync(
            client_id=client_id,
            weight=weight,
            staleness=self.staleness,
            current_weight=current_weight,
            current_round=current_round,
            client_round=client_round,
            alpha=self.alpha,
            beta=self.beta
        )

        # Counter plus one.
        self.aggregation_version += 1

        return current_global_w

    def get_field(self) -> list:
        """

        Get the fields needed for aggregation.

        Returns:
            List: Fields needed for aggregation

        """

        # Rounds of local training.
        return ['clientRound']


class SLMFed_asyn(BaseFed):
    """

    Asynchronous FL with SLMFed, inheriting from BaseFed class.

    """

    def __init__(
            self,
            target_acc: float,
            func: str = 'other',
            min_to_start: int = 2,
    ) -> None:
        """
        
        Initialize the SLMFed_asyn object.
                
        Args:
            target_acc(float): Target accuracy of the task.
            func (str):  Function to adjust aggregation weights. Default as 'other'.
            min_to_start (int): Minimum number of received local model parameters for global model aggregation. Default as 2.
            
        """

        # Super class init.
        super().__init__(
            name='SLMFed_asyn',
            synchronous=False,
            min_to_start=min_to_start
        )

        # Initialize object properties.
        self.func = func
        self.target_acc = target_acc
        loggerhear.log("Server Info  ", "Being Adopting SLMFed_asyn")

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
        client_id = []
        weight = dict()
        aggregate_percentage = dict()

        # Get the specified data.
        current_weight = datadict['current_w']
        current_acc = datadict['record'][-1]
        target_acc = self.target_acc
        parameter = datadict['parameter']
        while not parameter.empty():
            temp = parameter.get()
            client_id.append(temp['name'])
            weight[temp['name']] = temp['model']
            client_information_richness[temp['name']] = temp['aggregation_field']['informationRichness']
            client_datasize[temp['name']] = temp['aggregation_field']['dataSize']

        # Get weight of clients for aggregation.
        percentage = softmax_prob_from_indicators([client_information_richness, client_datasize])
        for i in range(len(client_id)):
            aggregate_percentage[client_id[i]] = percentage[i]

        # Calling aggregation function.
        current_global_w = SLMFedasyn(
            client_id=client_id,
            weight=weight,
            aggregate_percentage=aggregate_percentage,
            current_weight=current_weight,
            current_acc=current_acc,
            func=self.func,
            target_acc=target_acc
        )

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
