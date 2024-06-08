# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:43
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:43
from copy import deepcopy
from queue import Queue
from typing import List, Tuple

import pandas as pd

from golf_federated.server.process.strategy.aggregation.function import fedavg, SLMFedsyn, Cedarsyn
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
            'current_w': List,
            'parameter': Queue,
            'record'   : List
        }
    ) -> List:
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

    def get_field(self) -> List:
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
            'current_w': List,
            'parameter': Queue,
            'record'   : List
        }
    ) -> List:
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

    def get_field(self) -> List:
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
            'current_w': List,
            'parameter': Queue,
            'record'   : List
        }
    ) -> List:
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

    def get_field(self) -> List:
        """

        Get the fields needed for aggregation.

        Returns:
            List: Fields needed for aggregation

        """

        # Information richness and data size of client.
        return ['informationRichness', 'dataSize']

        return ['dataSize']


class Cedar_syn(BaseFed):
    """

        Synchronous FL with Cedar, inheriting from BaseFed class.

    """

    def __init__(
        self,
        dataset_path: str,
        min_to_start: int = 2,
        num_class: int = 0,
        detect: bool = False
    ) -> None:
        """

        Initialize the Cedar object.

        Args:
            dataset_path (str): Path to Stimuli.
            min_to_start (int): Minimum number of received local model parameters for global model aggregation. Default as 2.
            num_class (int): Number of data classes. Default as 0.
            detect (bool): Whether to detect malicious updates. Default as False.

        """

        # Super class init.
        super().__init__(
            name='Cedar_syn',
            synchronous=True,
            min_to_start=min_to_start
        )
        self.layer_num_list = []
        self.detect = detect
        self.num_class = num_class
        self.dataset_path = dataset_path
        self.stimulus_x, self.stimulus_y = self.prepare_stimulus_LFA(1)
        loggerhear.log("Server Info  ", "Being Adopting Cedar_syn")

    def aggregate(
        self,
        datadict: {
            'current_w': List,
            'parameter': Queue,
            'record'   : List
        }
    ) -> List:
        """

        Abstract method for aggregation.

        Args:
            datadict (dict): Data that will be input into the aggregation function, including current global model weights, client uploaded parameters and evaluation records.

        Returns:
            List: The model generated after aggregation. And use a list to store the parameters of different layers.

        """

        local_model = []
        upgrade_bool_dataframe = pd.DataFrame()
        upgrade_bool_list = []
        current_model = datadict['current_w']
        parameter = datadict['parameter']
        while not parameter.empty():
            temp = parameter.get()
            local_model.append(temp['model'])
            upgrade_bool_list.append(temp['aggregation_field']['upgrade_bool'])
            upgrade_bool_dataframe = pd.concat(
                [upgrade_bool_dataframe, pd.DataFrame(temp['aggregation_field']['upgrade_bool'])],
                axis=1)
            require_judge_layer = temp['aggregation_field']['REQUIRE_JUDGE_LAYER']
            NUM_LAYER = temp['aggregation_field']['NUM_LAYER']
        require_judge_layer = list(require_judge_layer)
        local_model_object = []
        for l_m in local_model:
            for k, v in l_m.items():
                if l_m[k] is None:
                    l_m[k] = current_model.state_dict()[k]
            temp_model = deepcopy(current_model)
            temp_model.load_state_dict(l_m)
            local_model_object.append(temp_model)
        layer_weight = []
        layer_sum = []
        for i in range(NUM_LAYER):
            if upgrade_bool_dataframe.iloc[i, :].sum() != 0:
                layer_sum.append(upgrade_bool_dataframe.iloc[i, :].sum())
                layer_weight.append(1 / upgrade_bool_dataframe.iloc[i, :].sum())
            else:
                layer_weight.append('inf')
        self.layer_num_list.append(layer_sum)
        current_global_w = Cedarsyn(local_model=local_model_object,
                                    detect=self.detect,
                                    stimulus_x=self.stimulus_x,
                                    current_model=current_model,
                                    num_layer=NUM_LAYER,
                                    layer_weight=layer_weight,
                                    upgrade_bool_list=upgrade_bool_list,
                                    require_judge_layer=require_judge_layer)

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
        return []

    def prepare_stimulus_LFA(
        self,
        each_num: int
    ) -> Tuple:
        """

        Prepare the stimuli

        Args:
            each_num (int): Number of samples per class in the stimuli.

        Returns:
            Tuple[torch.tensor,torch.tensor]: Stimuli samples and labels.

        """

        import torch
        stimulus_data_ori = torch.load(self.dataset_path)
        stimulus_x = []
        stimulus_y = []
        categorize_flag = []
        stimulus_list = [i for i in range(self.num_class)]
        for single_class in stimulus_list:
            categorize_flag.append([0, single_class])

        for i in range(len(stimulus_data_ori)):
            for class_flag in categorize_flag:
                if (stimulus_data_ori[i][1] == class_flag[1]) & (class_flag[0] < each_num):
                    if len(stimulus_x) == 0:
                        stimulus_x = torch.tensor(stimulus_data_ori[i][0]).unsqueeze(0)
                        stimulus_y.append(stimulus_data_ori[i][1])
                    else:
                        temp_b = torch.tensor(stimulus_data_ori[i][0]).unsqueeze(0)
                        stimulus_x = torch.cat([stimulus_x, temp_b], 0)
                        stimulus_y.append(stimulus_data_ori[i][1])

                    class_flag[0] = class_flag[0] + 1

        stimulus_y = torch.tensor(stimulus_y)
        return stimulus_x, stimulus_y
