# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:43
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:43

from abc import abstractmethod
from queue import Queue


class BaseFed(object):
    """

    Aggregation strategy base class.

    """

    def __init__(
            self,
            name: str,
            synchronous: bool,
            min_to_start: int
    ) -> None:
        """

        Initialize the base class object of the aggregation strategy, which is called when subclasses inherit.

        Args:
            name (str): Name of aggregation strategy.
            synchronous (bool): Synchronous FL or not.
            min_to_start (int): Minimum number of received local model parameters for global model aggregation.

        """

        # Initialize object properties.
        self.name = name
        self.synchronous = synchronous
        self.min_to_start = min_to_start
        self.aggregation_version = 0

    @abstractmethod
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

        pass

    @abstractmethod
    def get_field(self) -> list:
        """

        Abstract method for getting the fields needed for aggregation.

        Returns:
            List: Fields needed for aggregation

        """

        pass

    def get_aggregation_time(self) -> int:
        """

        Get the times of the executed aggregation.

        Returns:
            Int: Times of the executed aggregation.

        """

        return self.aggregation_version
