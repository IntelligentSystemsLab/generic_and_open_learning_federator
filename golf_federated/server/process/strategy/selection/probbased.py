# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/8 1:54
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/8 1:54

from golf_federated.server.process.strategy.selection.base import BaseSelect
from golf_federated.server.process.strategy.selection.function import random_select_with_percentage, \
    rank_select_with_percentage


class ProbRandomSelect(BaseSelect):
    """

    Random selection with probability, inheriting from BaseSelect class.

    """

    def __init__(
            self,
            client_list: list,
            select_num: int,
            select_probability: list
    ) -> None:
        """

        Initialize the ProbRandomSelect object.

        Args:
            client_list (list): List of total clients.
            select_num (int): Number of clients selected.
            select_probability (list): List of selection probability.

        """

        # Super class init.
        super().__init__(client_list, select_num)

        # Initialize object properties.
        self.select_probability = select_probability

    def select(self) -> list:
        """

        Client selection.

        Returns:
            List: List of clients selected.

        """

        # Calling selection function.
        return random_select_with_percentage(
            client_list=self.client_list,
            client_selected_probability=self.select_probability,
            select_percentage=len(self.client_list) / self.select_num
        )


class ProbRankedSelect(BaseSelect):
    """

    Ranked selection with probability, inheriting from BaseSelect class.

    """

    def __init__(
            self,
            client_list: list,
            select_num: int,
            select_probability: list
    ) -> None:
        """

        Initialize the ProbRankedSelect object.

        Args:
            client_list (list): List of total clients.
            select_num (int): Number of clients selected.
            select_probability (list): List of selection probability.

        """

        # Super class init.
        super().__init__(client_list, select_num)

        # Initialize object properties.
        self.select_probability = select_probability

    def select(self) -> list:
        """

        Client selection.

        Returns:
            List: List of clients selected.

        """

        # Calling selection function.
        return rank_select_with_percentage(
            client_list=self.client_list,
            client_selected_probability=self.select_probability,
            select_percentage=len(self.client_list) / self.select_num
        )
