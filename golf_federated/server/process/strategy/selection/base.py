# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/8 1:50
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/8 1:50

from abc import abstractmethod
from typing import List


class BaseSelect(object):
    """

    Selection strategy base class.

    """

    def __init__(
        self,
        client_list: List,
        select_num: int
    ) -> None:
        """

        Initialize the base class object of the selection strategy, which is called when subclasses inherit.

        Args:
            client_list (List): List of total clients.
            select_num (int): Number of clients selected.

        """

        # Initialize object properties.
        self.client_list = client_list
        self.select_num = select_num

    @abstractmethod
    def select(self) -> List:
        """

        Abstract method for client selection.

        Returns:
            List: List of clients selected.

        """

        pass
