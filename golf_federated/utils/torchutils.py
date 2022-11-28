# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/7 22:28
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/7 22:28

from numpy import ndarray
from torch.utils.data import Dataset


class simple_dataset(Dataset):
    """

    Simple Dataset implementation for PyTorch.

    """

    def __init__(
            self,
            data: ndarray,
            label: ndarray
    ) -> None:
        """

        Initialize the simple_dataset object.

        Args:
            data (numpy.ndarray): Data values.
            label (numpy.ndarray): Data labels.

        """

        # Initialize object properties.
        self.data = data
        self.label = label
        self.length = data.shape[0]

    def __getitem__(self, mask):
        label = self.label[mask]
        data = self.data[mask]
        return label, data

    def __len__(self):
        return self.length
