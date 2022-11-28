# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:46
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:46

from numpy import ndarray
import numpy as np

def accuracy(
        target: ndarray,
        prediction: ndarray
) -> float:
    """

    Calculation of Avvuracy.

    Args:
        target (numpy.ndarray): Ground truth.
        prediction (numpy.ndarray): Prediction result.

    Returns:
        Float: Accuracy.

    """

    return sum(np.array(target) == np.array(prediction)) / np.array(target).shape[0]


def mse(
        target: ndarray,
        prediction: ndarray
) -> float:
    """

    Calculation of mse.

    Args:
        target (numpy.ndarray): Ground truth.
        prediction (numpy.ndarray): Prediction result.

    Returns:
        Float: MSE.

    """

    return sum((np.array(target) - np.array(prediction)) ** 2) / np.array(target).shape[0]
