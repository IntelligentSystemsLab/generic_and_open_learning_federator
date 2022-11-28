# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/7 23:21
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/7 23:21

import os
from typing import Any
import numpy as np



def sim_cost(
        data: Any,
        file_path: bool = False,
        communication_num: int = 1
) -> float:
    """

    Estimated communication cost in terms of file size.

    Args:
        data (Any): Data or data files.
        file_path (bool): Whether it is a data file. Default as False.
        communication_num (int): Number of calculations.

    Returns:
        Float: Communication cost.

    """

    # Judge whether it is a data file. Default as False.
    if file_path:
        # Data file.
        path = data

    else:
        # Store in temporary file.
        path = 'temp.npy'
        np.save(path, data)

    # Read size of file.
    sim_cc = os.path.getsize(path) * communication_num

    return sim_cc
