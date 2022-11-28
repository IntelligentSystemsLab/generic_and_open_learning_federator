# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 13:18
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 13:18

from queue import Queue

from golf_federated.server.process.strategy.selection.base import BaseSelect
from golf_federated.server.process.config.model.base import BaseModel
from golf_federated.server.process.config.task.base import BaseTask
from golf_federated.server.process.strategy.aggregation.base import BaseFed
from golf_federated.server.process.strategy.evaluation.base import BaseEval


class SyncTask(BaseTask):
    """

    Synchronous Task object class, inheriting from Task class.

    """

    def __init__(
            self,
            task_name: str,
            maxround: int,
            aggregation: BaseFed,
            evaluation: BaseEval,
            model: BaseModel,
            select: BaseSelect,
            module_path: str = '',
            isdocker: bool = False,
            image_name: str = ''
    ) -> None:
        """

        Initialize the Synchronous Task object.

        Args:
            task_name (str): Name of the task.
            maxround (int): Maximum number of aggregation rounds.
            aggregation (golf_federated.server.process.strategy.aggregation.base.BaseFed): Aggregation strategy object.
            evaluation (golf_federated.server.process.strategy.evaluation.base.BaseEval): Evaluation strategy object.
            model (golf_federated.server.process.config.model.base.BaseModel): Model object.
            select (golf_federated.server.process.strategy.selection.base.BaseSelect): Select strategy object.
            module_path (str): File path to model module. Default as ''.
            isdocker (bool): Whether the task requires Docker. Default as False.
            image_name (str): Name of Docker image. Default as ''.

        """

        # Super class init.
        super().__init__(task_name, maxround, True, aggregation, evaluation, model, select, module_path,isdocker,image_name)

    def start_aggregation(
            self,
            aggregation_parameter: Queue
    )-> bool:
        """

        Judge whether the conditions for starting aggregation have been met.

        Args:
            aggregation_parameter (queue.Queue): Queue for storing aggregated parameters.

        Returns:
            Bool: Whether to start aggregation.

        """

        # The number of uploaded clients meets the requirements and all clients of the task are uploaded.
        if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= len(
                self.client_list):
            return True

        else:
            return False
