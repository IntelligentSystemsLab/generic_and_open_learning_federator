# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 14:45
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 14:45

import json
import os
import time
import zipfile
from abc import abstractmethod
from queue import Queue
from typing import List

import numpy as np

from golf_federated.server.process.config.model.base import BaseModel
from golf_federated.server.process.strategy.aggregation.base import BaseFed
from golf_federated.server.process.strategy.evaluation.base import BaseEval
from golf_federated.server.process.strategy.selection.base import BaseSelect


class BaseTask(object):
    """

    Task object class, the class function supports the main operation of task on Server.

    """

    def __init__(
        self,
        task_name: str,
        maxround: int,
        synchronous: bool,
        aggregation: BaseFed,
        evaluation: BaseEval,
        model: BaseModel,
        select: BaseSelect,
        module_path: str,
        isdocker: bool = False,
        image_name: str = ''
    ) -> None:
        """

        Initialize the Task object.

        Args:
            task_name (str): Name of the task.
            maxround (int): Maximum number of aggregation rounds.
            synchronous (bool): Whether the task is synchronous.
            aggregation (golf_federated.server.process.strategy.aggregation.base.BaseFed): Aggregation strategy object.
            evaluation (golf_federated.server.process.strategy.evaluation.base.BaseEval): Evaluation strategy object.
            model (golf_federated.server.process.config.model.base.BaseModel): Model object.
            select (golf_federated.server.process.strategy.selection.base.BaseSelect): Select strategy object.
            module_path (str): File path to model module.
            isdocker (bool): Whether the task requires Docker. Default as False.
            image_name (str): Name of Docker image. Default as ''.

        """

        # Initialize object properties.
        self.task_stop = False
        self.task_name = task_name
        self.maxround = maxround
        self.synchronous = synchronous
        self.aggregation = aggregation
        self.model = model
        self.evaluation = evaluation
        self.select = select
        self.module_path = module_path
        self.isdocker = isdocker
        self.image_name = image_name
        self.client_list = []
        self.round_time = time.time()
        self.time_record = []
        self.round_cost = 0
        self.cost_record = []
        self.info_path = ''
        self.weight_path = ''

    def start(
        self,
        client_list: List,
    ) -> None:
        """

        Start Task.

        Args:
            client_list (list): List of clients for this task.

        """

        # Update client list.
        self.client_list = client_list

        # Initialize object properties.
        self.round_time = time.time()
        self.round_cost = 0
        self.task_stop = False

    @abstractmethod
    def start_aggregation(
        self,
        aggregation_parameter: Queue
    ) -> bool:
        """

        Judge whether the conditions for starting aggregation have been met.

        Args:
            aggregation_parameter (queue.Queue): Queue for storing aggregated parameters.

        Returns:
            Bool: Whether to start aggregation.

        """

        pass

    def run_aggregation(
        self,
        aggregation_parameter: Queue
    ) -> bool:
        """

        Run global model aggregation.

        Args:
            aggregation_parameter (queue.Queue): Queue for storing aggregated parameters.

        Returns:
            Bool: Whether aggregation is executed.

        """

        # Judge whether to start aggregation.
        if self.start_aggregation(aggregation_parameter=aggregation_parameter):
            # Run global model aggregation.
            self.model.model_aggre(aggregation=self.aggregation, parameter=aggregation_parameter,
                                   record=self.evaluation.get_record())
            return True

        else:
            # Conditions for starting aggregation have not been met.
            return False

    def run_evaluation(self) -> bool:
        """

        Run global model evaluation.

        Returns:
            Bool: Evaluation result, indicating the continuation or completion of the task.

        """

        # Call model object to perform evaluation.
        self.model.model_eval(evaluation=self.evaluation)

        # Record time and initialize object property.
        self.time_record.append(time.time() - self.round_time)
        self.round_time = time.time()

        # Record communication cost and initialize object property.
        self.cost_record.append(self.round_cost)
        self.round_cost = 0

        # Multiple evaluation conditions.
        return self.evaluation.reach_target() or self.evaluation.reach_convergence() or self.aggregation.aggregation_version >= self.maxround

    def select_clients(self) -> List:
        """

        Select clients.

        Returns:
            List: Selected clients.

        """

        return self.select.select()

    def info_tozip(self) -> None:
        """

        Save task info to zip.

        """

        # Temporary folder.
        if not os.path.isdir('temp'):
            os.mkdir('temp')

        # Get aggregation field.
        agg_dict = {
            "aggregationField": self.aggregation.get_field()
        }

        # Save aggregation field to Json file.
        with open("temp/task_info.json", "w") as f:
            json.dump(agg_dict, f)

        # Task info zip.
        file_path = 'temp/' + self.task_name + '_info.zip'

        # Create the zip.
        info_zipper = zipfile.ZipFile(
            file_path,
            'w',
            compression=zipfile.ZIP_DEFLATED
        )

        # Write model module or Docker image into the zip.
        info_zipper.write(
            filename=self.module_path,
            arcname=self.image_name + '.tar' if self.isdocker else 'module.py'
        )

        # Write Json file into the zip.
        info_zipper.write(
            filename='temp/task_info.json',
            arcname='task_info.json'
        )

        # Close the zip.
        info_zipper.close()

        # Update corresponding file path.
        self.info_path = file_path

    def weight_tofile(self) -> None:
        """

        Save model weight to zip.

        """

        # Save model weight.
        np.save('weight.npy', self.model.get_weight())

        # Model weight zip.
        file_path = 'temp/' + self.task_name + '_weight.zip'

        # Create the zip.
        weight_zipper = zipfile.ZipFile(
            file_path,
            'w',
            compression=zipfile.ZIP_DEFLATED
        )

        # Write model weight into the zip.
        weight_zipper.write(
            filename='weight.npy',
            arcname='weight.npy'
        )

        # Close the zip.
        weight_zipper.close()

        # Update corresponding file path.
        self.weight_path = file_path
