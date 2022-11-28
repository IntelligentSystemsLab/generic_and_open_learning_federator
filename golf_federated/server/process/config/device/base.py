# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 10:41
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 10:41

from abc import abstractmethod
from queue import Queue
from numpy import ndarray

from golf_federated.server.process.config.task.base import BaseTask
from golf_federated.utils.log import loggerhear


class BaseServer(object):
    """

    Server object class, the class function supports the main operation of the Server.

    """

    def __init__(
            self,
            server_name: str,
            client_pool: list = [],
    ) -> None:
        """

        Initialize the Server object.

        Args:
            server_name (str): Name of the Server object.
            client_pool (list): Client pool with client object or client name as element. Default as [].

        """

        # Initialize object properties.
        self.server_name = server_name
        self.client_pool = client_pool
        self.client_selected = []
        self.aggregation_parameter = Queue()
        self.task_list = []

    @abstractmethod
    def start_task(
            self,
            *args
    ) -> None:
        """

        Abstract method for starting Task.

        Args:
            *args: Variable number of parameters, see instantiation methods for details.

        """

        pass

    def receive_parameter(
            self,
            client_name: str,
            client_model: ndarray,
            client_aggregation_field: dict,
            task: BaseTask
    ) -> None:
        """

        Receive parameters uploaded by clients.

        Args:
            client_name (str): Client name of upload parameters.
            client_model (numpy.ndarray): Uploaded model weight.
            client_aggregation_field (dict): Uploaded fields for aggregation and corresponding values.
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.

        """

        # Store the uploaded parameters in the queue.
        loggerhear.log('Server Info  ', "Server %s receives uploaded parameters from Client %s for Task %s" % (
            self.server_name, client_name, task.task_name))
        self.aggregation_parameter.put(
            {
                'name'             : client_name,
                'model'            : client_model,
                'aggregation_field': client_aggregation_field,
            }
        )

        # Call task object to perform model aggregation and evaluation.
        self.task_aggregation_and_evaluation(task=task, aggregation_parameter=self.aggregation_parameter)

    def task_aggregation_and_evaluation(
            self,
            task: BaseTask,
            aggregation_parameter: Queue
    ) -> None:
        """

        Model aggregation and evaluation.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.
            aggregation_parameter (queue.Queue): Queue for storing aggregated parameters.

        """

        # Judge whether the conditions for starting aggregation are met.
        if task.run_aggregation(aggregation_parameter):
            # Start aggregation.
            loggerhear.log('Server Info  ',
                           "Task %s on Server %s finishes aggregation" % (task.task_name, self.server_name))

            # Judge the evaluation situation.
            if task.run_evaluation():
                # Stop task.
                loggerhear.log('Server Info  ',
                               "Task %s on Server %s stops." % (task.task_name, self.server_name))
                self.task_stop(task=task)

            else:
                # Update model.
                loggerhear.log('Server Info  ',
                               "Task %s on Server %s updates global model." % (task.task_name, self.server_name))
                self.task_update_model(task=task)

    @abstractmethod
    def task_update_model(
            self,
            task: BaseTask
    ) -> None:
        """

        Abstract method for model update.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.

        """

        pass

    @abstractmethod
    def task_stop(
            self,
            task: BaseTask
    ) -> None:
        """

        Abstract method for stopping Task.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.

        """

        pass

    def get_task(
            self,
            task_name: str
    ) -> BaseTask:
        """

        Get the task object based on the task name.

        Args:
            task_name (str): Specific Task name.

        Returns:
            golf_federated.server.process.config.task.base.BaseTask: Specific Task object.

        """

        # Retrieve the Specific Task object.
        for i in self.task_list:
            if i.task_name == task_name:
                return i
