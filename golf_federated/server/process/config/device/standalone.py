# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 11:00
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 11:00

# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 11:01
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 11:01
from copy import deepcopy
from typing import List
from queue import Queue

from golf_federated.client.process.config.device.base import BaseClient
from golf_federated.server.process.config.device.base import BaseServer
from golf_federated.server.process.config.task.base import BaseTask
from golf_federated.utils.log import loggerhear


class StandAloneServer(BaseServer):
    """

    Stand-Alone Server object class, inheriting from Server class.

    """

    def start_task(
        self,
        task: BaseTask,
        client_objects: List[BaseClient]
    ) -> None:
        """

        Start task. In stand-alone cases, the Client objects are called directly.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.
            client_objects (List[golf_federated.client.process.config.device.base.BaseClient]): Client objects.

        """

        # Update the client list of Select object.
        task.select.client_list = client_objects

        # Get selected clients.
        selected_clients = task.select_clients()

        # Start Task.
        task.start(client_list=selected_clients)
        loggerhear.log('Server Info  ',
                       "Task %s on Server %s starts." % (task.task_name, self.server_name))

        # Store Client objects with a queue and update the aggregated fields of the selected Client objects.
        client_queue = Queue()
        for selected_client in selected_clients:
            client_queue.put(selected_client)
            selected_client.field = task.aggregation.get_field()

        # Circular queue to execute task workflow.
        while not task.task_stop:
            # Queue first client object out of the queue.
            client_from_queue = client_queue.get()

            # Local training.
            client_from_queue.train()

            # Upload parameters.
            self.receive_parameter(
                client_name=client_from_queue.client_name,
                client_model=client_from_queue.get_model(),
                client_aggregation_field=client_from_queue.get_field(),
                task=task
            )

            # Client objects out of the queue are re-queued to form a circular queue.
            client_queue.put(client_from_queue)

    def task_update_model(
        self,
        task: BaseTask
    ) -> None:
        """

        Update model.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.

        """

        for task_selected_client in task.client_list:
            task_selected_client.update_model(new_weight=task.model.get_weight())

    def task_stop(
        self,
        task: BaseTask
    ) -> None:
        """

        Stop task.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.

        """

        for task_selected_client in task.client_list:
            task_selected_client.stop()
        task.task_stop = True


class StandAloneCedarServer(BaseServer):
    """

    Stand-Alone Cedar Server object class, inheriting from Server class.

    """

    def __init__(
        self,
        server_name: str,
        evaluation_client: List,
        client_pool: List = []
    ) -> None:
        """

        Initialize the Server object.

        Args:
            server_name (str): Name of the Server object.
            evaluation_client (List): Evaluation clients.
            client_pool (List): Client pool with client object or client name as element. Default as [].

        """

        super().__init__(server_name, client_pool)
        self.evaluation_client = evaluation_client

    def start_task(
        self,
        task: BaseTask,
        client_objects: List[BaseClient]
    ) -> None:
        """

        Start task. In stand-alone cases, the Client objects are called directly.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.
            client_objects (List[golf_federated.client.process.config.device.base.BaseClient]): Client objects.

        """

        # Update the client list of Select object.
        task.select.client_list = client_objects
        task.evaluation_client = self.evaluation_client

        # Get selected clients.
        selected_clients = task.select_clients()

        # Start Task.
        task.start(client_list=selected_clients)
        loggerhear.log('Server Info  ',
                       "Task %s on Server %s starts." % (task.task_name, self.server_name))

        # Store Client objects with a queue and update the aggregated fields of the selected Client objects.
        client_queue = Queue()
        for selected_client in selected_clients:
            client_queue.put(selected_client)
            selected_client.field = task.aggregation.get_field()

        # Circular queue to execute task workflow.
        while not task.task_stop:
            # Queue first client object out of the queue.
            client_from_queue = client_queue.get()

            # Local training.
            client_from_queue.train()
            client_from_queue.trainer.local_stimulate()
            client_from_queue.trainer.global_stimulate()
            client_from_queue.trainer.calculate_RCS()

            # Upload parameters.
            self.receive_parameter(
                client_name=client_from_queue.client_name,
                client_model=client_from_queue.trainer.model.upgrade_weight,
                client_aggregation_field=client_from_queue.get_field(),
                task=task
            )

            # Client objects out of the queue are re-queued to form a circular queue.
            client_queue.put(client_from_queue)

    def task_update_model(
        self,
        task: BaseTask
    ) -> None:
        """

        Update model.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.

        """

        for task_selected_client in task.client_list:
            task_selected_client.trainer.model.model = deepcopy(task.model.model)

    def task_stop(
        self,
        task: BaseTask
    ) -> None:
        """

        Stop task.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.

        """

        for task_selected_client in task.client_list:
            task_selected_client.stop()
        task.task_stop = True
