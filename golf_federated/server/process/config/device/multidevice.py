# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 11:01
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 11:01
from typing import List

from golf_federated.server.communication.sse.schedule import publish_task_init, publish_update_model, publish_stop_train
from golf_federated.server.process.config.device.base import BaseServer
from golf_federated.server.process.config.task.base import BaseTask
from golf_federated.server.process.config.port.api import run_restful
from golf_federated.server.process.config.port.sse import init_sse
from golf_federated.utils.log import loggerhear


class MultiDeviceServer(BaseServer):
    """

    Multi-Device Server object class, inheriting from Server class.

    """

    def __init__(
            self,
            server_name: str,
            client_pool: List = [],
            api_host: str = '127.0.0.1',
            api_port: str = '7788',
            sse_host: str = '127.0.0.1',
            sse_port: str = '6379',
            sse_db: int = 6,
    ) -> None:
        """

        Initialize the Multi-Device Server object.

        Args:
            server_name (str): Name of the Server object.
            client_pool (list): Client pool with client object or client name as element. Default as [].
            api_host (str): Host name to connect to the API host. Default as '127.0.0.1'.
            api_port (str): Port number to connect to the API host. Default as '7788'.
            sse_host (str): Host name to connect to the SSE host. Default as '127.0.0.1'.
            sse_port (str): Port number to connect to the SSE host. Default as '6379'.
            sse_db (int): Adopted Database. Default as 6.

        """

        # Super class init.
        super().__init__(server_name=server_name, client_pool=client_pool)

        # Initialize object properties.
        self.api_host = api_host
        self.api_port = api_port
        self.sse_host = sse_host
        self.sse_port = sse_port
        self.sse_db = sse_db

    def start_server(self) -> None:
        """

        Start server.

        """

        # Initialize SSE.
        config = init_sse(
            server=self,
            host=self.sse_host,
            port=self.sse_port,
            db=self.sse_db
        )

        # Start restful API.
        run_restful(
            config=config,
            host=self.api_host,
            port=self.api_port,
        )

    def start_task(
            self,
            task: BaseTask
    ) -> None:
        """

        Start Task.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.

        """

        # Update the client list of Select object.
        task.select.client_list = self.client_pool

        # Get selected clients.
        client_selected = task.select_clients()

        # Start Task.
        task.start(client_selected)
        loggerhear.log('Server Info  ',
                       "Task %s on Server %s starts." % (task.task_name, self.server_name))

        # Record the started Task object.
        self.task_list.append(task)

        # Save task info.
        task.info_tozip()

        # Save initialized global model weight.
        task.weight_tofile()

        # Publish 'TaskInit' info.
        publish_task_init(
            host=self.sse_host,
            port=self.sse_port,
            db=self.sse_db
        )

    def client_register(
            self,
            client_name: str
    ) -> None:
        """

        Client register.

        Args:
            client_name (str): Name of the client to register.

        """

        self.client_pool.append(client_name)

    def task_update_model(
            self,
            task: BaseTask
    ) -> None:
        """

        Update model.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.

        """

        # Update the saved global model weight file.
        task.weight_tofile()

        # Publish 'UpdateModel' info.
        publish_update_model(
            host=self.sse_host,
            port=self.sse_port,
            db=self.sse_db
        )

    def task_stop(
            self,
            task: BaseTask
    ) -> None:
        """

        Stop task.

        Args:
            task (golf_federated.server.process.config.task.base.BaseTask): Corresponding Task object.

        """

        # Publish 'StopTrain' info.
        publish_stop_train(
            host=self.sse_host,
            port=self.sse_port,
            db=self.sse_db
        )
        task.task_stop = True
