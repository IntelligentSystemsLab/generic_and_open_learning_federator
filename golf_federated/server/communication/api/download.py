# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 13:14
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 13:14

from flask import Response

from golf_federated.server.utils.cost import sim_cost
from golf_federated.utils.log import loggerhear

# ToDo: Predefined server and task name, database will be introduced later.

server_name = 'server1'
task_name = 'task1'


def download_model(serverhere: object) -> Response:
    """

    Model download method for API.

    Args:
        serverhere (golf_federated.server.process.config.device.base.MultiDeviceServer): Server object.

    Returns:
        Response: Model weight file stream.

    """

    # TODO: Judge the client that sent the request.
    loggerhear.log("Server Info  ", "Server %s is being request to download the global model." % serverhere.server_name)

    # Get the Task object.
    task = serverhere.get_task(task_name)

    # Get file path to global model.
    file_path = task.weight_path

    # Calculate communication cost.
    task.round_cost += sim_cost(
        data=file_path,
        file_path=True,
        communication_num=1
    )

    return Response(send_chunk(file_path), content_type='application/octet-stream')


def download_info(serverhere: object) -> Response:
    """

    Task info download method for API.

    Args:
        serverhere (golf_federated.server.process.config.device.base.MultiDeviceServer): Server object.

    Returns:
        Response: Task info file stream.

    """

    # TODO: Judge the client that sent the request.
    loggerhear.log("Server Info  ", "Server %s is being request to download the task info." % serverhere.server_name)

    # Get the Task object.
    task = serverhere.get_task(task_name)

    # Get file path to task info.
    file_path = task.info_path

    # Calculate communication cost.
    task.round_cost += sim_cost(
        data=file_path,
        file_path=True,
        communication_num=1
    )

    return Response(send_chunk(file_path), content_type='application/octet-stream')


def send_chunk(file_path: str):
    """

    File stream transfer.

    Args:
        file_path (str): File path to transfer.

    """

    with open(file_path, 'rb') as target_file:
        while True:
            chunk = target_file.read(2 * 1024 * 1024)
            if not chunk:
                break
            yield chunk
