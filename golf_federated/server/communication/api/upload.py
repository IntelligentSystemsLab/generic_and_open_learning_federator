# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 13:14
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 13:14

import json
import zipfile
import numpy as np
from flask import request

from golf_federated.server.utils.cost import sim_cost
from golf_federated.utils.log import loggerhear

# ToDo: Predefined server and task name, database will be introduced later.

server_name = 'server1'
task_name = 'task1'


def upload_model(serverhere) -> str:
    """

    Model upload method for API.

    Args:
        serverhere (golf_federated.server.process.config.device.base.MultiDeviceServer): Server object.

    Returns:

        Str: "success" info.

    """

    # TODO: Judge the client that sent the request.
    loggerhear.log("Server Info  ", "Server %s is being request to upload the local model." % serverhere.server_name)

    # Read request data.
    data = request.get_data()

    # Extract client name.
    name = request.args.get("name")

    # Get the Task object.
    task = serverhere.get_task(task_name)

    # Calculate communication cost.
    task.round_cost += sim_cost(
        data=data,
        communication_num=1
    )

    # Read and save binary file stream.
    binfile = open('temp/' + name + 'localweight.zip', 'wb')
    binfile.write(data)
    binfile.close()

    # Read the saved zip file.
    fz = zipfile.ZipFile('temp/' + name + 'localweight.zip', 'r')
    for file in fz.namelist():
        fz.extract(file, 'temp/')
        if file == name + 'local_weight.npy':
            # Local model weight.
            client_model = np.load('temp/' + file, allow_pickle=True)

        elif file == name + 'upload_info.json':
            # Aggregate fields with corresponding values.
            f = open('temp/' + file, 'r')
            content = f.read()
            jsondata = json.loads(content)
            client_aggregation_field = jsondata['aggregation_field']
    fz.close()

    # Call the MultiDeviceServer object to receive local uploading parameters.
    serverhere.receive_parameter(
        client_name=name,
        client_model=client_model,
        client_aggregation_field=client_aggregation_field,
        task=task
    )

    return 'success'
