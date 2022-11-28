# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 14:26
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 14:26

from flask import request

from golf_federated.utils.log import loggerhear


def client_register(serverhere) -> str:
    """

    Client register download method for API.

    Args:
        serverhere (golf_federated.server.process.config.device.base.MultiDeviceServer): Server object.

    Returns:

        Str: "success" info.

    """

    # TODO: Judge the client that sent the request.
    loggerhear.log("Server Info  ", "Server %s is being request to register clients." % serverhere.server_name)

    # Read request data.
    data = request.get_data(as_text=True)

    # Extract client name.
    client_name = data[5:]

    # Call the MultiDeviceServer object to perform client registration.
    serverhere.client_register(client_name=client_name)

    return 'success'
