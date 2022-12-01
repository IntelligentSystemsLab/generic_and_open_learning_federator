# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/14 16:00
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/14 16:00

import requests

from golf_federated.utils.log import loggerhear


def client_register(
        host: str,
        port: str,
        client_name: str
) -> bytes:
    """

    Client registration.

    Args:
        host (str): Host name to connect to the host.
        port (str): Port number to connect to the host.
        client_name (str): Client name to register.

    Returns:
        Bytes: Request information.

    """

    # Send a request to the server.
    loggerhear.log(
        "Client Info  ",
        "Client %s is registering." % client_name
    )
    r = requests.post(
        "http://" + host + ":" + str(port) + "/client-register",
        data={
            'name': client_name,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    ).content

    return r
