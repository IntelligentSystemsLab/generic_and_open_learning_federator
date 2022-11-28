# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:13
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:13

from golf_federated.server.process.config.device.base import BaseServer
from golf_federated.server.communication.api.app import app


def init_sse(
        server: BaseServer,
        host: str,
        port: str,
        db: int = 6
) -> object:
    """

    Args:
        server (golf_federated.server.process.config.device.base.BaseServer): Corresponding Server object.
        host (str): Host name to connect to the SSE host.
        port (str): Port number to connect to the SSE host.
        db (int): Adopted Database.

    Returns:
        object: APP config object.

    """

    # Configure the server and related setting into flask app.
    class Config(object):
        app.config['SERVER_' + server.server_name] = server
        app.config['SERVER_' + server.server_name + 'REDIS_HOST'] = host
        app.config['SERVER_' + server.server_name + 'REDIS_PORT'] = port
        app.config['SERVER_' + server.server_name + 'REDIS_DB'] = db

    return Config
