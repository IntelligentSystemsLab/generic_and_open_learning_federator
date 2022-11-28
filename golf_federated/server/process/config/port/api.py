# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:12
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:12

from golf_federated.server.communication.api.app import app


def run_restful(
        config: object,
        host: str,
        port: str
) -> None:
    """

    The flask app is run directly here.
    This part will be modified after the web page is optimized.

    Args:
        config (object): Flask APP config.
        host (str): Uniform Resource Locator to connect to the host.
        port (str): Port number to connect to the host.

    """

    # APP config.
    app.config.from_object(config)

    # Run APP.
    app.run(threaded=True, debug=False, host=host, port=port)
