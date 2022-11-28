# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/14 16:00
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/14 16:00

import threading
from sseclient import SSEClient

from golf_federated.utils.log import loggerhear


def monitor(
        client: object,
        host: str,
        port: str,
        thread_name: str = 'monitor',

) -> None:
    """

    Create an information channel and listen to server pushes.

    Args:
        client (MultiDeviceClient): Client to listen.
        host (str): Host name to connect to the host.
        port (str): Port number to connect to the host.
        thread_name (str): Child thread name. Default as 'monitor'

    """

    # Define the thread running process function.
    def run_in_thread() -> None:
        """

        Child thread receives the information pushed by SSE(Server-Sent Events) client.

        """

        while True:
            # Receive information.
            try:
                messages = SSEClient("http://" + host + ":" + str(port) + "/sse")
            except:
                messages = []

            for msg in messages:
                # Judge information content.
                if msg.data == str(b'UpdateModel'):
                    # Update local model.
                    client.update_model()
                    client.train()
                    client.upload_local_weight()

                elif msg.data == str(b'StopTrain'):
                    # Stop local training.
                    client.stop()

                elif msg.data == str(b'TaskInit'):
                    # Init training task.
                    client.init_trainer()
                    client.train()
                    client.upload_local_weight()

    # Define the thread running process class.
    class ChildThread(threading.Thread):
        def __init__(
                self,
                name: str
        ) -> None:
            """

            Initialize child thread class object.

            Args:
                name (str):Child thread name.

            """

            threading.Thread.__init__(self)
            self.name = name

        def run(self) -> None:
            """

            Run the process of child thread.

            """

            run_in_thread()

    # Start child thread
    loggerhear.log(
        "Client Info  ",
        "Client %s is creating an information channel and listening to server pushes." % client.client_name
    )
    ChildThread(thread_name).start()
