# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/14 16:00
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/14 16:00

import json
import os
import zipfile
from typing import Tuple
import requests

from golf_federated.utils.log import loggerhear


def download_model(
        host: str,
        port: str,
        client_name: str
) -> str:
    """

    Client downloads global model.

    Args:
        host (str): Host name to connect to the host.
        port (str): Port number to connect to the host.
        client_name (str): Client name to download the global model.

    Returns:
        Str: File name of the global model.

    """

    # Send a download request to the server and get the file stream.
    loggerhear.log(
        "Client Info  ",
        "Client %s is downloading global model." % client_name
    )
    r = requests.get(
        "http://" + host + ":" + str(port) + "/download-model?name=" + str(client_name)
    ).content

    # Temporary folder.
    if not os.path.isdir(client_name + 'temp'):
        os.mkdir(client_name + 'temp')

    # Save the binary file stream as a zip.
    binfile = open(client_name + 'temp/weight.zip', 'wb')
    binfile.write(r)
    binfile.close()

    # Read the file.
    fz = zipfile.ZipFile(client_name + 'temp/weight.zip', 'r')
    filename = ''
    for file in fz.namelist():
        fz.extract(file, client_name + 'temp')
        filename = file
    fz.close()

    return filename


def download_info(
        host: str,
        port: str,
        client_name: str
) -> Tuple[bool, str, list]:
    """

    Client downloads task info.

    Args:
        host (str): Host name to connect to the host.
        port (str): Port number to connect to the host.
        client_name (str): Client name to download the task info.

    Returns:
        Tuple: Return as a tuple, including:
            isdocker (bool): Whether the task requires Docker.
            filename (str): File name of the task info.
            aggregation_field (list): Fields required for model aggregation.

    """

    # Send a download request to the server and get the file stream.
    loggerhear.log(
        "Client Info  ",
        "Client %s is downloading task info." % client_name
    )
    r = requests.get(
        "http://" + host + ":" + str(port) + "/download-info?name=" + str(client_name)
    ).content

    # Temporary folder.
    if not os.path.isdir(client_name + 'temp'):
        os.mkdir(client_name + 'temp')

    # Save the binary file stream as a zip.
    binfile = open(client_name + 'temp/info.zip', 'wb')
    binfile.write(r)
    binfile.close()

    # Read the files.
    fz = zipfile.ZipFile(client_name + 'temp/info.zip', 'r')
    isdocker = False
    filename = ''
    aggregation_field = []
    for file in fz.namelist():
        fz.extract(file, client_name + 'temp')

        # Judge the file name.
        if file == 'module.py':
            # Model module file for direct training.
            filename = client_name + 'temp.' + file.split('.')[0]
            isdocker = False

        elif file == 'task_info.json':
            # Json file of aggregated fields.
            f = open(client_name + 'temp/' + file, 'r')
            content = f.read()
            jsondata = json.loads(content)
            aggregation_field = jsondata['aggregationField']

        elif '.tar' in file:
            # Image file for Docker training.
            filename = client_name + 'temp/' + file
            isdocker = True
    fz.close()

    return isdocker, filename, aggregation_field
