# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/14 16:00
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/14 16:00

import json
import zipfile
from typing import List

import requests
import numpy as np

from golf_federated.utils.log import loggerhear


def upload_model(
    host: str,
    port: str,
    client_name: str,
    model: List,
    aggregation_field: dict
) -> bytes:
    """

    Client uploads local model and related files.

    Args:
        host (str): Host name to connect to the host.
        port (str): Port number to connect to the host.
        client_name (str): Client name to upload.
        model (List): Local model weight.
        aggregation_field (dict): Aggregate fields with corresponding values.

    Returns:
        Bytes: Request information.

    """

    # Save aggregate fields with corresponding values.
    data_dict = {
        'aggregation_field': aggregation_field
    }
    with open(client_name + "temp/upload_info.json", "w") as f:
        json.dump(data_dict, f)

    # Save local model weight.
    np.save(client_name + 'temp/local_weight.npy', model)

    # Save to zip file.
    model_zipper = zipfile.ZipFile(client_name + 'temp/upload.zip', 'w', compression=zipfile.ZIP_DEFLATED)
    model_zipper.write(
        filename=client_name + 'temp/local_weight.npy',
        arcname=client_name + 'local_weight.npy'
    )
    model_zipper.write(
        filename=client_name + 'temp/upload_info.json',
        arcname=client_name + 'upload_info.json'
    )
    model_zipper.close()

    # Compressed zip to binary file stream.
    binfile_first = open(client_name + 'temp/upload.zip', 'rb')
    bin_infomation = binfile_first.read()
    binfile_first.close()

    # Send a upload request to the server.
    loggerhear.log(
        "Client Info  ",
        "Client %s is uploading local model and related files." % client_name
    )
    r = requests.post(
        "http://" + host + ":" + str(port) + "/upload-model?name=%s" % client_name,
        data=bin_infomation,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    ).content

    return r
