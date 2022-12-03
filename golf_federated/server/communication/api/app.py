# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/14 16:00
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/14 16:00

from flask import Flask, Response

# Initialize Flask app.
app = Flask(__name__)

from golf_federated.server.communication.sse.schedule import server_sent_event
from golf_federated.server.communication.api.download import download_info, download_model
from golf_federated.server.communication.api.interact import client_register
from golf_federated.server.communication.api.upload import upload_model

# ToDo: Predefined server and task name, database will be introduced later.
server_name = 'server1'
task_name = 'task1'


# API for clients to download model.
@app.route('/download-model', methods=['POST', 'GET'])
def downloadmodel():
    """

    API for clients to download model.

    """

    return download_model(app.config['SERVER_' + server_name])


# API for clients to download task info.
@app.route('/download-info', methods=['POST', 'GET'])
def downloadinfo():
    """

    API for clients to download task info.

    """

    return download_info(app.config['SERVER_' + server_name])


# API for clients to register.
@app.route('/client-register', methods=['POST', 'GET'])
def clientregister():
    """

    API for clients to register.

    """

    return client_register(app.config['SERVER_' + server_name])


# API for clients to upload model.
@app.route('/upload-model', methods=['POST', 'GET'])
def uploadmodel():
    """

    API for clients to upload model.

    """

    return upload_model(app.config['SERVER_' + server_name])


# API for clients to listen to the SSE channel.
@app.route('/sse')
def sse():
    """

    API for clients to listen to the SSE channel.

    """

    # TODO: Push messages selectively.
    return Response(
        server_sent_event(
            host=app.config['SERVER_' + server_name + 'REDIS_HOST'],
            port=app.config['SERVER_' + server_name + 'REDIS_PORT'],
            db=app.config['SERVER_' + server_name + 'REDIS_DB']
        ),
        mimetype="text/event-stream"
    )
