# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 13:16
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 13:16

import redis

# ToDo: Predefined server name, database will be introduced later.
server_name = 'server1'


def server_sent_event(
        host: str,
        port: str,
        db: int
):
    """

    SSE for proactive server publishes messages.

    Args:
        host (str): Host name to connect to the SSE host.
        port (str): Port number to connect to the SSE host.
        db (int): Adopted Database.

    """

    # Redis Database.
    red = redis.StrictRedis(
        host=host,
        port=port,
        db=db
    )

    # Server publishes messages proactively.
    pubsub = red.pubsub()
    pubsub.subscribe('server')
    for message in pubsub.listen():
        '''
        if message['data'] == 1:
            pass
        else:
            print(message)
            print(type(message['data']))
            '''
        yield 'data: %s\n\n' % message['data']


def publish_update_model(
        host: str,
        port: str,
        db: int
) -> None:
    """

    Publish 'UpdateModel' info.

    Args:
        host (str): Host name to connect to the SSE host.
        port (str): Port number to connect to the SSE host.
        db (int): Adopted Database.

    """

    # Redis Database.
    red = redis.StrictRedis(
        host=host,
        port=port,
        db=db
    )

    # Server publishes messages proactively.
    red.publish('server', u'UpdateModel')


def publish_stop_train(
        host: str,
        port: str,
        db: int
) -> None:
    """

    Publish 'StopTrain' info.

    Args:
        host (str): Host name to connect to the SSE host.
        port (str): Port number to connect to the SSE host.
        db (int): Adopted Database.

    """

    # Redis Database.
    red = redis.StrictRedis(
        host=host,
        port=port,
        db=db
    )

    # Server publishes messages proactively.
    red.publish('server', u'StopTrain')


def publish_task_init(
        host: str,
        port: str,
        db: int
) -> None:
    """

    Publish 'TaskInit' info.

    Args:
        host (str): Host name to connect to the SSE host.
        port (str): Port number to connect to the SSE host.
        db (int): Adopted Database.

    """

    # Redis Database.
    red = redis.StrictRedis(
        host=host,
        port=port,
        db=db
    )

    # Server publishes messages proactively.
    red.publish('server', u'TaskInit')
