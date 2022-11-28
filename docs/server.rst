Server
===========


Dataset
-------

You can prepare some dataset for testing the performance of the model.
You should make your data in two ``.npy`` format files, one for the data,
the other for the labels, both of which should be ``array-like``, in the same order.

Here's an example:

.. code-block:: python

    from golf_federated.utils.data import CustomFederatedDataset

    data_dir = '../../../../data/non_iid_data_mnist_range5_label_client3/'
    x_test = [data_dir + 'x_test_%s.npy' % (str(i + 1)) for i in range(3)]
    y_test = [data_dir + 'y_test_%s.npy' % (str(i + 1)) for i in range(3)]
    mnist_fl_data = CustomFederatedDataset(
        test_data=x_test,
        test_label=y_test,
    )


Connection
----------

A server opens two ports, an API port and a SSE port. The SSE port is used to
send events to the client. The API port is used to send data from the client
to the server.

Example:

.. code-block:: python

    from golf_federated.server.process.config.device.multidevice import MultiDeviceServer


    server = MultiDeviceServer(
        server_name='server1',
        api_host='127.0.0.1',
        api_port='7788',
        sse_host='127.0.0.1',
        sse_port='6379',
        sse_db=6,
    )


Model
-----

In the model python file, you should define a function to generate the model.
At the end of the file, you should define the following variables:

- ``model``: the name of the function to generate the model
- ``optimizer``
- ``loss``
- ``batch_size``
- ``train_epoch``
- ``library``: the library you use, either ``tensorflow`` or ``pytorch``
- ``metrics``: a list of metrics to evaluate the model

Example:

.. code-block:: python

    import tensorflow as tf


    def create_cnn_for_mnist():
        return tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, kernel_size=(5, 5),
                    activation='relu',
                    input_shape=(28, 28, 1),
                ),
                tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation='softmax'),
            ]
        )


    model = 'create_cnn_for_mnist'
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.03)
    loss = tf.keras.losses.CategoricalCrossentropy()
    batch_size = 128
    train_epoch = 1
    library = 'tensorflow'
    metrics = ["accuracy"]


Task
----

The other configurations like max round, FL aggregation, evaluating method,
client selection method, etc. are defined in the task class.

Here's the options supported currently:

- FL aggregation: ``FedAvg``, ``FedProx``, ``SLMFed_syn`` (defined in ``golf_federated.server.process.strategy.aggregation.synchronous``)
- Evaluation: ``MSE``, ``Accuracy`` (defined in ``golf_federated.server.process.strategy.evaluation``)
- Selection method: ``RandomSelect``, ``AllSelect``

Example:

.. code-block:: python

    import tfmodule as module # your self defined model python file
    from golf_federated.server.process.strategy.aggregation.synchronous import FedAVG # `FedProx` and `SLMFed_syn` are also available
    from golf_federated.server.process.strategy.evaluation.classification import Accuracy
    # If it's a regression problem, you can use `MSE` instead
    # from golf_federated.server.process.strategy.evaluation.regression import MSE
    from golf_federated.server.process.strategy.selection.nonprobbased import AllSelect # `RandomSelect` is also available


    task = SyncTask(
        task_name='task1',
        maxround=5,
        aggregation=FedAVG(min_to_start=1),
        evaluation=Accuracy(target=0.9),
        model=TFMserver(
            module=module,
            test_data=mnist_fl_data.test_data,  # `mnist_fl_data` is defined above
            test_label=mnist_fl_data.test_label,
            process_unit='/cpu:0'
        ),
        select=AllSelect(
            client_list=[],
            select_num=2
        ),
        module_path='../../../../module/MNIST/tfmodule.py',
        isdocker=False,
    )


Run
---

After the configurations are done, you can run the server by:

.. code-block:: python

    import threading


    def start_task():
        while True:
            if len(server.client_pool) >= 2: # `server` is defined above
                server.start_task(
                    task=task,
                )
                break
    
    start_thread = threading.Thread(target=start_task)
    start_thread.start()

    server.start_server()
    

After the server is started, you can start the client to connect to the server.
As soon as at least two clients are connected, the server will start the task,
distribute the model to the clients, and start the training process.
