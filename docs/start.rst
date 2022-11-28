Getting Started
===============


Installation
------------

To install the latest version of the package, run the following command::

    pip install golf_federated

Alternatively, you can clone the repository and install the package manually::
  
    git clone git://github.com//golf_federated.git
    cd golf_federated
    python setup.py install


Quick Start
-----------

Here's a quick example of how to use the package. For more details, see the Configure section.


Configure your model
^^^^^^^^^^^^^^^^^^^^

You have two ways to configure your model, either by a yaml file or by a python file.

Here is how to configure your model by a python file:

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


Server code
^^^^^^^^^^^^^^^^

Configure and set up your server.

.. code-block:: python

    import warnings

    warnings.filterwarnings("ignore", category=Warning)

    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from golf_federated.server.process.strategy.evaluation.classification import Accuracy

    from golf_federated.server.process.strategy.aggregation.synchronous import FedAVG
    from golf_federated.server.process.strategy.selection.nonprobbased import AllSelect

    import sys

    sys.path.append('../../../../module/MNIST/')

    import tfmodule as module
    from golf_federated.utils.data import CustomFederatedDataset
    from golf_federated.server.process.config.task.synchronous import SyncTask
    from golf_federated.server.process.config.model.tfmodel import TensorflowModel as TFMserver
    from golf_federated.server.process.config.device.multidevice import MultiDeviceServer
    import threading


    def start_task():
        while True:
            if len(server.client_pool) >= 2:
                server.start_task(
                    task=task,
                )
                break


    if __name__ == '__main__':
        data_dir = '../../../../data/non_iid_data_mnist_range5_label_client3/'
        x_test = [data_dir + 'x_test_%s.npy' % (str(i + 1)) for i in range(3)]
        y_test = [data_dir + 'y_test_%s.npy' % (str(i + 1)) for i in range(3)]
        mnist_fl_data = CustomFederatedDataset(
            test_data=x_test,
            test_label=y_test,
        )
        server = MultiDeviceServer(
            server_name='server1',
            api_host='127.0.0.1',
            api_port='7788',
            sse_host='127.0.0.1',
            sse_port='6379',
            sse_db=6,
        )
        task = SyncTask(
            task_name='task1',
            maxround=5,
            aggregation=FedAVG(min_to_start=1),
            evaluation=Accuracy(target=0.9),
            model=TFMserver(
                module=module,
                test_data=mnist_fl_data.test_data,
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

        start_thread = threading.Thread(target=start_task)
        start_thread.start()

        server.start_server()


Client code
^^^^^^^^^^^^^^^^

Write some scripts to set up your client.

.. code-block:: python

    import warnings

    warnings.filterwarnings("ignore", category=Warning)

    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from golf_federated.utils.data import CustomFederatedDataset
    from golf_federated.client.process.config.device.multidevice import MultiDeviceClient

    if __name__ == '__main__':
        data_dir = '../../../../data/non_iid_data_mnist_range5_label_client3/'
        x_train = [data_dir + 'x_train_1.npy']
        y_train = [data_dir + 'y_train_1.npy']
        x_test = [data_dir + 'x_test_1.npy']
        y_test = [data_dir + 'y_test_1.npy']
        client_id = ['Client1']
        mnist_fl_data = CustomFederatedDataset(
            train_data=x_train,
            train_label=y_train,
            test_data=x_test,
            test_label=y_test,
            part_num=1,
            part_id=client_id,
            split_data=True,
        )
        data_client_n = mnist_fl_data.get_part_train(client_id[0])
        client_1 = MultiDeviceClient(
            client_name=client_id[0],
            api_host='127.0.0.1',
            api_port='7788',
            train_data=data_client_n[0],
            train_label=data_client_n[1],
        )

