Client
===========


Dataset
-------

You are required to provide training and testing sets.

Simply put the four types of paths (data and labels of the trainning set,
data and labels of the testing set) in four arrays, and pass them to the
``CustomFederatedDataset``. The ``CustomFederatedDataset`` also needs
a client id, which is used to identify the client, you can generate it
in random if you like.

.. code-block:: python

    from golf_federated.utils.data import CustomFederatedDataset


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


Connection and Run
------------------

Set the host and port of the server in the ``MultiDeviceClient`` class,
after the class is initialized, the connection will be established immediately,
and the client will wait for the server to send the task.

.. code-block:: python

    from golf_federated.client.process.config.device.multidevice import MultiDeviceClient


    data_client_n = mnist_fl_data.get_part_train(client_id[0])
    client_1 = MultiDeviceClient(
        client_name=client_id[0],
        api_host='127.0.0.1',
        api_port='7788',
        train_data=data_client_n[0],
        train_label=data_client_n[1],
    )
