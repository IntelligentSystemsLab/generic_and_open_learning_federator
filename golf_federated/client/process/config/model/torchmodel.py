# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/14 16:00
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/14 16:00

import torch
from numpy import ndarray
from torch.utils.data.dataloader import DataLoader

from golf_federated.utils.log import loggerhear
from golf_federated.client.process.config.model.base import BaseModel
from golf_federated.utils.torchutils import simple_dataset


class TorchModel(BaseModel):
    """

    PyTorch Model object class, inheriting from Model class.

    """

    def __init__(
        self,
        module: object,
        train_data: ndarray,
        train_label: ndarray,
        process_unit: str = "cpu"
    ) -> None:
        """

        Initialize the PyTorch Model object.

        Args:
            module (object): Model module, including predefined model structure, loss function, optimizer, etc.
            train_data (numpy.ndarray): Data values for training.
            train_label (numpy.ndarray): Data labels for training.
            process_unit (str): Processing unit to perform local training. Default as "cpu".

        """

        # Super class init.
        super().__init__(module, train_data, train_label, process_unit)
        self.optimizer = module.optimizer(self.model.parameters(), lr=module.learning_rate)

        # Initialize object properties.
        if train_data.shape[-1] == 1:
            self.train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1], train_data.shape[2])
        if module.torch_dataset is not None:
            self.train_dataset = module.torch_dataset(data=self.train_data, label=self.train_label)
        else:
            self.train_dataset = simple_dataset(data=self.train_data, label=self.train_label)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size)
        self.file_ext = '.pt'
        self.process_unit = torch.device(self.process_unit)

    def train(self) -> None:
        """

        Model training.

        """

        # Switch mode of model.
        self.model.train()

        # Client model training.
        for epoch in range(self.train_epoch):
            training_loss = 0.0
            training_acc = 0.0
            training_count = 0
            training_total = 0
            for data in self.train_dataloader:
                input = data[1].float().to(self.process_unit)
                label = data[0].float().to(self.process_unit)
                self.optimizer.zero_grad()
                output = self.model(input)
                if type(self.loss) == type(torch.nn.CrossEntropyLoss()):
                    label = torch.argmax(label, -1)
                loss = self.loss(output, label.long())
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
                _, pre = torch.max(output.data, dim=1)
                training_acc += (pre == label).sum().item()
                training_count += 1
                training_total += label.shape[0]

            loggerhear.log("Client Info  ", 'Epoch [%d/%d]:    Loss: %.4f       Accuracy: %.2f' % (
                epoch, self.train_epoch, training_loss / training_count, training_acc / training_total * 100))

    def predict(
        self,
        data: ndarray
    ) -> ndarray:
        """

        Model prediction.

        Args:
            data (numpy.ndarray): Data values for prediction.

        Returns:
            Numpy.ndarray: Prediction result.

        """

        with torch.no_grad():
            imput = data.to(self.process_unit)
            result = self.model(imput).cpu().numpy()
        return result
