# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/14 16:00
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/14 16:00
import gc
import random
from copy import deepcopy
from typing import List

import torch
import numpy as np
import torchvision
from numpy import ndarray
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef

from golf_federated.utils.data import DatasetSplit
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


class CedarModel(BaseModel):
    """

    Model object class in Cedar, inheriting from Model class.

    """

    def __init__(
        self,
        module: object,
        model: object,
        dataset: List,
        stimulus_x: ndarray = None,
        layer_fea: dict = None,
        process_unit: str = "cpu"
    ) -> None:
        """

        Initialize the PyTorch Model object.

        Args:
            module (object): Model module, including learning rate, loss function, optimizer, etc.
            model (object): Predefined model structure.
            dataset (List): Dataset list, consisting of "[[sample,label]]"
            stimulus_x (numpy.ndarray): Stimuli data.
            layer_fea (dict): E.g., for ResNet18, {'layer1': 'feat1', 'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'}
            process_unit (str): Processing unit to perform local training. Default as "cpu".

        """

        self.library = module.library
        self.model = model
        self.module = module
        self.loss = module.loss
        self.batch_size = module.batch_size
        self.train_epoch = module.train_epoch
        self.process_unit = process_unit
        self.divide_rate = module.divide_rate
        self.optimizer = module.optimizer(self.model.parameters(), lr=module.inner_learning_rate)
        self.outer_optimizer = module.optimizer(self.model.parameters(), lr=module.outer_learning_rate)
        divide_id = int(len(dataset) * self.divide_rate)
        support_set = DatasetSplit(dataset[:divide_id])
        self.support_loader = DataLoader(
            support_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        query_set = DatasetSplit(dataset[divide_id:])
        self.query_loader = DataLoader(
            query_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        if stimulus_x is not None:
            np.random.shuffle(stimulus_x)
            stimulus_x_temp = []
            for i in range(len(stimulus_x)):
                try:
                    stimulus_x_temp.append(stimulus_x[i].numpy())
                except:
                    stimulus_x_temp.append(stimulus_x[i])
            self.stimulus_x = torch.tensor(stimulus_x_temp)
            # if (self.dataset == 'fmnist' or self.dataset == 'fer') and self.net_name == 'densenet121':
            #     self.stimulus_x = torch.stack([self.stimulus_x, self.stimulus_x, self.stimulus_x], dim=1)
            #     self.stimulus_x = self.stimulus_x.squeeze(2)
        self.file_ext = '.pt'
        self.process_unit = torch.device(self.process_unit)
        self.deploy_epoch = module.deploy_epoch
        if layer_fea is not None:
            self.LAYER_FEA = layer_fea
            self.REQUIRE_JUDGE_LAYER = list(self.LAYER_FEA.keys())
            self.HOOK_RES = list(self.LAYER_FEA.values())
            self.NUM_LAYER = len(self.REQUIRE_JUDGE_LAYER)

        self.RCS = []
        self.never_selected = 1
        self.last_round_RC = []
        self.upgrade_bool = []
        self.upgrade_weight = None

    def train(self) -> None:
        """

        Model training.

        """
        
        self.global_net = deepcopy(self.model)

        self.loss = self.loss.to(self.process_unit)
        self.model = self.model.to(self.process_unit)
        self.optimizer = self.module.optimizer(self.model.parameters(), lr=self.module.inner_learning_rate)
        self.outer_optimizer = self.module.optimizer(self.model.parameters(), lr=self.module.outer_learning_rate)

        for epoch in range(self.train_epoch):
            i = 0
            gc.collect()
            torch.cuda.empty_cache()
            training_loss = 0.0
            training_count = 0
            training_total = 0
            for support, query in zip(self.support_loader, self.query_loader):
                support_x, support_y = support[0].to(self.process_unit), support[1].to(self.process_unit)
                query_x, query_y = query
                support_y = support_y.type(torch.LongTensor)
                query_y = query_y.type(torch.LongTensor)
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.process_unit)
                    support_y = support_y.cuda(self.process_unit)
                    query_x = query_x.cuda(self.process_unit)
                    query_y = query_y.cuda(self.process_unit)
                self.optimizer.zero_grad()
                output = self.model(support_x)
                loss = self.loss(output, support_y.long())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.outer_optimizer.zero_grad()
                output = self.model(query_x)
                loss = self.loss(output, query_y.long())
                loss.backward()
                self.outer_optimizer.step()
                self.outer_optimizer.zero_grad()
                i += 1
                gc.collect()
                torch.cuda.empty_cache()
                training_loss += loss.item()
                training_count += 1
                training_total += query_y.shape[0]
            loggerhear.log("Client Info  ", 'Epoch [%d/%d]:    Loss: %.4f' % (
                epoch, self.train_epoch, training_loss / training_total))
        self.loss = self.loss.to('cpu')
        self.model = self.model.to('cpu')
        del support_x, support_y, query_y, query_x
        gc.collect()
        torch.cuda.empty_cache()

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

    def test(self):
        """

        Model test in evaluation clients.

        Returns:
            Tuple[List,List,List,List,List,List]: Evaluation result, including Loss, Accuracy, Precision, Recall, F1-score, and Mcc.

        """

        self.loss = self.loss.to(self.process_unit)
        self.model = self.model.to(self.process_unit)

        for _ in range(self.deploy_epoch):
            for support in self.support_loader:
                self.optimizer.zero_grad()
                support_x, support_y = support
                support_y = support_y.type(torch.LongTensor)
                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.process_unit)
                    support_y = support_y.cuda(self.process_unit)
                output = self.model(support_x)
                output = torch.squeeze(output)
                loss = self.loss(output, support_y.long())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        loss_all, correct_all, total = 0.0, 0.0, 0.0
        precision_all, recall_all, f1_all = 0.0, 0.0, 0.0
        mcc_all = 0.0

        for query in self.query_loader:
            query_x, query_y = query
            query_y = query_y.type(torch.LongTensor)
            if torch.cuda.is_available():
                query_x = query_x.cuda(self.process_unit)
                query_y = query_y.cuda(self.process_unit)
            self.optimizer.zero_grad()
            output = self.model(query_x)
            output = torch.squeeze(output)
            loss = self.loss(output, query_y.long())
            loss_all += loss.item()
            total += len(query_y)

            y_hat = self.model(query_x)
            query_pred = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            correct = torch.eq(query_pred, query_y).sum().item()
            correct_all += correct
            query_y_c = query_y.cpu().numpy()  # ----
            query_pred_c = query_pred.cpu().numpy()  # ----
            precision = precision_score(query_pred_c, query_y_c, average='macro',zero_division=0)
            recall = recall_score(query_pred_c, query_y_c, average='macro',zero_division=0)
            f1 = f1_score(query_pred_c, query_y_c, average='macro',zero_division=0)
            mcc = matthews_corrcoef(query_pred_c, query_y_c)
            precision_all += precision
            recall_all += recall
            f1_all += f1
            mcc_all += mcc
        self.loss = self.loss.to('cpu')
        self.model = self.model.to('cpu')
        del query_y, query_x
        gc.collect()
        torch.cuda.empty_cache()
        init_loss_list = loss_all / len(self.query_loader)
        init_acc_list = correct_all / total
        init_pre_list = precision_all / len(self.query_loader)
        init_recall_list = recall_all / len(self.query_loader)
        init_f1_list = f1_all / len(self.query_loader)
        init_mcc_list = mcc_all / len(self.query_loader)

        return init_loss_list, init_acc_list, init_pre_list, init_recall_list, init_f1_list, init_mcc_list

    def localize(self):
        """

        Model localization.

        Returns:
            Tuple[numpy.array,numpy.array,numpy.array,numpy.array,numpy.array,numpy.array]: Evaluation result, including Loss, Accuracy, Precision, Recall, F1-score, and Mcc.

        """

        self.loss = self.loss.to(self.process_unit)
        self.model = self.model.to(self.process_unit)
        for support in self.support_loader:
            self.optimizer.zero_grad()
            support_x, support_y = support
            support_y = support_y.type(torch.LongTensor)
            if torch.cuda.is_available():
                support_x = support_x.cuda(self.process_unit)
                support_y = support_y.cuda(self.process_unit)
            output = self.model(support_x)
            output = torch.squeeze(output)
            loss = self.loss(output, support_y.long())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        loss_all, correct_all, total = 0.0, 0.0, 0.0
        precision_all, recall_all, f1_all = 0.0, 0.0, 0.0
        mcc_all = 0.0
        for query in self.query_loader:
            query_x, query_y = query
            query_y = query_y.type(torch.LongTensor)
            if torch.cuda.is_available():
                query_x = query_x.cuda(self.process_unit)
                query_y = query_y.cuda(self.process_unit)
            self.optimizer.zero_grad()
            output = self.model(query_x)
            output = torch.squeeze(output)
            loss = self.loss(output, query_y.long())
            loss_all += loss.item()

            y_hat = self.model(query_x)
            query_pred = F.softmax(y_hat, dim=1).argmax(dim=1)
            correct = torch.eq(query_pred, query_y).sum().item()
            correct_all += correct
            total += len(query_y)
            query_y_c = query_y.cpu().numpy()
            query_pred_c = query_pred.cpu().numpy()
            precision = precision_score(query_pred_c, query_y_c, average='macro',zero_division=0)
            recall = recall_score(query_pred_c, query_y_c, average='macro',zero_division=0)
            f1 = f1_score(query_pred_c, query_y_c, average='macro',zero_division=0)
            mcc = matthews_corrcoef(query_pred_c, query_y_c)
            precision_all += precision
            recall_all += recall
            f1_all += f1
            mcc_all += mcc
        self.loss = self.loss.to('cpu')
        self.model = self.model.to('cpu')
        del support_x, support_y, query_y, query_x
        gc.collect()
        torch.cuda.empty_cache()
        test_loss_list = loss_all / len(self.query_loader)
        test_acc_list = correct_all / total
        test_pre_list = precision_all / len(self.query_loader)
        test_recall_list = recall_all / len(self.query_loader)
        test_f1_list = f1_all / len(self.query_loader)
        test_mcc_list = mcc_all / len(self.query_loader)
        return np.array(test_loss_list), np.array(test_acc_list), np.array(test_pre_list), np.array(
            test_recall_list), np.array(test_f1_list), np.array(test_mcc_list)

    def local_stimulate(self):
        """

        Calculate the output of the stimuli for the local model in the current training round.

        """

        temp_local_net = deepcopy(self.model)
        Layer_fea = self.LAYER_FEA
        try:
            local_Layer_Getter = torchvision.models._utils.IntermediateLayerGetter(
                temp_local_net.features.to(self.process_unit),
                Layer_fea)
        except:
            local_Layer_Getter = torchvision.models._utils.IntermediateLayerGetter(temp_local_net.to(self.process_unit),
                                                                                   Layer_fea)

        self.local_stimulus_out = local_Layer_Getter(self.stimulus_x.to(self.process_unit))
        for hr in self.HOOK_RES:
            self.local_stimulus_out[hr] = self.local_stimulus_out[hr].cpu().detach().numpy()
        self.stimulus_x.to('cpu')
        del temp_local_net, local_Layer_Getter
        gc.collect()
        torch.cuda.empty_cache()

    def global_stimulate(self):
        """

        Calculate the output of the stimuli for the global model in the current training round.

        """

        temp_global_net = deepcopy(self.global_net)
        Layer_fea = self.LAYER_FEA
        try:
            global_Layer_Getter = torchvision.models._utils.IntermediateLayerGetter(
                temp_global_net.features.to(self.process_unit), Layer_fea)
        except:
            global_Layer_Getter = torchvision.models._utils.IntermediateLayerGetter(
                temp_global_net.to(self.process_unit),
                Layer_fea)

        self.global_stimulus_out = global_Layer_Getter(self.stimulus_x.to(self.process_unit))
        for hr in self.HOOK_RES:
            self.global_stimulus_out[hr] = self.global_stimulus_out[hr].cpu().detach().numpy()
        self.stimulus_x.to('cpu')
        del temp_global_net, global_Layer_Getter
        gc.collect()
        torch.cuda.empty_cache()

    def calculate_local_RDV(
        self,
        E_list: List
    ) -> List:
        """

        Calculate the RDV of the local model

        Args:
            E_list (List): Generated indexes for RDV

        Returns:
            List: RDV result.

        """

        RDV = []
        for fea in self.HOOK_RES:
            temp_RDV = []
            temp = self.local_stimulus_out[fea]
            for E_index in E_list:
                temp_RDV.append(np.linalg.norm(temp[E_index[0]] - temp[E_index[1]]))
            RDV.append(temp_RDV)
        return RDV

    def calculate_global_RDV(
        self,
        E_list: List
    ) -> List:
        """

        Calculate the RDV of the global model

        Args:
            E_list (List): Generated indexes for RDV

        Returns:
            List: RDV result.

        """

        RDV = []
        for fea in self.HOOK_RES:
            temp_RDV = []
            temp = self.global_stimulus_out[fea]
            for E_index in E_list:
                temp_RDV.append(np.linalg.norm(temp[E_index[0]] - temp[E_index[1]]))
            RDV.append(temp_RDV)
        return RDV

    def calculate_RCS(self):
        """

        Calculate the RCS of the global model

        """

        self.RCS = []
        E_list = self.generate_E(20)
        local_RDV = self.calculate_local_RDV(E_list)
        global_RDV = self.calculate_global_RDV(E_list)
        for i in range(self.NUM_LAYER):
            self.RCS.append(np.square(np.corrcoef(global_RDV[i], local_RDV[i])[0, 1]))
        self.upgrade_weight = self.model.state_dict()
        if self.never_selected == 1:
            self.upgrade_bool = [1 for i in range(self.NUM_LAYER)]
            self.upgrade_bool[np.random.randint(0, high=self.NUM_LAYER)] = 0
            self.last_round_RC = deepcopy(self.RCS)
            self.never_selected = 0
        else:
            last = deepcopy(self.last_round_RC)
            this = deepcopy(self.RCS)

            self.last_round_RC = deepcopy(self.RCS)
            temp_delta = []
            for i in range(self.NUM_LAYER):
                temp_delta.append(np.abs((last[i] - this[i]) / this[i]))
            self.upgrade_bool = [1 for i in range(self.NUM_LAYER)]
            min_indexes = np.argsort(temp_delta)[:min(3, int(self.NUM_LAYER / 4))]
            for idx in min_indexes:
                self.upgrade_bool[idx] = 0
                layer_name = self.REQUIRE_JUDGE_LAYER[idx]
                for k, v in self.upgrade_weight.items():
                    if layer_name in k:
                        self.upgrade_weight[k] = None

    def generate_E(
        self,
        E_number: int
    ) -> List:
        """

        Generate indexes for RDV

        Args:
            E_number (int): Number of indexes for RDV

        Returns:
            List: Generated indexes for RDV

        """

        i = 0
        E_list = []
        while i < E_number:
            E_element = [random.randint(0, 10), random.randint(0, 10)]
            if E_list.__contains__(E_element):
                continue
            else:
                E_list.append(E_element)
                i = i + 1
        return E_list

    def update_weight(
        self,
        new_model: object,
    ) -> None:
        """

        Update model weight.

        Args:
            new_model (object): Model weight for update.

        """
        for w, w_t in zip(self.model.parameters(), new_model.parameters()):
            w.data.copy_(w_t.data)
        self.global_net = deepcopy(self.model)

    def get_weight(self) -> List:
        """

        Get model weight.

        Returns:
            List: Model weight.

        """

        return self.model.state_dict()
