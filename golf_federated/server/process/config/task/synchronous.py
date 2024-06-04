# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 13:18
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 13:18
import os

import pandas as pd
import time
from queue import Queue
import numpy as np

from golf_federated.server.process.strategy.selection.base import BaseSelect
from golf_federated.server.process.config.model.base import BaseModel
from golf_federated.server.process.config.task.base import BaseTask
from golf_federated.server.process.strategy.aggregation.base import BaseFed
from golf_federated.server.process.strategy.evaluation.base import BaseEval


class SyncTask(BaseTask):
    """

    Synchronous Task object class, inheriting from Task class.

    """

    def __init__(
        self,
        task_name: str,
        maxround: int,
        aggregation: BaseFed,
        evaluation: BaseEval,
        model: BaseModel,
        select: BaseSelect,
        module_path: str = '',
        isdocker: bool = False,
        image_name: str = ''
    ) -> None:
        """

        Initialize the Synchronous Task object.

        Args:
            task_name (str): Name of the task.
            maxround (int): Maximum number of aggregation rounds.
            aggregation (golf_federated.server.process.strategy.aggregation.base.BaseFed): Aggregation strategy object.
            evaluation (golf_federated.server.process.strategy.evaluation.base.BaseEval): Evaluation strategy object.
            model (golf_federated.server.process.config.model.base.BaseModel): Model object.
            select (golf_federated.server.process.strategy.selection.base.BaseSelect): Select strategy object.
            module_path (str): File path to model module. Default as ''.
            isdocker (bool): Whether the task requires Docker. Default as False.
            image_name (str): Name of Docker image. Default as ''.

        """

        # Super class init.
        super().__init__(task_name, maxround, True, aggregation, evaluation, model, select, module_path, isdocker,
                         image_name)

    def start_aggregation(
        self,
        aggregation_parameter: Queue
    ) -> bool:
        """

        Judge whether the conditions for starting aggregation have been met.

        Args:
            aggregation_parameter (queue.Queue): Queue for storing aggregated parameters.

        Returns:
            Bool: Whether to start aggregation.

        """

        # The number of uploaded clients meets the requirements and all clients of the task are uploaded.
        if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= len(
            self.client_list):
            return True

        else:
            return False


class CedarTask(BaseTask):
    """

    Cedar Task object class, inheriting from Task class.

    """

    def __init__(
        self,
        task_name: str,
        maxround: int,
        aggregation: BaseFed,
        model: BaseModel,
        select: BaseSelect,
        dataset: str,
        last_path: str,
        path_now: str,
        evaluation: BaseEval,
        module_path: str = '',
        isdocker: bool = False,
        image_name: str = ''
    ) -> None:
        """

        Initialize the Synchronous Task object.

        Args:
            task_name (str): Name of the task.
            maxround (int): Maximum number of aggregation rounds.
            aggregation (golf_federated.server.process.strategy.aggregation.base.BaseFed): Aggregation strategy object.
            evaluation (golf_federated.server.process.strategy.evaluation.base.BaseEval): Evaluation strategy object.
            model (golf_federated.server.process.config.model.base.BaseModel): Model object.
            select (golf_federated.server.process.strategy.selection.base.BaseSelect): Select strategy object.
            module_path (str): File path to model module. Default as ''.
            isdocker (bool): Whether the task requires Docker. Default as False.
            image_name (str): Name of Docker image. Default as ''.

        """

        # Super class init.
        super().__init__(task_name, maxround, True, aggregation, evaluation, model, select, module_path, isdocker,
                         image_name)
        self.loss_list = []
        self.acc_list = []
        self.recall_list = []
        self.pre_list = []
        self.f1_list = []
        self.mcc_list = []
        self.dataset = dataset
        self.last_path = last_path
        self.path_now = path_now
        self.evaluation_client = []

    def start_aggregation(
        self,
        aggregation_parameter: Queue
    ) -> bool:
        """

        Judge whether the conditions for starting aggregation have been met.

        Args:
            aggregation_parameter (queue.Queue): Queue for storing aggregated parameters.

        Returns:
            Bool: Whether to start aggregation.

        """

        # The number of uploaded clients meets the requirements and all clients of the task are uploaded.
        if aggregation_parameter.qsize() >= self.aggregation.min_to_start and aggregation_parameter.qsize() >= len(
            self.client_list):
            return True

        else:
            return False

    def run_aggregation(
        self,
        aggregation_parameter: Queue
    ) -> bool:
        """

        Run global model aggregation.

        Args:
            aggregation_parameter (queue.Queue): Queue for storing aggregated parameters.

        Returns:
            Bool: Whether aggregation is executed.

        """

        # Judge whether to start aggregation.
        if self.start_aggregation(aggregation_parameter=aggregation_parameter):
            # Run global model aggregation.
            self.model.model_aggre(aggregation=self.aggregation, parameter=aggregation_parameter,
                                   record=self.evaluation.get_record())
            return True

        else:
            # Conditions for starting aggregation have not been met.
            return False

    def run_evaluation(self) -> bool:
        """

        Run global model evaluation.

        Returns:
            Bool: Evaluation result, indicating the continuation or completion of the task.

        """

        id_test = list(range(len(self.evaluation_client)))
        final_init_loss, final_init_acc, final_init_pre \
            , final_init_recall, final_init_f1 = [], [], [], [], []
        final_init_mcc = []
        for a, id in enumerate(id_test):
            self.evaluation_client[id].update_model(self.model.model)
            init_loss, init_acc, init_pre, init_recall, init_f1, init_mcc = self.evaluation_client[id].trainer.test()
            final_init_loss.append(init_loss)
            final_init_acc.append(init_acc)
            final_init_pre.append(init_pre)
            final_init_recall.append(init_recall)
            final_init_f1.append(init_f1)
            final_init_mcc.append(init_mcc)

        last_path = self.last_path
        loss_init_file_path = self.path_now + "/result_save/" + self.dataset + "/loss" + last_path
        acc_init_file_path = self.path_now + "/result_save/" + self.dataset + "/acc" + last_path
        pre_init_file_path = self.path_now + "/result_save/" + self.dataset + "/pre" + self.last_path
        recall_init_file_path = self.path_now + "/result_save/" + self.dataset + "/recall" + self.last_path
        f1_init_file_path = self.path_now + "/result_save/" + self.dataset + "/f1" + self.last_path
        mcc_init_file_path = self.path_now + "/result_save/" + self.dataset + "/mcc" + self.last_path
        if not os.path.exists(self.path_now + "/result_save"):
            os.mkdir(self.path_now + "/result_save")
        if not os.path.exists(self.path_now + "/result_save/" + self.dataset):
            os.mkdir(self.path_now + "/result_save/" + self.dataset)
        if not os.path.exists(self.path_now + "/result_save/" + self.dataset + "/loss"):
            os.mkdir(self.path_now + "/result_save/" + self.dataset + "/loss")
        if not os.path.exists(self.path_now + "/result_save/" + self.dataset + "/acc"):
            os.mkdir(self.path_now + "/result_save/" + self.dataset + "/acc")
        if not os.path.exists(self.path_now + "/result_save/" + self.dataset + "/pre"):
            os.mkdir(self.path_now + "/result_save/" + self.dataset + "/pre")
        if not os.path.exists(self.path_now + "/result_save/" + self.dataset + "/recall"):
            os.mkdir(self.path_now + "/result_save/" + self.dataset + "/recall")
        if not os.path.exists(self.path_now + "/result_save/" + self.dataset + "/f1"):
            os.mkdir(self.path_now + "/result_save/" + self.dataset + "/f1")
        if not os.path.exists(self.path_now + "/result_save/" + self.dataset + "/mcc"):
            os.mkdir(self.path_now + "/result_save/" + self.dataset + "/mcc")
        if not os.path.exists(loss_init_file_path):
            os.mkdir(loss_init_file_path)
        if not os.path.exists(acc_init_file_path):
            os.mkdir(acc_init_file_path)
        if not os.path.exists(pre_init_file_path):
            os.mkdir(pre_init_file_path)
        if not os.path.exists(recall_init_file_path):
            os.mkdir(recall_init_file_path)
        if not os.path.exists(f1_init_file_path):
            os.mkdir(f1_init_file_path)
        if not os.path.exists(mcc_init_file_path):
            os.mkdir(mcc_init_file_path)

        loss_init_path = os.path.join(loss_init_file_path, "{}.npy".format(self.aggregation.aggregation_version))
        acc_init_path = os.path.join(acc_init_file_path, "{}.npy".format(self.aggregation.aggregation_version))
        pre_init_path = os.path.join(pre_init_file_path, "{}.npy".format(self.aggregation.aggregation_version))
        recall_init_path = os.path.join(recall_init_file_path, "{}.npy".format(self.aggregation.aggregation_version))
        f1_init_path = os.path.join(f1_init_file_path, "{}.npy".format(self.aggregation.aggregation_version))
        mcc_init_path = os.path.join(mcc_init_file_path, "{}.npy".format(self.aggregation.aggregation_version))

        if self.aggregation.aggregation_version % 5 == 0:
            np.save(loss_init_path, final_init_loss)
            np.save(acc_init_path, final_init_acc)
            np.save(pre_init_path, final_init_pre)
            np.save(recall_init_path, final_init_recall)
            np.save(f1_init_path, final_init_f1)
            np.save(mcc_init_path, final_init_mcc)
        if self.aggregation.aggregation_version % 20 == 0 or self.aggregation.aggregation_version <= 10:
            print('loss/acc/pre/recall/f1/mcc: ', np.mean(final_init_loss), np.mean(final_init_acc),
                  np.mean(final_init_pre), np.mean(final_init_recall), np.mean(final_init_f1), np.mean(final_init_mcc))

        self.loss_list.append(np.mean(final_init_loss))
        self.acc_list.append(np.mean(final_init_acc))
        self.pre_list.append(np.mean(final_init_pre))
        self.recall_list.append(np.mean(final_init_recall))
        self.f1_list.append(np.mean(final_init_f1))
        self.mcc_list.append(np.mean(final_init_mcc))

        return self.aggregation.aggregation_version >= self.maxround

    def run_localization(self, local_test_epoch) -> bool:
        """

        Run model localization.

        """

        id_test = list(range(len(self.evaluation_client)))
        avg_loss_list = np.zeros([local_test_epoch])
        avg_acc_list = np.zeros([local_test_epoch])
        avg_pre_list = np.zeros([local_test_epoch])
        avg_recall_list = np.zeros([local_test_epoch])
        avg_f1_list = np.zeros([local_test_epoch])
        avg_mcc_list = np.zeros([local_test_epoch])
        for id, j in enumerate(id_test):
            self.evaluation_client[j].update_model(self.model.model)
        for epoch in range(local_test_epoch):
            test_loss_list = np.zeros([len(self.evaluation_client)])
            test_acc_list = np.zeros([len(self.evaluation_client)])
            test_pre_list = np.zeros([len(self.evaluation_client)])
            test_recall_list = np.zeros([len(self.evaluation_client)])
            test_f1_list = np.zeros([len(self.evaluation_client)])
            test_mcc_list = np.zeros([len(self.evaluation_client)])
            for a, id in enumerate(id_test):
                test_loss_list[id], test_acc_list[id], test_pre_list[id], test_recall_list[id], test_f1_list[id], \
                test_mcc_list[id] = self.evaluation_client[id].trainer.localize()
            avg_loss_list[epoch] = np.mean(test_loss_list)
            avg_acc_list[epoch] = np.mean(test_acc_list)
            avg_pre_list[epoch] = np.mean(test_pre_list)
            avg_recall_list[epoch] = np.mean(test_recall_list)
            avg_f1_list[epoch] = np.mean(test_f1_list)
            avg_mcc_list[epoch] = np.mean(test_mcc_list)
        result = list(
            zip(avg_loss_list, avg_acc_list, avg_pre_list, avg_recall_list, avg_f1_list, avg_mcc_list))  # -------
        dataframe = pd.DataFrame(result, columns=['loss', 'acc', 'pre', 'recall', 'f1', 'mcc'])  # --------
        return dataframe

    def select_clients(self) -> list:
        """

        Select clients.

        Returns:
            List: Selected clients.

        """

        return self.select.select()

    def weight_tofile(self, save_path) -> None:
        """

        Save model weight to zip.

        """

        # Save model weight.

        import torch
        torch.save(self.model.model.state_dict(), save_path)

    def save_result(self, save_path):
        """

        Save evaulation result.

        """

        result = list(zip(self.loss_list, self.acc_list, self.pre_list, self.recall_list, self.f1_list, self.mcc_list))
        dataframe = pd.DataFrame(result,
                                 columns=['loss', 'acc', 'pre', 'recall', 'f1', 'mcc'])
        dataframe.to_excel(save_path, index=False)

    def save_result_layer(self, save_path):
        """

        Save result of layer.

        """

        result = list(self.aggregation.layer_num_list)
        dataframe = pd.DataFrame(result)
        dataframe.to_excel(save_path, index=False)
