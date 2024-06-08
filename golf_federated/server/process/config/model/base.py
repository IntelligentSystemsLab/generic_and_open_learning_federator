# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 10:52
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 10:52

from abc import abstractmethod
from queue import Queue
from typing import List

from numpy import ndarray

from golf_federated.server.process.strategy.aggregation.base import BaseFed
from golf_federated.server.process.strategy.evaluation.base import BaseEval
from golf_federated.utils.model import get_model_parameter, set_model_parameter


class BaseModel(object):
    """

    Model object class, the class function supports the main operation of model on Server.

    """

    def __init__(
        self,
        module: object,
        test_data: ndarray,
        test_label: ndarray,
        process_unit: str
    ) -> None:
        """

        Initialize the Model object.

        Args:
            module (object): Model module, including predefined model structure, loss function, optimizer, etc.
            test_data (numpy.ndarray): Data values for evaluation.
            test_label (numpy.ndarray): Data labels for evaluation.
            process_unit: Processing unit to perform evaluation.

        """

        # Initialize object properties.
        self.model = getattr(module, module.model)()
        self.library = module.library
        self.test_data = test_data
        self.test_label = test_label
        self.process_unit = process_unit

    @abstractmethod
    def predict(self) -> ndarray:
        """

        Abstract method for model prediction.

        Returns:
            Numpy.ndarray: Prediction result.

        """

        pass

    def get_weight(self) -> List:
        """

        Get model weight.

        Returns:
            List: Model weight.

        """

        return get_model_parameter(
            model=self.model,
            library=self.library,
        )

    def update_weight(
        self,
        new_weight: List,
    ) -> None:
        """

        Update model weight.

        Args:
            new_weight (list): Model weight for update.

        """

        self.model = set_model_parameter(
            model=self.model,
            w=new_weight,
            library=self.library
        )

    def model_aggre(
        self,
        aggregation: BaseFed,
        parameter: Queue,
        record: List
    ) -> None:
        """

        Global model aggregation.

        Args:
            aggregation (golf_federated.server.process.strategy.aggregation.base.BaseFed): Aggregation strategy object.
            parameter (queue.Queue): Uploaded parameters.
            record (List): Records of evaluation.

        """

        # Call aggregation strategy object to aggregate the new global model weight.
        new_weight = aggregation.aggregate(
            {
                'current_w': self.get_weight(),
                'parameter': parameter,
                'record'   : record
            }
        )

        # Update global model weight.
        self.update_weight(new_weight=new_weight)

    def model_eval(
        self,
        evaluation: BaseEval
    ) -> None:
        """

        Global model evaluation.

        Args:
            evaluation (golf_federated.server.process.strategy.evaluation.base.BaseEval): Evaluation strategy object.

        """

        # Call evaluation strategy object to evaluate global model.
        evaluation.eval(
            target=self.test_label,
            prediction=self.predict()
        )
