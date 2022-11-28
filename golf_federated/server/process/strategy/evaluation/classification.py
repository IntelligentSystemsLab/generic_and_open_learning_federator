# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:50
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:50

from numpy import ndarray

from golf_federated.server.process.strategy.evaluation.base import BaseEval
from golf_federated.utils.log import loggerhear
from golf_federated.utils.data import onehot_to_label
from golf_federated.server.process.strategy.evaluation.function import accuracy


class Accuracy(BaseEval):
    """

    Accuracy of classification problems, inheriting from BaseEval class.

    """

    def __init__(
            self,
            target
    ) -> None:
        """

        Initialize the Accuracy object.

        Args:
            target (float): Target of the indicator.

        """

        # Super class init.
        super().__init__(
            name='accuracy',
            question_type='classification',
            positive=True,
            target=target
        )
        loggerhear.log("Server Info  ", "Evaluate %s question with %s." % (self.question_type, self.name))

    def eval(
            self,
            target: ndarray,
            prediction: ndarray,
    ) -> float:
        """

        Calculation of evaluation metric.

        Args:
            target (numpy.ndarray): Ground truth.
            prediction (numpy.ndarray): Prediction result.

        Returns:
            Float: Accuracy.

        """

        # Unify data with labels.
        target_label = onehot_to_label(target)
        prediction_label = onehot_to_label(prediction)

        # Calling calculation function.
        result = accuracy(
            target=target_label,
            prediction=prediction_label
        )

        # Record the evaluation result.
        self.record.append(result)
        loggerhear.log("Server Info  ", "Evaluate result of %s is %f." % (self.name, result))

        return result
