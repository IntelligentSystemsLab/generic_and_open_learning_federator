# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/3 12:43
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/3 12:43

from abc import abstractmethod
from numpy import ndarray


class BaseEval(object):
    """

    Evaluation strategy base class.

    """

    def __init__(
            self,
            name: str,
            question_type: str,
            positive: bool,
            target: float,
            convergence: float = 0.001
    ):
        """

        Initialize the base class object of the evaluation strategy, which is called when subclasses inherit.

        Args:
            name (str): Name of the evaluation strategy.
            question_type (str): Type of evaluation task.
            positive (bool): Whether it is a positive indicator.
            target (float): Target of the indicator.
            convergence (float): Convergence precision.

        """

        # Initialize object properties.
        self.name = name
        self.question_type = question_type
        self.positive = positive
        self.target = target
        self.convergence = convergence
        self.record = []

    @abstractmethod
    def eval(
            self,
            target: ndarray,
            prediction: ndarray,
    ):
        """

        Abstract method for calculation of evaluation metrics.

        Args:
            target (numpy.ndarray): Ground truth.
            prediction (numpy.ndarray): Prediction result.

        """

        pass

    def reach_target(self) -> bool:
        """

        Judge whether the target is reached.

        Returns:
            Bool: Whether the target is reached.

        """

        # Judge whether the indicator is positive or negative.
        if self.positive:
            # Positive.
            return self.record[-1] > self.target

        else:
            # Negative.
            return self.record[-1] < self.target

    def reach_convergence(self) -> bool:
        """

        Judge whether convergence.

        Returns:
            Bool: Whether convergence.

        """

        # Judgment begins after ten rounds
        if len(self.record) >= 10:
            # Convergence in the range of 10 rounds.
            return max(self.record[-10:]) - min(self.record[-10:]) < self.convergence

        else:
            return False

    def get_record(self) -> list:
        """

        Get the evaluation record.

        Returns:
            List: Evaluation record.

        """

        return self.record
