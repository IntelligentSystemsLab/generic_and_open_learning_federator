# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2022/11/10 13:30
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2022/11/10 13:30
from copy import deepcopy
from typing import List

from golf_federated.client.process.config.device.base import BaseClient
from golf_federated.client.process.config.trainer.base import BaseTrainer


class StandAloneClient(BaseClient):
    """

    Stand-Alone Client object class, inheriting from Client class.

    """

    def __init__(
        self,
        client_name: str,
    ) -> None:
        """

        Initialize the Stand-Alone Client object.

        Args:
            client_name (str): Name of the Client object.

        """

        # Super class init.
        super().__init__(
            client_name
        )

        # Initialize object properties.
        self.trainer = None

    def init_trainer(
        self,
        trainer: BaseTrainer
    ) -> None:
        """

        Initialize the Trainer.

        Args:
            trainer (golf_federated.client.process.config.trainer.base.BaseTrainer): Predefined Trainer object.

        """

        self.trainer = trainer

    def update_model(
        self,
        new_weight: List
    ) -> None:
        """

        Update local model weight.

        Args:
            new_weight (List): Model weight for update.

        """

        self.trainer.update_model(new_weight=new_weight)


class CedarClient(BaseClient):
    """

    Cedar Client object class, inheriting from Client class.

    """

    def __init__(
        self,
        client_name: str,
    ) -> None:
        """

        Initialize the Stand-Alone Client object.

        Args:
            client_name (str): Name of the Client object.

        """

        # Super class init.
        super().__init__(
            client_name
        )

        # Initialize object properties.
        self.trainer = None

    def init_trainer(
        self,
        trainer: BaseTrainer
    ) -> None:
        """

        Initialize the Trainer.

        Args:
            trainer (golf_federated.client.process.config.trainer.base.BaseTrainer): Predefined Trainer object.

        """

        self.trainer = trainer

    def update_model(
        self,
        new_weight: List
    ) -> None:
        """

        Update local model weight.

        Args:
            new_weight (List): Model weight for update.

        """

        self.trainer.model.model = deepcopy(new_weight)

    def get_field(self) -> dict:
        """

        Get data of specified fields for model aggregation.

        Returns:
            Dict: Data of specified fields for model aggregation.

        """

        # Initialize dictionary.
        field = dict()

        field['upgrade_bool'] = self.trainer.model.upgrade_bool
        field['NUM_LAYER'] = self.trainer.model.NUM_LAYER
        field['REQUIRE_JUDGE_LAYER'] = self.trainer.model.REQUIRE_JUDGE_LAYER

        return field
