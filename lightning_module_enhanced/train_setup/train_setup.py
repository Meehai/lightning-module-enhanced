"""Module that implements a standard setup process of the lightning module via a train config file"""
from typing import Callable
from functools import partial
from torch import optim
from torch.nn import functional as F
from torchmetrics.functional.classification import accuracy
from torchmetrics.functional.regression.mae import mean_absolute_error
from torchmetrics.functional.regression.mse import mean_squared_error

from ..schedulers import ReduceLROnPlateauWithBurnIn
from ..logger import logger
from ..trainable_module import TrainableModule
from ..metrics import CallableCoreMetric

class TrainSetup:
    """
    Train Setup class.
    It will add the necessary attributes of a TrainableModule given a config file.
    The necessary attributes are: optimizer & criterion.
    The optional attributes are: scheduler, metrics & callbacks.
    """

    def __init__(self, module: TrainableModule, train_cfg: dict):
        assert isinstance(module, TrainableModule), f"Got {type(module)}"
        assert isinstance(train_cfg, dict), f"Got {type(train_cfg)}"
        self.module = module
        self.train_cfg = train_cfg
        self._setup()

    def _setup(self):
        """The main function of this class"""
        train_cfg = self.train_cfg
        assert train_cfg is not None

        if "criterion" in train_cfg:
            self.module.criterion_fn = TrainSetup.parse_criterion(train_cfg["criterion"])
        else:
            logger.warning("TrainSetup was called without 'criterion' key in the config")

        if "optimizer" in train_cfg:
            optimizer_type, optimizer_args = TrainSetup.parse_optimizer(train_cfg["optimizer"])
            self.module.optimizer = optimizer_type(self.module.parameters(), **optimizer_args)
        else:
            logger.warning("TrainSetup was called without 'optimizer' key in the config")

        if "scheduler" in self.train_cfg:
            assert "optimizer" in train_cfg, "Scheduler is defined but no optimizer is defined"
            scheduler_type, opt_args = TrainSetup.parse_scheduler(train_cfg["scheduler"])
            self.module.scheduler_dict = {"scheduler": scheduler_type(optimizer=self.module.optimizer), **opt_args}

        if "metrics" in train_cfg:
            self.module.metrics = TrainSetup.parse_metrics(train_cfg["metrics"])

        if "callbacks" in train_cfg:
            self.module.callbacks = TrainSetup.parse_callbacks(train_cfg["callbacks"])

    @staticmethod
    def parse_optimizer(cfg):
        """Parses the possible optimizers"""
        logger.debug(f"Setting optimizer from config: '{cfg['type']}'")
        optimizer_type = getattr(optim, cfg["type"])
        return optimizer_type, cfg["args"]

    @staticmethod
    def parse_criterion(cfg):
        """Parses the possible criteria"""
        # TODO: allow criterion provided as callable from outside
        logger.debug(f"Setting the criterion to: '{cfg['type']}' based on provided cfg.")
        criterion_type = {
            "mse": F.mse_loss,
            "l1": F.l1_loss,
            "cross_entropy": F.cross_entropy
        }[cfg["type"]]
        return criterion_type

    @staticmethod
    def parse_scheduler(cfg: dict):
        """Setup the scheduler following Pytorch Lightning's requirements."""
        assert "type" in cfg and "optimizer_args" in cfg and "monitor" in cfg["optimizer_args"], cfg
        scheduler_type = {
            "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
            "ReduceLROnPlateauWithBurnIn": ReduceLROnPlateauWithBurnIn,
        }[cfg["type"]]
        return partial(scheduler_type, **cfg["args"]), cfg["optimizer_args"]

    @staticmethod
    def parse_metrics(cfg: dict) -> dict[str, tuple[Callable, str]]:
        """Setup the metrics from the config file. Only a couple of them are available."""
        metrics = {}
        for metric_dict in cfg:
            metric_type, metric_args = metric_dict["type"], metric_dict.get("args", {})
            assert metric_type in ("accuracy", "l1", "mse"), metric_type
            if metric_type == "accuracy":
                assert "num_classes" in metric_args and "task" in metric_args
                metrics[metric_type] = CallableCoreMetric(metric_fn=partial(accuracy, **metric_args),
                                                          higher_is_better=True)
            if metric_type == "l1":
                assert metric_args == {}
                metrics[metric_type] = CallableCoreMetric(mean_absolute_error, higher_is_better=False)
            if metric_type == "mse":
                assert metric_args == {}
                metrics[metric_type] = CallableCoreMetric(mean_squared_error, higher_is_better=False)
        return metrics

    @staticmethod
    # pylint: disable=unused-argument
    def parse_callbacks(cfg: dict):
        """TODO: callbacks"""
        return None
