"""Module that implements a standard setup process of the lightning module via a train config file"""
from typing import Dict
from functools import partial
from torch import optim
from torch.nn import functional as F
from torchmetrics.functional.classification import accuracy
from torchmetrics.functional.regression.mae import mean_absolute_error
from torchmetrics.functional.regression.mse import mean_squared_error

from ..schedulers import ReduceLROnPlateauWithBurnIn
from ..logger import logger
from ..trainable_module import TrainableModule

class TrainSetup:
    """
    Train Setup class.
    It will add the necessary attributes of a TrainableModule given a config file.
    The necessary attributes are: optimizer & criterion.
    The optional attributes are: scheduler, metrics & callbacks.
    """

    def __init__(self, module: TrainableModule, train_cfg: Dict):
        assert isinstance(module, TrainableModule), f"Got {type(module)}"
        assert isinstance(train_cfg, Dict), f"Got {type(train_cfg)}"
        self.module = module
        self.train_cfg = train_cfg
        self._setup()

    def _setup(self):
        """The main function of this class"""
        if self.train_cfg is None:
            logger.info("Train cfg is None. Returning early.")
            return

        self._setup_optimizer()
        self._setup_criterion()
        self._setup_scheduler()
        self._setup_metrics()
        self._setup_callbacks()

    def _setup_optimizer(self):
        assert "optimizer" in self.train_cfg, "Optimizer not in train cfg"
        cfg: Dict = self.train_cfg["optimizer"]

        logger.debug(f"Setting optimizer from config: '{cfg['type']}'")
        optimizer_type = {
            "adamw": optim.AdamW,
            "adam": optim.Adam,
            "sgd": optim.SGD,
            "rmsprop": optim.RMSprop
        }[cfg["type"]]
        self.module.optimizer = optimizer_type(self.module.parameters(), **cfg["args"])

    def _setup_criterion(self):
        """Checks if the base model has the 'criterion_fn' property, and if True, uses this."""
        assert "criterion" in self.train_cfg, "Train cfg has no criterion"
        cfg = self.train_cfg["criterion"]

        logger.debug(f"Setting the criterion to: '{cfg['type']}' based on provided cfg.")
        criterion_type = {
            "mse": F.mse_loss,
            "l1": F.l1_loss,
            "cross_entropy": F.cross_entropy
        }[cfg["type"]]
        self.module.criterion_fn = criterion_type

    def _setup_scheduler(self):
        """Setup the scheduler following Pytorch Lightning's requirements."""
        if "scheduler" not in self.train_cfg:
            logger.debug("Scheduler not defined in train_cfg. Returning early")
            return

        cfg = self.train_cfg["scheduler"]
        assert "type" in cfg and "optimizer_args" in cfg and "monitor" in cfg["optimizer_args"], cfg
        assert self.module.optimizer is not None, "Cannot setup scheduler before optimizer."
    
        scheduler_type = {
            "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
            "ReduceLROnPlateauWithBurnIn": ReduceLROnPlateauWithBurnIn,
        }[cfg["type"]]
        scheduler = scheduler_type(optimizer=self.module.optimizer, **cfg["args"])
        self.module.scheduler_dict = {"scheduler": scheduler, **cfg["optimizer_args"]}

    def _setup_metrics(self):
        """Setup the metrics from the config file. Only a couple of them are available."""
        if "metrics" not in self.train_cfg:
            logger.debug("Metrics not defined in train_cfg. Returning early")
            return

        metrics = {}
        cfg = self.train_cfg["metrics"]
        for metric_dict in cfg:
            metric_type, metric_args = metric_dict["type"], metric_dict.get("args", {})
            assert metric_type in ("accuracy", "l1", "mse"), metric_type
            if metric_type == "accuracy":
                assert "num_classes" in metric_args and "task" in metric_args
                metrics[metric_type] = (partial(accuracy, **metric_args), "max")
            if metric_type == "l1":
                assert metric_args == {}
                metrics[metric_type] = (mean_absolute_error, "min")
            if metric_type == "mse":
                assert metric_args == {}
                metrics[metric_type] = (mean_squared_error, "min")
        self.module.metrics = metrics

    def _setup_callbacks(self):
        """TODO: callbacks"""
