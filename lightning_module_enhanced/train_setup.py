"""Module that implements a standard setup process of the lightning module via a train config file"""
from typing import Dict
from torch import optim
from .schedulers import ReduceLROnPlateauWithBurnIn

from .logger import logger

class TrainSetup:
    """Train Setup class"""
    def __init__(self, module: "LightningModuleEnhanced", train_cfg: Dict):
        assert isinstance(train_cfg, Dict), f"Got {type(train_cfg)}"
        self.module = module
        self.train_cfg = train_cfg
        if not hasattr(self.module, "optimizer"):
            self.module.optimizer = None
        if not hasattr(self.module, "scheduler_dict"):
            self.module.scheduler_dict = None
        if not hasattr(self.module, "criterion_fn"):
            self.module.criterion_fn = None
        if not hasattr(self.module, "metrics"):
            self.module.metrics = None
        if not hasattr(self.module, "callbacks"):
            self.module.calbacks = None

    def _setup_optimizer(self):
        if self.module.optimizer is not None:
            logger.debug2("Optimizer is already set. Returning early.")
            return

        if "optimizer" in self.train_cfg:
            str_optimizer_type = self.train_cfg["optimizer"]["type"]
            logger.debug2("Optimizer defined in train_cfg.")
            optimizer_type = {
                "adamw": optim.AdamW,
                "adam": optim.Adam,
                "sgd": optim.SGD,
                "rmsprop": optim.RMSprop
            }[str_optimizer_type]
            # assert self.module.num_trainable_params > 0, "Module has no trainable params!"
            self.module.optimizer = optimizer_type(self.module.parameters(), **self.train_cfg["optimizer"]["args"])
            return

        # Last hope is adding optimizer from base model
        if hasattr(self.module.base_model, "optimizer") and self.module.base_model is not None:
            logger.debug2("Optimizer set from base model")
            self.module.optimizer = self.module.base_model.optimizer

    def _setup_scheduler(self):
        """Setup the scheduler following Pytorch Lightning's requirements."""
        if self.module.scheduler_dict is not None:
            logger.debug2("Scheduler was already set. Returning early")
            return

        if "scheduler" in self.train_cfg:
            logger.debug2("Scheduler defined in train_cfg.")

            assert self.module.optimizer is not None, "Cannot setup scheduler before optimizer."
            scheduler_type = {
                "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
                "ReduceLROnPlateauWithBurnIn": ReduceLROnPlateauWithBurnIn
            }[self.train_cfg["scheduler"]["type"]]
            scheduler = scheduler_type(optimizer=self.module.optimizer, **self.train_cfg["scheduler"]["args"])

            if not "optimizer_args" in self.train_cfg["scheduler"]:
                logger.debug2("Scheduler set, but no optimizer args. Adding a default to track 'val_loss'")
                self.train_cfg["scheduler"]["optimizer_args"] = {"monitor": "val_loss"}

            self.module.scheduler_dict = {"scheduler": scheduler, **self.train_cfg["scheduler"]["optimizer_args"]}
            return

        # Last hope is adding scheduler from base model
        if hasattr(self.module.base_model, "scheduler_dict") and self.module.base_model.scheduler_dict is not None:
            logger.debug2("Scheduler set from base model")
            self.module.scheduler_dict = self.module.base_model.scheduler_dict

    def _setup_criterion(self):
        """Checks if the base model has the 'criterion_fn' property, and if True, uses this."""
        if self.module.criterion_fn is not None:
            logger.debug2("Criterion fn was already set. Returning early.")
            return

        if hasattr(self.module.base_model, "criterion_fn") and self.module.base_model.criterion_fn is not None:
            logger.debug("Base model has criterion_fn attribute. Using these by default")
            self.module.criterion_fn = self.module.base_model.criterion_fn

    def _setup_metrics(self):
        """Checks if the base model has the 'metrics' property, and if True, uses it."""
        if len(self.module.metrics) > 1:
            logger.debug2("Metrics were already set. Returning early.")
            return

        if hasattr(self.module.base_model, "metrics") and self.module.base_model.metrics is not None:
            logger.debug("Base model has metrics attribute. Using these by default")
            self.module.metrics = self.module.base_model.metrics

    def _setup_callbacks(self):
        """Checks if the base model has the 'callbacks' property, and if True, uses it."""
        if len(self.module.callbacks) > len(self.module._default_callbacks): # pylint: disable=protected-access
            logger.debug2("Callbacks were already set. Returning early.")
            return

        if hasattr(self.module.base_model, "callbacks") and self.module.base_model.callbacks is not None:
            logger.debug("Base model has callbacks attribute. Using these by default")
            self.module.callbacks = self.module.base_model.callbacks

    def setup(self):
        """The main function of this class"""
        if hasattr(self.module.base_model, "setup_model_for_train"):
            logger.debug(f"Model {self.module.base_model} has setup_model_for_train() method. Calling it first.")
            self.module.base_model.setup_model_for_train(self.train_cfg)
        if self.train_cfg is None:
            logger.debug("Train cfg is None. Returning early.")
            return
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_criterion()
        self._setup_metrics()
        self._setup_callbacks()

    def __call__(self):
        return self.setup()
