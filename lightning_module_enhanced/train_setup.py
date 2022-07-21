"""Module that implements a standard setup process of the lightning module via a train config file"""
from typing import Dict
from .logger import logger
from .callbacks import MetadataCallback

from torch import optim

class TrainSetup:
    def __init__(self, module: "LightningModuleEnhanced", train_cfg: Dict):
        self.module = module
        self.train_cfg = train_cfg
    
    def _setup_optimizer(self):
        if "optimizer" not in self.train_cfg:
            logger.debug("Optimizer not defined in train_cfg. Skipping.")
            return

        optimizer_type = {
            "adamw": optim.AdamW,
            "adam": optim.Adam,
            "sgd": optim.SGD,
            "rmsprop": optim.RMSprop
        }[self.train_cfg["optimizer"]["type"]]
        self.module.optimizer = optimizer_type(self.module.parameters(), **self.train_cfg["optimizer"]["args"])
        logger.info(f"Setting optimizer to {self.module.optimizer}")

    def _setup_scheduler(self):
        """Setup the scheduler following Pytorch Lightning's requirements."""
        if "scheduler" not in self.train_cfg:
            logger.debug("Scheduler not defined in train_cfg. Skipping.")
            return

        assert self.module.optimizer is not None, "Cannot setup scheduler before optimizer."
        scheduler_type = {
            "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau
        }[self.train_cfg["scheduler"]["type"]]
        scheduler = scheduler_type(optimizer=self.module.optimizer, **self.train_cfg["scheduler"]["args"])

        if not "optimizer_args" in self.train_cfg["scheduler"]:
            logger.debug("Scheduler set, but no optimizer args. Adding a default to track 'val_loss'")
            self.train_cfg["scheduler"]["optimizer_args"] = {"monitor": "val_loss"}
        
        logger.info(f"Setting scheduler to {scheduler}")
        self.module.scheduler_dict = {"scheduler": scheduler, **self.train_cfg["scheduler"]["optimizer_args"]}

    def setup(self):
        assert self.module.num_trainable_params > 0, "Module has no trainable params!"
        if self.train_cfg is None:
            logger.info("Train cfg is None. Returning early.")
            return
        self._setup_optimizer()
        self._setup_scheduler()
