"""Experiment base class"""
from __future__ import annotations
from typing import Union, List, Dict
from copy import deepcopy
from abc import ABC, abstractmethod
import pandas as pd
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

class Experiment(ABC):
    def __init__(self, trainer: Union[Trainer, Experiment]):
        self._trainer = trainer
        self.done = False

        # post fit artefacts
        self.fit_trainers = []
        self.fit_metrics = []
        self.df_fit_metrics: pd.DataFrame = None
        self.ix = None

    @property
    def trainer(self):
        res = self._trainer
        assert res is not None
        return res

    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer

    @property
    def logger(self):
        """The current experiment's logger"""
        return self.trainer.logger

    @logger.setter
    def logger(self, logger):
        self.trainer.logger = logger

    @property
    @abstractmethod
    def checkpoint_callback(self):
        """The current experiment's checkpoint callback."""

    @property
    def experiment_dir_name(self) -> str:
        """The experiment's directory name in the logger. Defaults to the str of the type"""
        return str(type(self)).split(".")[-1][0:-2]

    def test(self, *args, **kwargs):
        assert self.done is True
        return self.trainer.test(*args, **kwargs)

    def do_one_iteration(self, ix: int, model: LightnimgModule, dataloader: Dataloader,
                         val_dataloaders: List[DataLoader], *args, **kwargs) -> Dict[str, float]:
        # Seed
        seed_everything(ix)
        # Copy old trainer and update the current one
        old_trainer = deepcopy(self.trainer)
        iter_model = deepcopy(model)
        iter_model.reset_parameters()

        version_prefix = f"{self.trainer.logger.version}/" if self.trainer.logger.version != "" else ""
        # {prev-experiment}_{current-experiment}_{ix}_{total}
        version = f"{version_prefix}{self.experiment_dir_name}_{ix}_{len(self)}"
        new_logger = TensorBoardLogger(save_dir=self.trainer.logger.save_dir,
                                        name=self.trainer.logger.name, version=version)
        self.trainer.logger = new_logger

        # Train on train
        self.trainer.fit(iter_model, dataloader, val_dataloaders, *args, **kwargs)

        # Test on best ckpt and validation
        ckpt_path = self.trainer.checkpoint_callback.best_model_path
        res = self.trainer.test(iter_model, val_dataloaders, ckpt_path=ckpt_path)[0]
        del iter_model
        # Save this experiment's results
        self.fit_trainers.append(deepcopy(self.trainer))
        self.fit_metrics.append(res)
        # Restore old trainer and return the experiment's metrics
        self.trainer = old_trainer
        return res
