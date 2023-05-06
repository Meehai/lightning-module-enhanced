"""Experiment base class"""
from __future__ import annotations
from typing import Union, List, Dict
from copy import deepcopy
from abc import ABC
from pathlib import Path
import torch as tr
import pandas as pd
import numpy as np
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from ..logger import logger as lme_logger

class Experiment(ABC):
    """Experiment class implementation"""
    def __init__(self, trainer: Union[Trainer, Experiment]):
        self.done = False

        # Stuff that is pinned to the experiment: model, trainer, train and validation set/dataloader
        self._trainer: Trainer = trainer
        self._model: LightningModule = None
        self._train_dataloaders: List[DataLoader] = None
        self._val_dataloaders: List[DataLoader] = None
        self._train_dataset: Dataset = None
        self._dataloader_params: Dict = None
        self._fit_params: Dict = None
        self._res_path: Path = None

        # post fit artefacts
        self.fit_metrics: Dict[str, Dict[str, tr.Tensor]] = {}
        self.checkpoint_callbacks: Dict[str, ModelCheckpoint] = {}
        self._df_fit_metrics: pd.DataFrame = None
        self.best_id: str = None

    @property
    def trainer(self):
        """The trainer of this eperiment. Might be overwritten during the experiment, but will stay as the original one
        before/after .fit() is called
        """
        res = self._trainer
        assert res is not None
        return res

    @trainer.setter
    def trainer(self, trainer: Union[Trainer, Experiment]):
        self._trainer = trainer

    @property
    def logger(self):
        """The current experiment's logger"""
        return self.trainer.logger

    @logger.setter
    def logger(self, logger):
        self.trainer.logger = logger

    @property
    def log_dir(self):
        """Current trainer's log dir. This updates during each experiment"""
        return self.trainer.log_dir

    @property
    def checkpoint_callback(self):
        """The current experiment's checkpoint callback."""
        assert self.done is True
        return self.checkpoint_callbacks[self.best_id]

    @property
    def id_to_ix(self):
        """Experiment id to it's index"""
        return {v: k for k, v in self.ix_to_id.items()}

    # Experiments should probably update these
    @property
    def ix_to_id(self) -> Dict[int, str]:
        """Experiment index to unique id. Ids are only used to store/load human identifiable information"""
        return {ix: ix for ix in range(len(self))}

    @property
    def df_fit_metrics(self) -> pd.DataFrame:
        """Converts the fit metrics to a dataframe"""
        return pd.DataFrame(self.fit_metrics).transpose()

    @property
    def experiment_dir_name(self) -> str:
        """The experiment's directory name in the logger. Defaults to the str of the type"""
        return str(type(self)).split(".")[-1][0:-2]

    def on_iteration_start(self, ix: int):
        """Callback called at the start of each iteration"""

    def on_iteration_end(self, ix: int):
        """Callback called at the end of each iteration"""

    def on_experiment_start(self):
        """Callback called at the start of the experiment"""

    def on_experiment_end(self):
        """Callback called at the end of the experiment"""

    def test(self, *args, **kwargs):
        """Test wrapper to call the original trainer's test()"""
        assert self.done is True
        return self.trainer.test(*args, **kwargs)

    def _fit_setup(self, model: LightningModule, train_dataloaders: DataLoader,
                   val_dataloaders: List[DataLoader], **kwargs):
        """called whenever .fit() is called first time to pin the model and dataloaders to the experiment"""
        assert self.done is False, "Cannot fit twice"
        self._model = model
        self._train_dataloaders = train_dataloaders
        self._val_dataloaders = val_dataloaders
        self._train_dataset = train_dataloaders.dataset
        self._dataloader_params = {
            "collate_fn": train_dataloaders.collate_fn,
            "num_workers": train_dataloaders.num_workers,
            "batch_size": train_dataloaders.batch_size,
        }
        self._fit_params = kwargs
        self._res_path = Path(self.trainer.log_dir) / "results.npy"
        for cb in model.configure_callbacks():
            assert not isinstance(cb, ModelCheckpoint), "Subset experiment cannot have another ModelCheckpoint"

        if self._res_path.exists():
            results = np.load(self._res_path, allow_pickle=True).item()
            self.fit_metrics = results["fit_metrics"]
            ids = list(results['fit_metrics'].keys())
            lme_logger.info(f"Loading previously done experiments (ids: {ids}) from '{self._res_path}'")
            model_checkpoints = {k: ModelCheckpoint() for k in results["checkpoint_callbacks_state"].keys()}
            for k, v in results["checkpoint_callbacks_state"].items():
                model_checkpoints[k].load_state_dict(v)
            self.checkpoint_callbacks = model_checkpoints

    def _after_iteration(self):
        """Saves the npy and dataframe of the results"""
        np.save(self._res_path, {"fit_metrics": self.fit_metrics,
                                 "checkpoint_callbacks_state": {k: v.state_dict() \
                                    for k, v in self.checkpoint_callbacks.items()}})
        self.df_fit_metrics.to_csv(f"{self.trainer.log_dir}/results.csv")

    def _do_one_iteration(self, ix: int, model: LightningModule, dataloader: DataLoader,
                          val_dataloaders: List[DataLoader]) -> Dict[str, float]:
        """The main function of this experiment. Does all the rewriting logger logic and starts the experiment."""
        # Copy old trainer and update the current one
        old_trainer = deepcopy(self.trainer)
        iter_model = deepcopy(model)
        iter_model.reset_parameters()
        # Seed
        seed_everything(ix + len(self))
        version_prefix = f"{self.trainer.logger.version}/" if self.trainer.logger.version != "" else ""
        # add "version_" only if it is a regular version stored as number (default of Pytorch Lightning)
        try:
            _ = int(self.trainer.logger.version)
            version_prefix = f"version_{version_prefix}"
        except ValueError:
            pass
        # [{prev-experiment}_]{current-experiment}_{id}
        eid = self.ix_to_id[ix]
        version = f"{version_prefix}{self.experiment_dir_name}_{eid}"
        new_logger = TensorBoardLogger(save_dir=self.trainer.logger.save_dir,
                                       name=self.trainer.logger.name, version=version)
        self.trainer.logger = new_logger

        # Train on train
        self.trainer.fit(iter_model, dataloader, val_dataloaders, **self._fit_params)

        # Test on best ckpt and validation
        model_ckpt: ModelCheckpoint = self.trainer.checkpoint_callback
        res = self.trainer.test(iter_model, val_dataloaders, ckpt_path=model_ckpt.best_model_path)[0]
        # Save this experiment's results
        self.fit_metrics[eid] = res
        self.checkpoint_callbacks[eid] = deepcopy(model_ckpt)

        # Cleanup. Remove the model, restore old trainer and return the experiment's metrics
        del iter_model
        self.trainer = old_trainer
        return res

    def fit(self, model, train_dataloaders, val_dataloaders, **kwargs):
        """The main function, uses same args as a regular pl.Trainer"""
        assert self.done is False, "Cannot fit twice"
        self._fit_setup(model, train_dataloaders, val_dataloaders, **kwargs)
        self.on_experiment_start()

        for i in range(len(self)):
            if self.ix_to_id[i] in self.fit_metrics:
                assert self.ix_to_id[i] in self.checkpoint_callbacks
                lme_logger.debug(f"Experimnt id '{self.ix_to_id[i]}' already exists. Returning early.")
                continue
            self.on_iteration_start(i)
            self._do_one_iteration(i, self._model, self._train_dataloaders, self._val_dataloaders)
            self.on_iteration_end(i)
            self._after_iteration()
        self.on_experiment_end()

        self.done = True
        self.best_id = self.df_fit_metrics.iloc[self.df_fit_metrics["loss"].argmin()].name
