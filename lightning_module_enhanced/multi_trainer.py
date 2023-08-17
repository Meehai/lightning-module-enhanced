"""Experiment base class"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import os
import shutil
import pandas as pd
import numpy as np
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .logger import logger as lme_logger

class MultiTrainer:
    """MultiTrainer class implementation. Extends Trainer to train >1 identical networks w/ diff seeds seamlessly"""
    def __init__(self, trainer: Trainer, num_trains: int, relevant_metric: str = "loss"):
        self.done = False
        assert isinstance(trainer, Trainer), f"Expected pl.Trainer, got {type(trainer)}"
        self.num_trains = num_trains
        self.relevant_metric = relevant_metric

        # Stuff that is pinned to the experiment: model, trainer, train and validation set/dataloader
        self._trainer: Trainer = trainer

    # Properties

    @property
    def trainer(self):
        """The trainer of this eperiment. Might be overwritten during the experiment, but will stay as the original one
        before/after .fit() is called
        """
        res = self._trainer
        assert res is not None
        return res

    @trainer.setter
    def trainer(self, trainer: Trainer):
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
    def fit_metrics(self) -> pd.DataFrame:
        """Converts the fit metrics to a dataframe"""
        res = {}
        for i in range(self.num_trains):
            results_file = Path(self.logger.log_dir) / self.experiment_dir_name / f"{i}" / "results.npy"
            if results_file.exists():
                res[i] = np.load(results_file, allow_pickle=True).item()
        return pd.DataFrame(res).transpose()

    @property
    def experiment_dir_name(self) -> str:
        """The experiment's directory name in the logger. Defaults to the str of the type"""
        return str(type(self)).split(".")[-1][0:-2]

    @property
    def done_so_far(self) -> int:
        """return the number of experiments done so far"""
        return len(self.fit_metrics)

    @property
    def best_id(self) -> int:
        """The best experiment id. Only valid after the experiment is done"""
        assert self.done is True, "Cannot get best_id before the experiment is done"
        return self.fit_metrics[self.relevant_metric].argmin()

    # Public methods

    def test(self, *args, **kwargs):
        """Test wrapper to call the original trainer's test()"""
        assert self.done is True
        return self.trainer.test(*args, **kwargs)

    def fit(self, model: LightningModule, train_dataloaders: DataLoader,
            val_dataloaders: list[DataLoader] | None = None, **kwargs):
        """The main function, uses same args as a regular pl.Trainer"""
        assert self.done is False, "Cannot fit twice"
        self._fit_setup(model)

        train_fit_params = []
        for i in range(self.num_trains):
            if i in self.fit_metrics.index:
                lme_logger.debug(f"MultiTrain id '{i}' already exists. Returning early.")
                continue
            train_fit_params.append((i, model, train_dataloaders, val_dataloaders, kwargs))

        _ = list(map(self._do_one_iteration, train_fit_params))
        self._post_fit()
        self.done = True

    # Private methods

    def _fit_setup(self, model: LightningModule):
        """called whenever .fit() is called first time to pin the model and dataloaders to the experiment"""
        if hasattr(model, "configure_callbacks"):
            cbs = model.configure_callbacks()
            cbs_list = cbs if isinstance(cbs, list) else [cbs]
            for cb in cbs_list:
                assert not isinstance(cb, ModelCheckpoint), "Cannot have another ModelCheckpoint (?)"

    def _post_fit(self):
        """called after all experiments have finished. symlink the best experiment's files to the root of the logger"""
        best_id = self.fit_metrics[self.relevant_metric].argmin()
        best_experiment_path = Path(self.logger.log_dir) / self.experiment_dir_name / f"{best_id}"
        assert best_experiment_path.exists() and len(list(best_experiment_path.iterdir())) > 0, best_experiment_path
        # symlink the best experiment to the root of the logger
        for file in best_experiment_path.iterdir():
            if file.name == "results.npy":
                continue
            out_path = Path(self.logger.log_dir) / file.name
            if out_path.exists() or out_path.is_symlink():
                lme_logger.warning(f"'{out_path}' exists. Removing it first.")
                if out_path.is_dir() and not out_path.is_symlink():
                    shutil.rmtree(out_path)
                else:
                    out_path.unlink()
            os.symlink(file.relative_to(out_path.parent), out_path)
        self.fit_metrics.to_csv(Path(self.logger.log_dir) / self.experiment_dir_name / "fit_metrics.csv")

    def _do_one_iteration(self, params: tuple[int, LightningModule, DataLoader, list[DataLoader] | None, dict]):
        """The main function of this experiment. Does all the rewriting logger logic and starts the experiment."""
        ix, model, dataloader, val_dataloaders, kwargs = params
        # Copy old trainer and update the current one
        iter_trainer = deepcopy(self.trainer)
        iter_model = deepcopy(model)
        # Seed the model with the index of the experiment
        seed_everything(ix + self.num_trains)
        if hasattr(iter_model, "reset_parameters"):
            iter_model.reset_parameters()
        # update the version based on the logger, experiment dir name and index. We are reusing log_dir which
        # consistes of `save_dir/name/version` of the original logger. We are adding MultiTrainer (as dir name) and
        # the index of the experiment to the version resulting in `save_dir/name/version/MultiTrainer/ix`
        # PS: do not put version=ix (as int). Lightning will add a 'version_' prefix to it and it will be a mess.
        iter_logger = type(self.trainer.logger)(save_dir=self.trainer.logger.log_dir,
                                                name=self.experiment_dir_name, version=f"{ix}")
        iter_trainer.logger = iter_logger

        # Train on train
        iter_trainer.fit(iter_model, dataloader, val_dataloaders, **kwargs)

        # Test on best ckpt and validation (or train if no validation set is provided)
        model_ckpt: ModelCheckpoint = iter_trainer.checkpoint_callback
        test_loader = val_dataloaders if val_dataloaders is not None else dataloader
        res = iter_trainer.test(iter_model, test_loader, ckpt_path=model_ckpt.best_model_path)[0]
        # Save this experiment's results as 'iteration_results.npy'
        np.save(f"{iter_logger.log_dir}/results.npy", res)

        # Cleanup. Remove the model, restore old trainer and return the experiment's metrics
        del iter_model
        del iter_trainer
        del iter_logger

    def __len__(self):
        return self.num_trains
