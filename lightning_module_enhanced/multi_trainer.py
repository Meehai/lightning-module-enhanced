"""Experiment base class"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from multiprocessing import cpu_count
from typing import NamedTuple, Optional
import os
import shutil
import csv
import numpy as np
import torch as tr
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.loggers import Logger
from pytorch_lightning.accelerators import CPUAccelerator, CUDAAccelerator
from torch.utils.data import DataLoader
from pool_resources import PoolResources, TorchResource

from .logger import lme_logger as logger

PLTrainerArgs = NamedTuple("PLTrainerArgs", model=LightningModule, train_dataloaders=Optional[DataLoader],
                           val_dataloaders=Optional[DataLoader], datamodule=Optional[LightningDataModule],
                           ckpt_path=Optional[Path])

class MultiTrainer:
    """MultiTrainer class implementation. Extends Trainer to train >1 identical networks w/ diff seeds seamlessly"""
    def __init__(self, trainer: Trainer, num_trains: int, relevant_metric: str = "loss", n_devices: int = -1):
        assert isinstance(trainer, Trainer), f"Expected pl.Trainer, got {type(trainer)}"
        if len(trainer.device_ids) > 1:
            logger.debug(f"Trainer found to have more than one device ID: {trainer.device_ids}. This is not going"
                         " to be used. We use one device per each train in a MultiTrainer")

        self.trainer: Trainer = trainer
        self.num_trains = num_trains
        self.relevant_metric = relevant_metric
        self.n_devices = n_devices
        self.resources = self._get_parallel_devices()
        self.done = False

        self.is_cuda_accelerator = isinstance(self.trainer.accelerator, CUDAAccelerator)
        self.pool_map = map
        if len(self.resources) > 0:
            self.pool_map = PoolResources(self.resources, timeout=1, pbar=False, n_raises_allowed=5).map

    # Properties

    @property
    def logger(self):
        """The current experiment's logger"""
        return self.trainer.logger

    @logger.setter
    def logger(self, pl_logger: Logger):
        self.trainer.logger = pl_logger

    @property
    def log_dir(self):
        """Current trainer's log dir. This updates during each experiment"""
        return self.trainer.log_dir

    @property
    def fit_metrics(self) -> list[dict[str, float]]:
        """Converts the fit metrics to a dataframe"""
        loaded = []
        for i in range(self.num_trains):
            results_file = Path(f"{self.logger.log_dir}/MultiTrainer/{i}/results.npy")
            if results_file.exists():
                loaded.append(np.load(results_file, allow_pickle=True).item())
        assert all(loaded[0].keys() == r.keys() for r in loaded), f"Not all keys are identical: {loaded}"
        return loaded

    @property
    def done_so_far(self) -> int:
        """return the number of experiments done so far"""
        return len(self.fit_metrics)

    @property
    def metrics(self) -> list[str]:
        """The list of metrics that were logged"""
        return self.fit_metrics[0].keys()

    @property
    def best_id(self) -> int:
        """The best experiment id. Only valid after the experiment is done"""
        assert self.done is True, "Cannot get best_id before the experiment is done"
        relevant_metric_scores = [x[self.relevant_metric] for x in self.fit_metrics]
        return np.argmin(relevant_metric_scores)

    # Public methods

    def test(self, *args, **kwargs):
        """Test wrapper to call the original trainer's test()"""
        assert self.done is True
        return self.trainer.test(*args, **kwargs)

    def fit(self, model: LightningModule, train_dataloaders: DataLoader,
            val_dataloaders: list[DataLoader] | None = None, datamodule: LightningDataModule | None = None,
            ckpt_path: Path | None = None):
        """The main function, uses same args as a regular pl.Trainer"""
        assert self.done is False, "Cannot fit twice"

        train_fit_params = []
        for i in range(self.num_trains):
            if Path(f"{self.logger.log_dir}/MultiTrainer/{i}/results.npy").exists():
                logger.debug(f"MultiTrain id '{i}' already exists. Returning early.")
                continue
            train_fit_params.append((i, {"model": deepcopy(model), "train_dataloaders": train_dataloaders,
                                         "val_dataloaders": val_dataloaders, "datamodule": datamodule,
                                         "ckpt_path": ckpt_path}))

        _ = list(self.pool_map(self._do_one_iteration, train_fit_params))
        self._post_fit()
        self.done = True

    # Private methods

    def _get_parallel_devices(self) -> list[TorchResource]:
        if self.n_devices == 0:
            return []
        assert isinstance(self.trainer.accelerator, (CPUAccelerator, CUDAAccelerator)), self.trainer.accelerator

        if self.n_devices == -1:
            n_devices = cpu_count() if isinstance(self.trainer.accelerator, CPUAccelerator) else tr.cuda.device_count()
            self.n_devices = min(n_devices, self.num_trains)
            logger.debug(f"n devices set to -1. Using all resources: {self.n_devices}")

        logger.debug(f"Accelerator: '{'cpu' if isinstance(self.trainer.accelerator, CPUAccelerator) else 'gpu'}'")
        if isinstance(self.trainer.accelerator, CPUAccelerator):
            assert cpu_count() >= self.n_devices, f"Expected {self.n_devices}, got {cpu_count()}"
            return [TorchResource(f"cpu:{ix + 1}") for ix in range(self.n_devices)] # Cpu cannot start with 0 ffs.
        assert tr.cuda.device_count() >= self.n_devices, f"Expected {self.n_devices}, got {tr.cuda.device_count()}"
        return [TorchResource(f"cuda:{ix}") for ix in range(self.n_devices)]

    def _post_fit(self):
        """called after all experiments have finished. symlink the best experiment's files to the root of the logger"""
        relevant_metric_scores = [x[self.relevant_metric] for x in self.fit_metrics]
        best_id = np.argmin(relevant_metric_scores)
        best_experiment_path = Path(f"{self.logger.log_dir}/MultiTrainer/{best_id}")
        assert best_experiment_path.exists() and len(list(best_experiment_path.iterdir())) > 0, best_experiment_path
        # symlink the best experiment to the root of the logger
        for file in best_experiment_path.iterdir():
            if file.name == "results.npy":
                continue
            out_path = Path(self.logger.log_dir) / file.name
            if out_path.exists() or out_path.is_symlink():
                logger.debug(f"'{out_path}' exists. Removing it first.")
                if out_path.is_dir() and not out_path.is_symlink():
                    shutil.rmtree(out_path)
                else:
                    out_path.unlink()
            os.symlink(file.relative_to(out_path.parent), out_path)
        writer = csv.DictWriter(open(f"{self.logger.log_dir}/MultiTrainer/fit_metrics.csv", "w"), self.metrics)
        writer.writeheader()
        writer.writerows(self.fit_metrics)

    def _do_one_iteration(self, params: tuple[int, PLTrainerArgs]):
        """The main function of this experiment. Does all the rewriting logger logic and starts the experiment."""
        ix, trainer_args = params

        # Iter model setup
        # Seed the model with the index of the experiment
        seed_everything(ix + self.num_trains)
        if hasattr(trainer_args["model"], "reset_parameters"):
            trainer_args["model"].reset_parameters()

        # Iter trainer setup
        # update the version based on the logger, experiment dir name and index. We are reusing log_dir which
        # consistes of `save_dir/name/version` of the original logger. We are adding MultiTrainer (as dir name) and
        # the index of the experiment to the version resulting in `save_dir/name/version/MultiTrainer/ix`
        # PS: do not put version=ix (as int). Lightning will add a 'version_' prefix to it and it will be a mess.
        iter_logger = type(self.trainer.logger)(save_dir=self.trainer.logger.log_dir,
                                                name="MultiTrainer", version=f"{ix}")
        # 1 device per training only. Either 1 GPU or 1 CPU.
        devices = [self.resources[ix % len(self.resources)].device.index] if self.is_cuda_accelerator else 1
        iter_trainer = Trainer(logger=iter_logger, accelerator=self.trainer.accelerator,
                               devices=devices, max_epochs=self.trainer.max_epochs, enable_model_summary=False)

        # Train on train
        iter_trainer.fit(**trainer_args)

        # Test on best ckpt and validation (or train if no validation set is provided)
        model_ckpt = iter_trainer.checkpoint_callback
        assert model_ckpt is not None
        test_loader = trainer_args["val_dataloaders"]
        if test_loader is None:
            logger.warning("No validation set was provided. Testing on train set, this can lead to bad results!")
            test_loader = trainer_args["train_dataloaders"]
        res = iter_trainer.test(trainer_args["model"], test_loader, ckpt_path=model_ckpt.best_model_path)[0]
        # Save this experiment's results as 'iteration_results.npy'
        np.save(f"{iter_logger.log_dir}/results.npy", res)

        # Cleanup. Remove the model, restore old trainer and return the experiment's metrics
        del trainer_args

    def __len__(self):
        return self.num_trains
