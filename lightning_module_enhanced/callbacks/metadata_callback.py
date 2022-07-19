"""Metadata Callback module"""
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
import json
import pytorch_lightning as pl
import torch as tr

from ..logger import logger


class MetadataCallback(pl.Callback):
    """Metadata Callback for a CoreModule. Stores various information about a training."""
    def __init__(self):
        self.log_dir = None
        self.log_file_path = None
        self.metadata = {
            "epoch_metrics": {},
            "hparams_current": None,
        }

    def _log_dict(self, key_val: Dict[str, Any]):
        """Log an entire dictionary of key value, by adding each key to the current metadata"""
        for key, val in key_val.items():
            self._log(key, val)

    def _log(self, key: str, value: Any):
        """Adds a key->value pair to the current metadata"""
        self.metadata[key] = value

    def save_epoch_metric(self, key: str, value: tr.Tensor, epoch: int):
        """Adds a epoch metric to the current metadata"""
        if key not in self.metadata["epoch_metrics"]:
            self.metadata["epoch_metrics"][key] = {}
        if epoch != 0:
            # Epoch 0 can sometimes have a validation sanity check fake epoch
            assert epoch not in self.metadata["epoch_metrics"][key], f"Cannot overwrite existing epoch metric '{key}'"
        # Apply .tolist(). Every metric should be able to be converted as list, such that it can be stored in a JSON.
        self.metadata["epoch_metrics"][key][epoch] = value.tolist()

    def _setup(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, start_type: str):
        """Called to set the log dir based on the first logger"""
        assert self.log_dir is None
        log_dir = trainer.log_dir
        self.log_dir = Path(log_dir).absolute()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.log_dir / "metadata.json"
        logger.debug(f"Metadata logger set up to '{self.log_file_path}'")

        self._log_dict({"input_shape": pl_module.base_model.input_shape,
                        "output_shape": pl_module.base_model.output_shape,
                        "base_model": pl_module.base_model.__class__.__name__
                        })
        # default metadata
        now = datetime.now()
        self._log_dict({
            f"{start_type}_start_timestamp": datetime.timestamp(now),
            f"{start_type}_start_date": str(now),
            f"{start_type}_hparams": pl_module.hparams
        })

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        """At the start of the .fit() loop, add the sizes of all train/validation dataloaders"""
        self._setup(trainer, pl_module, start_type="fit")

        if pl_module.trainer.train_dataloader is not None:
            self._log("train dataset size", len(pl_module.trainer.train_dataloader))
        if pl_module.trainer.val_dataloaders is not None:
            for i, dataloader in enumerate(pl_module.trainer.val_dataloaders):
                self._log(f"val dataset {i} size", len(dataloader.dataset))

    def on_test_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        """At the start of the .test() loop, add the sizes of all test dataloaders"""
        self._setup(trainer, pl_module, start_type="test")
        self.metadata["epoch_metrics"] = {}
        self._log("test_start_hparams", pl_module.hparams)
        for i, dataloader in enumerate(pl_module.trainer.test_dataloaders):
            self._log(f"test dataset {i} size", len(dataloader.dataset))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        """Saves the metadata as a json on the train dir"""
        # Always update the current hparams such that, for test modes, we get the loaded stats
        self._log("Best model path", trainer.checkpoint_callback.best_model_path)
        metadata = {**self.metadata, "hparams_current": pl_module.hparams}
        with open(self.log_file_path, "w", encoding="utf8") as fp:
            json.dump(metadata, fp, indent=4)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        now = datetime.now()
        self._log_dict({
            "train_end_timestamp": datetime.timestamp(now),
            "train_end_date": str(now),
            "train_duration": str(now - datetime.fromtimestamp(self.metadata["fit_start_timestamp"]))
        })

    def __str__(self):
        return f"Metadata Callback. Log dir: '{self.log_dir}'"
