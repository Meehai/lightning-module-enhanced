"""Metaata logger module"""
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
import json
from pytorch_lightning import LightningModule
import torch as tr


class MetadataLogger:
    """Metadata Logger for a CoreModule. Stores various information about a training."""
    def __init__(self, model: LightningModule):
        self.model = model
        self.log_dir = None
        self.log_file_path = None
        self.metadata = None
        self.reset()

    def reset(self):
        """Resets this metadata logger to initial state"""
        self.metadata = {
            "epoch_metrics": {},
            "hparams_current": None,
        }
        self.log_timestamp("create_time")

    def log_timestamp(self, key: str):
        """Logs the timestamp wtih a given key"""
        now = datetime.now()
        self.log(key, {
            "timestamp": datetime.timestamp(now),
            "date": str(now)
        })

    def log_dict(self, key_val: Dict[str, Any]):
        """Log an entire dictionary of key value, by adding each key to the current metadata"""
        for key, val in key_val.items():
            self.log(key, val)

    def log(self, key: str, value: Any):
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

    def save(self):
        """Saves the metadata as a json on the train dir"""
        # Always update the current hparams such that, for test modes, we get the loaded stats
        metadata = {**self.metadata, "hparams_current": self.model.hparams}
        with open(self.log_file_path, "w", encoding="utf8") as fp:
            json.dump(metadata, fp, indent=4)

    def setup(self):
        """Called to set the log dir based on the first logger"""
        self.reset()
        log_dir = self.model.loggers[0].log_dir
        self.log_dir = Path(log_dir).absolute()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.log_dir / "metadata.json"

    def on_fit_start(self, model: LightningModule):
        """At the start of the .fit() loop, add the sizes of all train/validation dataloaders"""
        self.setup()
        assert self.model.trainer is not None
        self.log("fit_start_hparams", model.hparams)

        if model.trainer.train_dataloader is not None:
            self.log("train dataset size", len(model.trainer.train_dataloader))
        if model.trainer.val_dataloaders is not None:
            for i, dataloader in enumerate(model.trainer.val_dataloaders):
                self.log(f"val dataset {i} size", len(dataloader.dataset))

    def on_test_start(self, model: LightningModule):
        """At the start of the .test() loop, add the sizes of all test dataloaders"""
        self.setup()
        assert self.model.trainer is not None
        self.metadata["epoch_metrics"] = {}
        self.log("test_start_hparams", model.hparams)
        for i, dataloader in enumerate(model.trainer.test_dataloaders):
            self.log(f"test dataset {i} size", len(dataloader.dataset))

    def __str__(self):
        return f"Metadata Logger. Log dir: '{self.log_dir}'"
