"""Metadata Callback module"""
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch as tr

from ..logger import logger

def parsed_str_type(item: Any) -> str:
    """Given an object with a type of the format: <class 'A.B.C.D'>, parse it and return 'A.B.C.D'"""
    return str(type(item)).rsplit(".", maxsplit=1)[-1][0:-2]

def json_encode_val(value: Any) -> str:
    """Given a potentially unencodable json value (but stringable), convert to string if needed"""
    try:
        _ = json.dumps(value)
        encodable_value = value
    except TypeError:
        encodable_value = str(value)
    return encodable_value


class MetadataCallback(pl.Callback):
    """Metadata Callback for a CoreModule. Stores various information about a training."""
    def __init__(self):
        self.log_dir = None
        self.log_file_path = None
        self.metadata = None

    def log_metadata_dict(self, key_val: Dict[str, Any]):
        """Log an entire dictionary of key value, by adding each key to the current metadata"""
        for key, val in key_val.items():
            self.log_metadata(key, val)

    def log_metadata(self, key: str, value: Any):
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

    def _setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prefix: str):
        """Called to set the log dir based on the first logger for train and test modes"""

        self.metadata = {
            "epoch_metrics": {},
            "hparams_current": None,
        }

        log_dir = trainer.log_dir
        self.log_dir = Path(log_dir).absolute()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.log_dir / f"{prefix}_metadata.json"
        logger.debug(f"Metadata logger set up to '{self.log_file_path}'")

        self.log_metadata("base_model", pl_module.base_model.__class__.__name__)
        self.log_metadata("summary", str(pl_module.summary))
        # default metadata
        now = datetime.now()
        self.log_metadata_dict({
            f"{prefix}_start_timestamp": datetime.timestamp(now),
            f"{prefix}_start_date": str(now),
            f"{prefix}_hparams": pl_module.hparams
        })
        self.save()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """At the start of the .fit() loop, add the sizes of all train/validation dataloaders"""
        self._setup(trainer, pl_module, prefix="fit")

        if pl_module.trainer.train_dataloader is not None:
            self.log_metadata("train dataset size", len(pl_module.trainer.train_dataloader))
        if pl_module.trainer.val_dataloaders is not None:
            for i, dataloader in enumerate(pl_module.trainer.val_dataloaders):
                self.log_metadata(f"val dataset {i} size", len(dataloader.dataset))

        # optimizer metadata at fit start
        optimizer = pl_module.optimizer
        optimizer_sd = [o.state_dict() for o in optimizer] if isinstance(optimizer, list) else [optimizer.state_dict()]
        optimizer_lrs = [o["param_groups"][0]["lr"] for o in optimizer_sd]
        optimizer_dict = {
            "Type": parsed_str_type(optimizer),
            "Starting LR": optimizer_lrs
        }
        self.log_metadata("Optimizer", optimizer_dict)

        # scheduler metadata at fit start
        if pl_module.scheduler_dict is not None:
            scheduler = pl_module.scheduler_dict["scheduler"]
            scheduler_metadata = {
                "Type": parsed_str_type(scheduler),
                **{k: v for k, v in pl_module.scheduler_dict.items() if k != "scheduler"}
            }
            if hasattr(scheduler, "mode"):
                scheduler_metadata["mode"] = scheduler.mode
            if hasattr(scheduler, "factor"):
                scheduler_metadata["factor"] = scheduler.factor
            if hasattr(scheduler, "patience"):
                scheduler_metadata["patience"] = scheduler.patience
            self.log_metadata("Scheduler", scheduler_metadata)

        # early stopping at fit start
        early_stoppings = list(filter(lambda x: isinstance(x, EarlyStopping), trainer.callbacks))
        if len(early_stoppings) > 0:
            assert len(early_stoppings) == 1
            es_dict = {
                "monitor": early_stoppings[0].monitor,
                "mode": early_stoppings[0].mode,
                "patience": early_stoppings[0].patience
            }
            self.log_metadata("Early Stopping", es_dict)

        # model checkpoint metadata at fit start
        model_checkpoint_dict = {"monitor": trainer.checkpoint_callback.monitor,
                                 "mode": trainer.checkpoint_callback.mode}
        self.log_metadata("Model Checkpoint", model_checkpoint_dict)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """At the start of the .test() loop, add the sizes of all test dataloaders"""
        self._setup(trainer, pl_module, prefix="test")
        self.metadata["epoch_metrics"] = {}
        self.log_metadata("test_start_hparams", pl_module.hparams)
        for i, dataloader in enumerate(pl_module.trainer.test_dataloaders):
            self.log_metadata(f"test dataset {i} size", len(dataloader.dataset))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Saves the metadata as a json on the train dir"""
        # Always update the current hparams such that, for test modes, we get the loaded stats
        self.log_metadata("Best model path", trainer.checkpoint_callback.best_model_path)
        self.log_metadata("hparams_current", pl_module.hparams)
        if "train dataset size" not in self.metadata:
            self.log_metadata("train dataset size", len(trainer.train_dataloader.dataset.datasets))
            self.log_metadata("validation dataset size", len(trainer.val_dataloaders[0].dataset))
        self.save()

    # pylint: disable=unused-argument
    def _on_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, prefix: str):
        """Adds the end timestamp and saves the json on the disk for train and test modes."""
        now = datetime.now()
        start_timestamp = datetime.fromtimestamp(self.metadata[f"{prefix}_start_timestamp"])
        self.log_metadata_dict({
            f"{prefix}_end_timestamp": datetime.timestamp(now),
            f"{prefix}_end_date": str(now),
            f"{prefix}_duration": str(now - start_timestamp)
        })
        self.save()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        best_checkpoint = Path(trainer.checkpoint_callback.best_model_path)
        if not (best_checkpoint.exists() and best_checkpoint.is_file()):
            logger.warning("No best model path exists. Probably trained without validation set. Using last.")
            best_checkpoint = Path(trainer.checkpoint_callback.last_model_path)

        # Store the best model dict key to have metadata about that particular checkpoint as well
        assert best_checkpoint.exists() and best_checkpoint.is_file(), "Best checkpoint does not exist."
        best_model_pkl = tr.load(best_checkpoint, map_location="cpu")
        best_model_dict = {
            "Hyper parameters": best_model_pkl["hyper_parameters"],
            "Optimizers LR": [o["param_groups"][0]["lr"] for o in best_model_pkl["optimizer_states"]]
        }
        if pl_module.scheduler_dict is not None and hasattr(pl_module.scheduler_dict["scheduler"], "factor"):
            first_lr = self.metadata["Optimizer"]["Starting LR"][0]
            last_lr = best_model_dict["Optimizers LR"][0]
            factor = pl_module.scheduler_dict["scheduler"].factor
            num_reduces = 0 if first_lr == last_lr else int((last_lr / first_lr) / factor)
            best_model_dict["Scheduler num LR reduces"] = num_reduces

        self.log_metadata("Best Model", best_model_dict)

        self._on_end(trainer, pl_module, "fit")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._on_end(trainer, pl_module, "test")

    def save(self):
        """Saves the file on disk"""
        # Force epoch metrics to be at the end for viewing purposes.
        metadata = {k: v for k, v in self.metadata.items() if k != "epoch_metrics"}
        metadata["epoch_metrics"] = self.metadata["epoch_metrics"]
        with open(self.log_file_path, "w", encoding="utf8") as fp:
            json.dump(metadata, fp, indent=4)

    def __str__(self):
        return f"Metadata Callback. Log dir: '{self.log_dir}'"

    def state_dict(self) -> Dict[str, Any]:
        return json.dumps(self.metadata)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.metadata = json.loads(state_dict)
