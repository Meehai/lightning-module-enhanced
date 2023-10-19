"""Metadata Callback module"""
from typing import Dict, Any, List
from pathlib import Path
from io import FileIO
from datetime import datetime
import json
import pytorch_lightning as pl
import torch as tr
from overrides import overrides
from torch.optim import Optimizer
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from ..logger import logger
from ..utils import parsed_str_type


class MetadataCallback(pl.Callback):
    """Metadata Callback for LME. Stores various information about a training."""
    def __init__(self):
        self.log_dir = None
        self.log_file_path = None
        self.metadata = None

    def log_epoch_metric(self, key: str, value: tr.Tensor, epoch: int, prefix: str):
        """Adds a epoch metric to the current metadata. Called from LME"""
        # test and train get the unprefixed key.
        prefixed_key = f"{prefix}{key}"
        if prefixed_key not in self.metadata["epoch_metrics"]:
            self.metadata["epoch_metrics"][prefixed_key] = {}
        if epoch in self.metadata["epoch_metrics"][prefixed_key]:
            raise ValueError(f"Metric '{prefixed_key}' at epoch {epoch} already exists in metadata")
        self.metadata["epoch_metrics"][prefixed_key][epoch] = value.tolist()

    @overrides
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """At the start of the .fit() loop, add the sizes of all train/validation dataloaders"""
        if pl_module.trainer.state.stage == "sanity_check":
            return
        self._setup(trainer, prefix="fit")
        self._log_optimizer_fit_start(pl_module)
        self._log_scheduler_fit_start(pl_module)
        self._log_early_stopping_fit_start(pl_module)
        self._log_model_checkpoint_fit_start(pl_module)
        self._log_model_summary(pl_module)
        self.metadata["fit_params"] = pl_module.hparams
        self._log_timestamp_start(prefix="fit")

    @overrides
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_best_model_epoch_end(pl_module)
        self._log_scheduler_train_end(pl_module)
        self._log_timestamp_end("fit")
        self.save()

    @overrides
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.trainer.state.stage == "sanity_check":
            return
        if "train_dataset_size" not in self.metadata:
            self.metadata["train_dataset_size"] = len(trainer.train_dataloader.dataset)
        if "validation_dataset_size" not in self.metadata and trainer.val_dataloaders is not None:
            self.metadata["validation_dataset_size"] = len(pl_module.trainer.val_dataloaders.dataset)

    @overrides
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Saves the metadata as a json on the train dir"""
        # Always update the current hparams such that, for test modes, we get the loaded stats
        self.metadata["hparams_current"] = pl_module.hparams
        self._log_timestamp_epoch_end()
        self.save()

    @overrides
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """At the start of the .test() loop, add the sizes of all test dataloaders"""
        self._setup(trainer, prefix="test")
        self.metadata["test_start_hparams"] = pl_module.hparams
        self.metadata["test_dataset_size"] = len(pl_module.trainer.test_dataloaders.dataset)
        self._log_model_summary(pl_module)
        self.metadata["test_hparams"] = pl_module.hparams
        self._log_timestamp_start(prefix="test")

    @overrides
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_timestamp_end("test")
        self.save()

    @overrides
    def state_dict(self) -> Dict[str, Any]:
        return json.dumps(self.metadata)

    @overrides
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.metadata = json.loads(state_dict)

    def save(self):
        """Saves the file on disk"""
        # Force epoch metrics to be at the end for viewing purposes.
        if self.log_file_path is None:
            return
        metadata = {k: v for k, v in self.metadata.items() if k != "epoch_metrics"}
        metadata["epoch_metrics"] = self.metadata["epoch_metrics"]
        with open(self.log_file_path, "w", encoding="utf8") as fp:
            try:
                json.dump(metadata, fp, indent=4)
            except TypeError as ex:
                self._debug_metadata_json_dump(metadata, fp)
                raise TypeError(ex)

    # private methods

    def _setup(self, trainer: pl.Trainer, prefix: str):
        """Called to set the log dir based on the first logger for train and test modes"""
        assert prefix in ("fit", "test"), prefix
        # flushing the metrics can happen in 3 cases:
        # 1) metadata is None, so we just initialie it
        # 2) metadata is not None, we are training the same model with a new trainer, so it starts again from epoch 0.
        #    Note, in this case we also need to check for ckpt_path, becaus at this point current_epoch is 0, but we
        #    may be resuning.
        # 3) metadata is not None, we are testing the model, so we don't want to have a test metadata with train metrics
        if self.metadata is None or \
           (self.metadata is not None and prefix == "fit" and trainer.current_epoch == 0
                and trainer.ckpt_path is None) or \
           (self.metadata is not None and prefix == "test"):
            self.metadata = {
                "epoch_metrics": {},
                "hparams_current": None,
            }

        # using trainer.logger.log_dir will have errors for non TensorBoardLogger (at least in lightning 1.8)
        log_dir = None
        if len(trainer.loggers) > 0:
            log_dir = trainer.loggers[0].log_dir
            self.log_dir = Path(log_dir).absolute()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file_path = self.log_dir / f"{prefix}_metadata.json"
            logger.debug(f"Metadata logger set up to '{self.log_file_path}'")
        else:
            logger.warning("No logger provided to Trainer. Metadata will not be stored on disk!")

        self.save()

    def _log_optimizer_fit_start(self, pl_module: LightningModule, configure_optimizers_result=None):
        """optimizer metadata at fit start"""
        if configure_optimizers_result is None:
            self.metadata["optimizer"] = self._log_optimizer_fit_start(pl_module, pl_module.configure_optimizers())
            return None
        if isinstance(configure_optimizers_result, list):
            return [self._log_optimizer_fit_start(pl_module, o) for o in configure_optimizers_result]
        if isinstance(configure_optimizers_result, dict):
            assert "optimizer" in configure_optimizers_result
            return self._log_optimizer_fit_start(pl_module, configure_optimizers_result["optimizer"])
        if isinstance(configure_optimizers_result, Optimizer):
            optim_sd = configure_optimizers_result.state_dict()
            return {
                "type": parsed_str_type(configure_optimizers_result),
                "starting_lr": [o["lr"] for o in optim_sd["param_groups"]]
            }
        raise ValueError(f"configure optimizers result type not yet supported: {configure_optimizers_result}")

    def _log_scheduler_fit_start(self, pl_module: LightningModule):
        """logs information about the scheduler, if it exists"""
        if not hasattr(pl_module, "scheduler_dict"):
            return
        if pl_module.scheduler_dict is None:
            return
        scheduler = pl_module.scheduler_dict["scheduler"]
        scheduler_metadata = {
            "type": parsed_str_type(scheduler),
            **{k: v for k, v in pl_module.scheduler_dict.items() if k != "scheduler"}
        }
        if hasattr(scheduler, "mode"):
            scheduler_metadata["mode"] = scheduler.mode
        if hasattr(scheduler, "factor"):
            scheduler_metadata["factor"] = scheduler.factor
        if hasattr(scheduler, "patience"):
            scheduler_metadata["patience"] = scheduler.patience
        self.metadata["scheduler"] = scheduler_metadata

    def _log_scheduler_train_end(self, pl_module: LightningModule):
        """updates bset model dict with the number of learning rate reduces done by the scheduler during training"""
        if not hasattr(pl_module, "scheduler_dict"):
            return
        if pl_module.scheduler_dict is None:
            return
        if not hasattr(pl_module.scheduler_dict["scheduler"], "factor"):
            return
        best_model_dict = self.metadata["best_model"]
        first_lr = self.metadata["optimizer"][0]["starting_lr"][0]
        last_lr = best_model_dict["optimizers_lr"][0]
        factor = pl_module.scheduler_dict["scheduler"].factor
        num_reduces = 0 if first_lr == last_lr else int((last_lr / first_lr) / factor)
        best_model_dict["scheduler_num_lr_reduced"] = num_reduces

    def _log_early_stopping_fit_start(self, pl_module: LightningModule):
        assert pl_module.trainer is not None, "Invalid call to this function, trainer is not set."
        early_stopping_cbs = list(filter(lambda x: isinstance(x, EarlyStopping), pl_module.trainer.callbacks))
        # no early stopping for this train, simply return
        if len(early_stopping_cbs) == 0:
            return
        assert len(early_stopping_cbs) == 1, early_stopping_cbs
        early_stopping_cb: EarlyStopping = early_stopping_cbs[0]
        es_dict = {
            "monitor": early_stopping_cb.monitor,
            "mode": early_stopping_cb.mode,
            "patience": early_stopping_cb.patience
        }
        self.metadata["Early Stopping"] = es_dict

    def _log_model_summary(self, pl_module: LightningModule):
        """model's layers and number of parameters"""
        assert hasattr(pl_module, "base_model")
        res = {"name": pl_module.base_model.__class__.__name__}
        layer_summary = {}
        num_params, num_trainable_params = 0, 0
        for name, param in pl_module.base_model.named_parameters():
            num_params += param.numel()
            num_trainable_params += param.numel() * param.requires_grad
            layer_summary[name] = f"count: {param.numel()}. requires_grad: {param.requires_grad}"
        res["parameter_count"] = {"total": num_params, "trainable": num_trainable_params}
        res["layer_summary"] = layer_summary
        self.metadata["model_parameters"] = res

    def _get_monitored_model_checkpoint(self, pl_module: LightningModule) -> ModelCheckpoint:
        monitors: List[str] = pl_module.checkpoint_monitors
        assert len(monitors) > 0, "At least one monitor must be present."
        if len(monitors) > 1:
            logger.warning(f"More than one monitor provided: {monitors}. Keeping only first")
        monitor = monitors[0]
        prefix = "val_" if pl_module.trainer.enable_validation else ""
        cb = [cb for cb in pl_module.trainer.checkpoint_callbacks if cb.monitor == f"{prefix}{monitor}"]
        if len(cb) > 1:
            logger.warning(f"More than one callback for monitor '{monitor}' found: {cb}")
        assert len(cb) > 0, f"Monitor '{monitor}' not found in model checkpoints: {monitors} (prefix: {prefix})"
        return cb[0]

    def _log_model_checkpoint_fit_start(self, pl_module: LightningModule):
        cb = self._get_monitored_model_checkpoint(pl_module)
        # model checkpoint metadata at fit start
        model_checkpoint_dict = {"monitors": cb.monitor, "mode": cb.mode}
        self.metadata["model_checkpoint"] = model_checkpoint_dict

    def _log_best_model_epoch_end(self, pl_module: LightningModule):
        # find best and last modelcheckpoint callbacks. best will be None if we don't use a validation set loader.
        cb = self._get_monitored_model_checkpoint(pl_module)
        ckpt_path = Path(cb.best_model_path)
        assert ckpt_path.exists() and ckpt_path.is_file(), "Best checkpoint does not exist."

        best_model_pkl = tr.load(ckpt_path, map_location="cpu")
        best_model_dict = {
            "path": str(ckpt_path),
            "hyper_parameters": best_model_pkl.get("hyper_parameters", {}),
            "optimizers_lr": [o["param_groups"][0]["lr"] for o in best_model_pkl["optimizer_states"]],
            "epoch": best_model_pkl["epoch"],
        }

        self.metadata["best_model"] = best_model_dict

    def _log_timestamp_start(self, prefix: str):
        """Logs the timestamp of fit_start or test_start"""
        now = datetime.now()
        self.metadata[f"{prefix}_start_timestamp"] = datetime.timestamp(now)
        self.metadata[f"{prefix}_start_date"] = str(now)
        if prefix == "fit":
            self.metadata["epoch_timestamps"] = []

    def _log_timestamp_end(self, prefix: str):
        """Adds the end timestamp and saves the json on the disk for train and test modes."""
        now = datetime.now()
        start_timestamp = datetime.fromtimestamp(self.metadata[f"{prefix}_start_timestamp"])
        self.metadata[f"{prefix}_end_timestamp"] = datetime.timestamp(now)
        self.metadata[f"{prefix}_end_date"] = str(now)
        self.metadata[f"{prefix}_duration"] = str(now - start_timestamp)

    def _log_timestamp_epoch_end(self):
        """compute the average durations from fit_start to now (might be wrong for retrainings though)"""
        self.metadata["epoch_timestamps"].append(datetime.timestamp(datetime.now()))
        start = self.metadata["fit_start_timestamp"]
        timestamps = tr.DoubleTensor([start, *self.metadata["epoch_timestamps"]], device="cpu")
        self.metadata["epoch_average_duration"] = (timestamps[1:] - timestamps[0:-1]).mean().item()

    def _debug_metadata_json_dump(self, metadata: Dict[str, Any], fp: FileIO) -> None:
        # for debugging purposes
        for k in metadata:
            try:
                json.dump({k: metadata[k]}, fp)
            except TypeError:
                logger.debug(f"Cannot serialize key '{k}'")

    def __str__(self):
        return f"Metadata Callback. Log dir: '{self.log_dir}'"
