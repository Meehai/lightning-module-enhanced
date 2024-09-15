"""Metadata Callback module"""
from __future__ import annotations
import sys
from typing import Any, IO
from pathlib import Path
from datetime import datetime
import json
import pytorch_lightning as pl
import torch as tr
from overrides import overrides
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Checkpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..logger import lme_logger as logger
from ..utils import parsed_str_type, make_list, flat_if_one


class MetadataCallback(pl.Callback):
    """Metadata Callback for LME. Stores various information about a training."""
    def __init__(self):
        self.log_dir = None
        self.log_file_path = None
        self.metadata: dict[str, Any] = None

    def log_epoch_metric(self, key: str, value: tr.Tensor, epoch: int, prefix: str):
        """Adds a epoch metric to the current metadata. Called from LME"""
        # test and train get the unprefixed key.
        prefixed_key = f"{prefix}{key}"
        if prefixed_key not in self.metadata["epoch_metrics"]:
            self.metadata["epoch_metrics"][prefixed_key] = {}
        if epoch in self.metadata["epoch_metrics"][prefixed_key]:
            raise ValueError(f"Metric '{prefixed_key}' at epoch {epoch} already exists in metadata")
        self.metadata["epoch_metrics"][prefixed_key][epoch] = value.tolist()

    @rank_zero_only
    @overrides(check_signature=False)
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """At the start of the .fit() loop, add the sizes of all train/validation dataloaders"""
        if pl_module.trainer.state.stage == "sanity_check":
            return
        self._setup(trainer, prefix="fit")
        self.metadata["optimizer"] = self._log_optimizer_fit_start(pl_module)
        if (sch := self._log_scheduler_fit_start(pl_module)) is not None:
            self.metadata["scheduler"] = sch
        if (es := self._log_early_stopping_fit_start(pl_module)) is not None:
            self.metadata["early_stopping"] = es
        self.metadata["model_checkpoint"] = self._log_model_checkpoint_fit_start(pl_module)
        self.metadata["model_parameters"] = self._log_model_summary(pl_module)
        self.metadata["fit_hparams"] = pl_module.hparams.get("metadata_hparams")
        self.metadata = {**self.metadata, **self._log_timestamp_start(prefix="fit")}

    @rank_zero_only
    @overrides(check_signature=False)
    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.metadata["best_model"] = self._log_best_model_fit_end(pl_module)
        self.metadata = {**self.metadata, **self._log_timestamp_end("fit")}
        self.save()
        if any(isinstance(x, WandbLogger) for x in trainer.loggers):
            wandb_logger: WandbLogger = [x for x in trainer.loggers if isinstance(x, WandbLogger)][0]
            wandb_logger.experiment.log_artifact(self.log_file_path)

    @rank_zero_only
    @overrides(check_signature=False)
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if pl_module.trainer.state.stage == "sanity_check":
            return
        if "train_dataset_size" not in self.metadata:
            self.metadata["train_dataset_size"] = len(trainer.train_dataloader.dataset)
        if "validation_dataset_size" not in self.metadata and trainer.val_dataloaders is not None:
            self.metadata["validation_dataset_size"] = len(pl_module.trainer.val_dataloaders.dataset)

    @rank_zero_only
    @overrides(check_signature=False)
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Saves the metadata as a json on the train dir"""
        # Always update the current hparams such that, for test modes, we get the loaded stats
        hist = self._log_optimizer_lr_history_epoch_end(pl_module)
        assert len(hist) == len(optims := make_list(self.metadata["optimizer"])), f"{len(hist)} vs {len(optims)}"
        # TODO: make a test with >1 optimizer and test that this works
        for o_history, o_hist in zip(optims, hist):
            o_history["lr_history"] = o_hist
        self.metadata = {**self.metadata, **self._log_timestamp_train_epoch_end()}
        # TODO: call metrics_history here
        self.save()

    @rank_zero_only
    @overrides(check_signature=False)
    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """At the start of the .test() loop, add the sizes of all test dataloaders"""
        self._setup(trainer, prefix="test")
        self.metadata["test_dataset_size"] = len(pl_module.trainer.test_dataloaders.dataset) # type: ignore
        self.metadata["model_parameters"] = self._log_model_summary(pl_module)
        self.metadata["test_hparams"] = pl_module.hparams.get("metadata_hparams")
        self.metadata = {**self.metadata, **self._log_timestamp_start(prefix="test")}

    @rank_zero_only
    @overrides(check_signature=False)
    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_timestamp_end("test")
        self.save()

    @overrides(check_signature=False)
    def state_dict(self) -> dict[str, Any]:
        return json.dumps(self.metadata) # type: ignore

    @overrides(check_signature=False)
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.metadata = json.loads(state_dict) # type: ignore

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

    def _flush_metadata(self):
        self.metadata = {
            "run_command": " ".join(sys.argv),
            "epoch_metrics": {},
        }

    def _setup(self, trainer: pl.Trainer, prefix: str):
        """Called to set the log dir based on the first logger for train and test modes"""
        assert prefix in ("fit", "test"), prefix
        # flushing the metrics can happen in 3 cases:
        # 1) metadata is None, so we just initialie it
        # 2) metadata is not None, we are training the same model with a new trainer, so it starts again from epoch 0.
        #    Note, in this case we also need to check for ckpt_path, becaus at this point current_epoch is 0, but we
        #    may be resuning.
        # 3) metadata is not None, we are testing the model, so we don't want to have a test metadata with train metrics
        if self.metadata is None:
            self._flush_metadata()
        elif self.metadata is not None and prefix == "fit" and trainer.current_epoch == 0 and trainer.ckpt_path is None:
            self._flush_metadata()
        elif self.metadata is not None and prefix == "test":
            self._flush_metadata()

        # using trainer.logger.log_dir will have errors for non TensorBoardLogger (at least in lightning 1.8)
        log_dir = None
        if len(trainer.loggers) > 0:
            log_dir = trainer.loggers[0].log_dir
            self.log_dir = Path(log_dir).absolute()
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file_path = self.log_dir / f"{prefix}_metadata.json"
            logger.debug(f"Metadata logger set up to '{self.log_file_path}'")
        else:
            logger.debug("No logger provided to Trainer. Metadata will not be stored on disk!")

        self.save()

    def _get_optimizer_current_lr(self, optimizer: tr.optim.Optimizer | dict) -> float:
        assert isinstance(optimizer, (tr.optim.Optimizer, dict)), f"Must be optimizer or state_dict: {type(optimizer)}"
        sd = optimizer.state_dict() if isinstance(optimizer, tr.optim.Optimizer) else optimizer
        res = [o["lr"] for o in sd["param_groups"]]
        assert all(x == res[0] for x in res), f"Not supporting differnt lrs at param groups in same optim: {res}"
        return res[0]

    def _log_model_summary(self, pl_module: "LME") -> dict:
        """model's layers and number of parameters"""
        assert hasattr(pl_module, "base_model")
        base_model: pl.LightningModule = pl_module.base_model
        res = {"name": base_model.__class__.__name__}
        layer_summary = {}
        num_params, num_trainable_params = 0, 0
        for name, param in base_model.named_parameters():
            num_params += param.numel()
            num_trainable_params += param.numel() * param.requires_grad
            layer_summary[name] = f"count: {param.numel()}. requires_grad: {param.requires_grad}"
        res["parameter_count"] = {"total": num_params, "trainable": num_trainable_params}
        res["layer_summary"] = layer_summary
        return res

    def _get_monitored_model_checkpoint(self, pl_module: "LME") -> ModelCheckpoint | None:
        """returns the (first) model checkpoint, as provided by model.checkpoint_monitors. Usually the 'loss' monitor"""
        monitors: list[str] = pl_module.checkpoint_monitors # type: ignore
        trainer: pl.Trainer = pl_module.trainer
        if len(monitors) == 0:
            logger.debug("No monitors were found. Best checkpoint metadata will not be stored")
            return None
        if len(monitors) > 1:
            logger.debug(f"More than one monitor provided: {monitors}. Keeping only first")
        monitor = monitors[0]
        prefix = "val_" if trainer.enable_validation else ""
        callbacks: list[Checkpoint] = trainer.checkpoint_callbacks
        cbs: list[ModelCheckpoint] = [_cb for _cb in callbacks if isinstance(_cb, ModelCheckpoint)]
        cbs = [_cb for _cb in cbs if _cb.monitor == f"{prefix}{monitor}"]
        assert len(cbs) > 0, f"Monitor '{monitor}' not found in model checkpoints: {monitors} (prefix: {prefix})"
        if len(cbs) > 1:
            logger.debug(f"More than one callback for monitor '{monitor}' found: {cbs}")
        return cbs[0]

    # [fit/test/predict]_start private methods

    def _log_timestamp_start(self, prefix: str) -> dict:
        """Logs the timestamp of fit_start or test_start"""
        now = datetime.now()
        res = {
            f"{prefix}_start_timestamp": datetime.timestamp(now),
            f"{prefix}_start_date": f"{now}",
        }
        if prefix == "fit":
            res["epoch_timestamps"] = []
        return res

    def _log_optimizer_fit_start(self, pl_module: "LME") -> dict | list[dict]:
        """optimizer metadata at fit start"""
        def _log_one_optimizer_fit_start(optimizer: tr.optim.Optimizer) -> dict:
            return {
                "type": parsed_str_type(optimizer),
                "starting_lr": self._get_optimizer_current_lr(optimizer),
                "lr_history": [],
            }
        res = [_log_one_optimizer_fit_start(o) for o in make_list(pl_module.optimizer)]
        return flat_if_one(res)

    def _log_scheduler_fit_start(self, pl_module: "LME") -> dict | list[dict] | None:
        """logs information about the scheduler, if it exists"""
        def _log_one_scheduler_fit_start(scheduler_dict: dict) -> dict:
            return {
                "type": parsed_str_type(scheduler_dict["scheduler"]),
                **{k: v for k, v in scheduler_dict.items() if k != "scheduler"}
            }
        if pl_module.scheduler is None:
            return None
        res = [_log_one_scheduler_fit_start(sch) for sch in make_list(pl_module.scheduler)]
        return flat_if_one(res)

    def _log_early_stopping_fit_start(self, pl_module: "LME"):
        assert pl_module.trainer is not None, "Invalid call to this function, trainer is not set."
        early_stopping_cbs = list(filter(lambda x: isinstance(x, EarlyStopping), pl_module.trainer.callbacks))
        # no early stopping for this train, simply return
        if len(early_stopping_cbs) == 0:
            return None
        assert len(early_stopping_cbs) == 1, early_stopping_cbs
        early_stopping_cb: EarlyStopping = early_stopping_cbs[0]
        es_dict = {
            "monitor": early_stopping_cb.monitor,
            "min_delta": early_stopping_cb.min_delta,
            "mode": early_stopping_cb.mode,
            "patience": early_stopping_cb.patience
        }
        return es_dict

    def _log_model_checkpoint_fit_start(self, pl_module: "LME") -> dict:
        cb = self._get_monitored_model_checkpoint(pl_module)
        if cb is None:
            return {}
        return {"monitors": cb.monitor, "mode": cb.mode}

    # [fit/test/predict]_end private methods

    def _log_timestamp_end(self, prefix: str):
        """Adds the end timestamp and saves the json on the disk for train and test modes."""
        now = datetime.now()
        start_timestamp = datetime.fromtimestamp(self.metadata[f"{prefix}_start_timestamp"])
        res = {
            f"{prefix}_end_timestamp": datetime.timestamp(now),
            f"{prefix}_end_date": f"{now}",
            f"{prefix}_duration": f"{now - start_timestamp}"
        }
        return res

    def _log_scheduler_best_model_fit_end(self, pl_module: "LME", best_epoch: int) -> int | None:
        """updates bset model dict with the number of learning rate reduces done by the scheduler during training"""
        if pl_module.scheduler is None:
            return None
        assert len(make_list(pl_module.optimizer)) == 1 # TODO: also in TrainableMododule
        assert len(make_list(pl_module.scheduler)) == 1, f"Only 1 scheduler support now, got {len(pl_module.scheduler)}"
        sch: tr.optim.lr_scheduler.LRScheduler = pl_module.scheduler["scheduler"]
        if not hasattr(sch, "factor"):
            logger.debug(f"Scheduler {sch} doesn't have a factor attribute")
            return None
        first_lr = self.metadata["optimizer"]["lr_history"][0]
        last_lr = self.metadata["optimizer"]["lr_history"][best_epoch]
        res = 0 if first_lr == last_lr else int(first_lr / last_lr * sch.factor)
        return res

    def _log_best_model_fit_end(self, pl_module: "LME") -> dict:
        # find best and last modelcheckpoint callbacks. best will be None if we don't use a validation set loader.
        cb = self._get_monitored_model_checkpoint(pl_module)
        if cb is None:
            return {}
        ckpt_path = Path(cb.best_model_path)
        assert ckpt_path.exists() and ckpt_path.is_file(), "Best checkpoint does not exist."

        best_model_pkl = tr.load(ckpt_path, map_location="cpu")
        best_model_dict = {
            "path": f"{ckpt_path}",
            "hyper_parameters": best_model_pkl.get("hyper_parameters", {}),
            "epoch": best_model_pkl["epoch"],
            "optimizer_lr": flat_if_one([self._get_optimizer_current_lr(o) for o in best_model_pkl["optimizer_states"]])
        }
        if (sch := self._log_scheduler_best_model_fit_end(pl_module, best_model_pkl["epoch"])) is not None:
            best_model_dict["scheduler_num_lr_reduced"] = sch
        return best_model_dict

    # train_epoch_end private methods

    def _log_timestamp_train_epoch_end(self) -> dict:
        """
        Compute the average durations from fit_start to now. If we retrain, this will be wron, because the first epochs
        will have appened at some timestamps and the new one to another timestamp.
        TODO: remove old ones or linearly approximate based on epoch_averagae_duration when they would've happen?
        """
        new_timestamps = [*self.metadata["epoch_timestamps"], datetime.timestamp(datetime.now())]
        tr_timestamps = tr.DoubleTensor([self.metadata["fit_start_timestamp"], *new_timestamps], device="cpu")
        res = {
            "epoch_timestamps": new_timestamps,
            "epoch_average_duration": (tr_timestamps[1:] - tr_timestamps[0:-1]).mean().item()
        }
        return res

    def _log_optimizer_lr_history_epoch_end(self, pl_module: "LME") -> list[list[float]]:
        """
        At the end of each epoch, for each optimizer (can be >1), append the current LR. Only 1 LR allowed per optim.
        Return: A list of list (all lrs for each optim), even if just 1: [[0.01, 0.001] (o1), [0.01, 0.01] (o2)]
        """
        res = []
        for o, o_meta in zip(make_list(pl_module.optimizer), make_list(self.metadata["optimizer"])):
            res.append([*o_meta["lr_history"], self._get_optimizer_current_lr(o)])
        return res

    def _debug_metadata_json_dump(self, metadata: dict[str, Any], fp: IO) -> None:
        logger.debug("=================== Debug metadata =====================")
        for k in metadata:
            try:
                json.dump({k: metadata[k]}, fp)
            except TypeError:
                logger.debug(f"Cannot serialize key '{k}'")
        logger.debug("=================== Debug metadata =====================")

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"Metadata Callback. Path: '{self.log_file_path}'"
