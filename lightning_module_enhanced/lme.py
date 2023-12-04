"""Generic Pytorch Lightning Graph module on top of a Graph module"""
from __future__ import annotations
from typing import Any, Sequence, Callable, Union
from copy import deepcopy
from pathlib import Path
import shutil
from overrides import overrides
import torch as tr
import pytorch_lightning as pl
from lightning_fabric.utilities.seed import seed_everything
from lightning_fabric.utilities.exceptions import MisconfigurationException
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn, optim
from torchinfo import summary, ModelStatistics

from .trainable_module import TrainableModuleMixin
from .metrics import CoreMetric
from .logger import logger
from .utils import to_tensor, to_device, tr_detach_data

# pylint: disable=too-many-ancestors, arguments-differ, unused-argument, abstract-method
class LightningModuleEnhanced(TrainableModuleMixin, pl.LightningModule):
    """

        Generic pytorch-lightning module for ml models.

        Callbacks and metrics are called based on the order described here:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#hooks

        Takes a :class:`torch.nn.Module` as the underlying model and implements
        all the training/validation/testing logic in a unified way.

        Attributes:
            base_model: The base :class:`torch.nn.Module`
            optimizer: The optimizer used for training runs
            scheduler_dict: The oprimizer scheduler used for training runs, as well as the monitored metric
            criterion_fn: The criterion function used for training runs
            metrics: The dictionary (Name => CoreMetric) of all implemented metrics for this module
            callbacks: The list of callbacks for this module
            summary: A summary including parameters, layers and tensor shapes of this module
            metadata_callback: The metadata logger for this run
            device: The torch device this module is residing on
            trainable_params: The number of trainable parameters of this module

        Args:
            base_model: The base :class:`torch.nn.Module`
    """
    def __init__(self, base_model: nn.Module):
        assert isinstance(base_model, nn.Module), f"Expected a nn.Module, got {type(base_model)}"
        if isinstance(base_model, LightningModuleEnhanced):
            raise ValueError("Cannot have nested LME modules. LME must extend only a basic torch nn.Module")
        super().__init__()
        self.base_model = base_model
        self.automatic_optimization = False
        self._active_run_metrics: dict[str, dict[str, CoreMetric]] = {}
        self._summary: ModelStatistics = None
        self._model_algorithm = LightningModuleEnhanced.feed_forward_algorithm
        self.cache_result = None

    # Getters and setters for properties

    @property
    def device(self) -> tr.device:
        """Gets the device of the model, assuming all parameters are on the same device."""
        return next(self.base_model.parameters()).device

    @property
    def num_params(self) -> int:
        """Returns the total number of parameters of this module"""
        return self.summary.total_params

    @property
    def num_trainable_params(self) -> int:
        """Returns the trainable number of parameters of this module"""
        return self.summary.trainable_params

    @property
    def summary(self) -> ModelStatistics:
        """Prints the summary (layers, num params, size in MB), with the help of torchinfo module."""
        self._summary = summary(self.base_model, verbose=0, depth=3) if self._summary is None else self._summary
        return self._summary

    @property
    def trainable_params(self) -> bool:
        """Checks if the module is trainable"""
        return self.num_trainable_params > 0

    @trainable_params.setter
    def trainable_params(self, value: bool):
        """Sets all the parameters of this module to trainable or untrainable"""
        logger.debug(f"Setting parameters of the model to '{value}'.")
        for param in self.base_model.parameters():
            param.requires_grad_(value)
        # Reset summary such that it is recomputted if necessary (like for counting num trainable params)
        self._summary = None

    @property
    def model_algorithm(self) -> callable:
        """The model algorithm, used at both training, validation, test and inference."""
        return self._model_algorithm

    @model_algorithm.setter
    def model_algorithm(self, value: callable):
        assert isinstance(value, Callable), f"Expected a callable, got {type(value)}"
        self._model_algorithm = value

    # Overrides on top of the standard pytorch lightning module

    @overrides
    def on_fit_start(self) -> None:
        self._active_run_metrics[""] = self.metrics
        if self.trainer.enable_validation:
            cloned_metrics = deepcopy(self.metrics)
            self._active_run_metrics["val_"] = cloned_metrics
        self._reset_all_active_metrics()
        self._set_metrics_running_model()
        self._copy_loaded_checkpoints()

    @overrides
    def on_fit_end(self):
        # note: order is important here
        self._unset_metrics_running_model()
        self._active_run_metrics = {}

    @overrides
    def on_test_start(self) -> None:
        self._active_run_metrics[""] = self.metrics
        self._set_metrics_running_model()

    @overrides
    def on_test_end(self):
        # note: order is important here
        self._unset_metrics_running_model()
        self._active_run_metrics = {}

    @overrides
    def training_step(self, batch: dict, batch_idx: int, *args, **kwargs) -> Union[tr.Tensor, dict[str, Any]]:
        """Training step: returns batch training loss and metrics."""
        # Warning: if not using lightning's self.optimizers(), and rather trying to user our self.optimizer, will
        # result in checkpoints not being saved.
        # After more digging it's because self.optimizers() doesn't return self.optimizer (the torch optimizer), but
        # rather lightning's warpper on top of it that can be used using other trainer strategies (ddp) and also
        # updates some internal states, like trainer.global_step.
        _opt = self.optimizers()
        opts: list[LightningOptimizer] = _opt if isinstance(_opt, list) else [_opt]
        for opt in opts:
            opt.zero_grad()
        train_metrics = self.model_algorithm(self, batch)
        assert isinstance(train_metrics, dict), type(train_metrics)
        assert "loss" in train_metrics.keys(), train_metrics.keys()
        self._update_metrics_at_batch_end(train_metrics)
        # Manual optimization like real men. We disable automatic_optimization in the constructor.
        self.manual_backward(train_metrics["loss"])
        for opt in opts:
            opt.step()

    @overrides
    def validation_step(self, batch: dict, batch_idx: int, *args, **kwargs):
        """Validation step: returns batch validation loss and metrics."""
        val_metrics = self.model_algorithm(self, batch)
        assert isinstance(val_metrics, dict), type(val_metrics)
        assert "loss" in val_metrics.keys(), val_metrics.keys()
        self._update_metrics_at_batch_end(val_metrics)

    @overrides
    def test_step(self, batch: dict, batch_idx: int, *args, **kwargs):
        """Testing step: returns batch test loss and metrics."""
        test_metrics = self.model_algorithm(self, batch)
        assert isinstance(test_metrics, dict), type(test_metrics)
        self._update_metrics_at_batch_end(test_metrics)

    @overrides
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch["data"])

    @overrides
    def configure_callbacks(self) -> Sequence[pl.Callback] | pl.Callback:
        return self.callbacks

    @overrides
    def on_train_epoch_end(self) -> None:
        """Computes epoch average train loss and metrics for logging."""
        # If validation is enabled (for train loops), add "val_" metrics for all logged metrics.
        self._run_and_log_metrics_at_epoch_end(self.metrics.keys())
        self._reset_all_active_metrics()
        if self.scheduler_dict is not None:
            scheduler: optim.lr_scheduler.LRScheduler = self.scheduler_dict["scheduler"]
            monitor: str = self.scheduler_dict["monitor"]
            # on_train_epoch_end.val_loss
            # TODO: perhaps find some better way to do this
            # pylint: disable=protected-access
            try:
                epoch_result: tr.Tensor = self._trainer._results[f"on_train_epoch_end.{monitor}"].value
            except KeyError:
                logger.debug(f"It may be the case that your scheduler monitor is wrong. Monitor: '{monitor}'. "
                             f"All results: {list(self._trainer._results.keys())}")
                raise MisconfigurationException
            # TODO: this assumes that it's reduce lr on plateau and not something else.
            scheduler.step(epoch_result)

    @overrides
    def on_test_epoch_end(self):
        self._run_and_log_metrics_at_epoch_end(self.metrics.keys())
        self._reset_all_active_metrics()

    # TODO: perhaps refactor this since we don't need to have a dict like this with automatic optimization turned off
    # main idea would be to return just the optimizer here, and no scheduler dict (throws warning now) and have
    # a self.scheduler that does this behind the scenes
    @overrides
    def configure_optimizers(self) -> dict:
        """Configure the optimizer/scheduler/monitor."""
        if self.optimizer is None:
            raise ValueError("No optimizer. Use model.optimizer=optim.XXX or add an optimizer property in base model")

        if isinstance(self.optimizer, list):
            res = [{"optimizer": o} for o in self.optimizer]
        else:
            res = [{"optimizer": self.optimizer}]

        if self.scheduler_dict is None:
            return res

        if isinstance(self.scheduler_dict, list):
            res_scheduler = [{"lr_scheduler": s} for s in self.scheduler_dict]
        else:
            res_scheduler = [{"lr_scheduler": self.scheduler_dict}]

        assert len(res) == len(res_scheduler), "Something is messed up in your configs. Num optimizer: " \
            f"{res} ({len(res)}), schedulers: {res_scheduler} ({len(res_scheduler)})"

        for i in range(len(res)):
            res[i] = {**res[i], **res_scheduler[i]}
        return res

    # Public methods

    def forward(self, *args, **kwargs):
        tr_args = to_device(args, self.device)
        tr_kwargs = to_device(kwargs, self.device)
        res = self.base_model.forward(*tr_args, **tr_kwargs)
        return res

    def np_forward(self, *args, **kwargs):
        """Forward numpy data to the model, returns whatever the model returns, usually torch data"""
        tr_args = to_tensor(args)
        tr_kwargs = to_tensor(kwargs)
        with tr.no_grad():
            y_tr = self.forward(*tr_args, **tr_kwargs)
        return y_tr

    def reset_parameters(self, seed: int = None):
        """Resets the parameters of the base model. Applied recursively as much as possible."""
        if seed is not None:
            seed_everything(seed)
        num_params = len(tuple(self.parameters()))
        if num_params == 0:
            return
        for layer in self.base_model.children():
            if LightningModuleEnhanced(layer).num_params == 0:
                continue

            if not hasattr(layer, "reset_parameters"):
                logger.debug2(f"Layer {layer} has params, but no reset_parameters() method. Trying recursively")
                layer = LightningModuleEnhanced(layer)
                layer.reset_parameters(seed)
            else:
                layer.reset_parameters()

    def load_state_from_path(self, path: str) -> LightningModuleEnhanced:
        """Loads the state dict from a path"""
        # if path is remote (gcs) download checkpoint to a temp dir
        logger.info(f"Loading weights and hyperparameters from '{Path(path).absolute()}'")
        ckpt_data = tr.load(path, map_location="cpu")
        self.load_state_dict(ckpt_data["state_dict"])
        if "hyper_parameters" in ckpt_data:
            self.save_hyperparameters(ckpt_data["hyper_parameters"])
        return self

    def state_dict(self):
        return self.base_model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        return self.base_model.load_state_dict(state_dict, strict)

    @staticmethod
    def feed_forward_algorithm(model: LightningModuleEnhanced, batch: dict) -> dict[str, tr.Tensor]:
        """
        Generic step for computing the forward pass, loss and metrics. Simple feed-forward algorithm by default.
        Must return a dict of type: {metric_name: metric_tensor} for all metrics.
        'loss' must be in there as well unless you update `training_step` as well in your module.
        """
        x = batch["data"]
        assert isinstance(x, (dict, tr.Tensor)), type(x)
        # This allows {"data": {"a": ..., "b": ...}} to be mapped to forward(a, b)
        y = model.forward(x)
        model.cache_result = tr_detach_data(y)
        gt = to_device(to_tensor(batch["labels"]), model.device)
        return model.lme_metrics(y, gt, include_loss=True)

    def lme_metrics(self, y: tr.Tensor, gt: tr.Tensor, include_loss: bool = True) -> dict[str, tr.Tensor]:
        """
        Pass through all the metrics of this batch and call forward. This updates the metric state for this batch
        Parameters:
        - y the output of the model
        - gt the ground truth
        - include_loss Whether to include the loss in the returned metrics. This can be useful when using a different
        model_algorithm, where we want to compute the loss manually as well.
        """
        prefix = self._prefix_from_trainer()
        if prefix not in self._active_run_metrics:
            raise KeyError(f"Prefix '{prefix}' not found in active run metrics. Set model.metrics={{...}} first. "
                           "Also, this method is meant to be ran from a pl.Trainer.fit() call, not manually.")

        metrics = {}
        for metric_name, metric_fn in self._active_run_metrics[prefix].items():
            if metric_name == "loss" and not include_loss:
                continue
            metric_fn: CoreMetric = self._active_run_metrics[prefix][metric_name]
            # Call the metric and update its state
            with (tr.enable_grad if metric_fn.requires_grad else tr.no_grad)():
                metrics[metric_name]: tr.Tensor = metric_fn.forward(y, gt)
        return metrics

    # Private methods

    def _update_metrics_at_batch_end(self, batch_results: dict[str, tr.Tensor]):
        prefix = self._prefix_from_trainer()
        if set(batch_results.keys()) != set(self.metrics.keys()):
            raise ValueError(f"Not all expected metrics ({self.metrics.keys()}) were computed "
                             f"this batch: {batch_results.keys()}")
        for metric_name, metric in self._active_run_metrics[prefix].items():
            metric.batch_update(tr_detach_data(batch_results[metric_name]))

    def _run_and_log_metrics_at_epoch_end(self, metrics_to_log: list[str]):
        """Runs and logs a given list of logged metrics. Assume they all exist in self.metrics"""
        all_prefixes = self._active_run_metrics.keys()
        for metric_name in metrics_to_log:
            for prefix in all_prefixes:
                metric_fn: CoreMetric = self._active_run_metrics[prefix][metric_name]
                # Get the metric's epoch result
                metric_epoch_result = metric_fn.epoch_result()
                # Log the metric at the end of the epoch. Only log on pbar the val_loss, loss is tracked by default
                prog_bar = (metric_name == "loss" and prefix == "val_")

                value_reduced = metric_fn.epoch_result_reduced(metric_epoch_result)
                if value_reduced is not None:
                    self.log(f"{prefix}{metric_name}", value_reduced, prog_bar=prog_bar, on_epoch=True)
                # Call the metadata callback for the full result, since it can handle any sort of metrics
                self.metadata_callback.log_epoch_metric(metric_name, metric_epoch_result,
                                                        self.trainer.current_epoch, prefix)

    def _reset_all_active_metrics(self):
        """ran at epoch end to reset the metrics"""
        for prefix in self._active_run_metrics.keys():
            for metric in self._active_run_metrics[prefix].values():
                metric.reset()

    def _set_metrics_running_model(self):
        """ran at fit/test start to set the running model"""
        for prefix in self._active_run_metrics.keys():
            for metric in self._active_run_metrics[prefix].values():
                metric.running_model = lambda: self

    def _unset_metrics_running_model(self):
        """ran at fit/test end to unset the running model"""
        for prefix in self._active_run_metrics.keys():
            for metric in self._active_run_metrics[prefix].values():
                metric.running_model = None

    def _copy_loaded_checkpoints(self):
        """copies the loaded checkpoint to the log dir"""
        ckpt_dir = Path(self.logger.log_dir) / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True, parents=False)
        if self.trainer.ckpt_path is not None:
            shutil.copyfile(self.trainer.ckpt_path, ckpt_dir / "loaded.ckpt")
            best_model_path = Path(self.trainer.checkpoint_callback.best_model_path)
            if best_model_path.exists() and best_model_path.is_file():
                new_best_path = ckpt_dir / best_model_path.name
                shutil.copyfile(best_model_path, new_best_path)
            logger.debug("Loaded best and last checkpoint before resuming.")

    def _prefix_from_trainer(self) -> str:
        """returns the prefix: "" (for training), "val_" for validating or "" for testing as well"""
        assert self.trainer.training or self.trainer.validating or self.trainer.testing \
            or self.trainer.sanity_checking, self.trainer.state
        prefix = "" if self.trainer.training or self.trainer.testing else "val_"
        return prefix
