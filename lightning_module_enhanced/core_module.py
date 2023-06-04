"""Generic Pytorch Lightning Graph module on top of a Graph module"""
from __future__ import annotations
from typing import Dict, List, Union, Any, Sequence
from copy import deepcopy
from pathlib import Path
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
class CoreModule(TrainableModuleMixin, pl.LightningModule):
    """

        Generic pytorch-lightning module for ml models.

        Callbacks and metrics are called based on the order described here:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#hooks

        Takes a :class:`torch.nn.Module` as the underlying model and implements
        all the training/validation/testing logic.

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
        super().__init__()
        self.base_model = base_model
        self.automatic_optimization = False
        self._active_run_metrics: Dict[str, Dict[str, CoreMetric]] = {}
        self._summary: ModelStatistics = None

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

    # Overrides on top of the standard pytorch lightning module
    @overrides
    def on_fit_start(self) -> None:
        self._active_run_metrics[""] = self.metrics
        if self.trainer.enable_validation:
            cloned_metrics = deepcopy(self.metrics)
            self._active_run_metrics["val_"] = cloned_metrics
        self._reset_all_active_metrics()
        self._set_metrics_running_model()

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
    def training_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs) -> Union[tr.Tensor, Dict[str, Any]]:
        """Training step: returns batch training loss and metrics."""
        # Warning: if not using lightning's self.optimizers(), and rather trying to user our self.optimizer, will
        # result in checkpoints not being saved.
        # After more digging it's because self.optimizers() doesn't return self.optimizer (the torch optimizer), but
        # rather lightning's warpper on top of it that can be used using other trainer strategies (ddp) and also
        # updates some internal states, like trainer.global_step.
        _opt = self.optimizers()
        opts: List[LightningOptimizer] = _opt if isinstance(_opt, list) else [_opt]
        for opt in opts:
            opt.zero_grad()
        train_metrics = self.model_algorithm(train_batch)
        self._update_metrics_at_batch_end(train_metrics)
        # Manual optimization like real men. We disable automatic_optimization in the constructor.
        self.manual_backward(train_metrics["loss"])
        for opt in opts:
            opt.step()

    @overrides
    def validation_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs):
        """Validation step: returns batch validation loss and metrics."""
        val_metrics = self.model_algorithm(train_batch, prefix="val_")
        self._update_metrics_at_batch_end(val_metrics, prefix="val_")

    @overrides
    def test_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs):
        """Testing step: returns batch test loss and metrics. No prefix."""
        test_metrics =  self.model_algorithm(train_batch)
        self._update_metrics_at_batch_end(test_metrics)

    @overrides
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch["data"])

    @overrides
    def configure_callbacks(self) -> Union[Sequence[pl.Callback], pl.Callback]:
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
    def configure_optimizers(self) -> Dict:
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
            if CoreModule(layer).num_params == 0:
                continue

            if not hasattr(layer, "reset_parameters"):
                logger.debug2(f"Layer {layer} has params, but no reset_parameters() method. Trying recursively")
                layer = CoreModule(layer)
                layer.reset_parameters(seed)
            else:
                layer.reset_parameters()

    def load_state_from_path(self, path: str) -> CoreModule:
        """Loads the state dict from a path"""
        # if path is remote (gcs) download checkpoint to a temp dir
        logger.info(f"Loading weights and hyperparameters from '{Path(path).absolute()}'")
        ckpt_data = tr.load(path, map_location="cpu")
        self.load_state_dict(ckpt_data["state_dict"])
        self.save_hyperparameters(ckpt_data["hyper_parameters"])
        return self

    def state_dict(self):
        return self.base_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        return self.base_model.load_state_dict(state_dict, strict)

    def model_algorithm(self, train_batch: Dict, prefix: str = "") -> Dict[str, tr.Tensor]:
        """
        Generic step for computing the forward pass, loss and metrics. Simple feed-forward algorithm by default.
        Must return a dict of type: {metric_name: metric_tensor} for all metrics.
        'loss' must be in there as well unless you update `training_step` as well in your module.
        """
        # TODO: simplify data/labels requirement. Some models have other forward() header, like gcns having
        # (x, edge_index) always. Perhaps if 'data' has a tensor, it's assumed 'x', otherwise, if it has
        # a dict, all of them are passed automatically: {'data': {'x': ..., 'edge_index': ...}, 'gt': ...}
        y = self.forward(train_batch["data"])
        gt = to_device(to_tensor(train_batch["labels"]), self.device)
        return self.lme_metrics(y, gt, prefix)

    def lme_metrics(self, y: tr.Tensor, gt: tr.Tensor, prefix: str) -> Dict[str, tr.Tensor]:
        """pass through all the metrics of this batch and call forward. This updates the metric state for this batch"""
        metrics = {}
        for metric_name, metric_fn in self._active_run_metrics[prefix].items():
            metric_fn: CoreMetric = self._active_run_metrics[prefix][metric_name]
            # Call the metric and update its state
            with (tr.enable_grad if metric_fn.requires_grad else tr.no_grad)():
                metrics[metric_name]: tr.Tensor = metric_fn.forward(y, gt)
        return metrics

    def _update_metrics_at_batch_end(self, batch_results: Dict[str, tr.Tensor], prefix: str = ""):
        if set(batch_results.keys()) != set(self.metrics.keys()):
            raise ValueError(f"Not all expected metrics ({self.metrics.keys()}) were computed "
                             f"this batch: {batch_results.keys()}")
        for metric_name, metric in self._active_run_metrics[prefix].items():
            metric.batch_update(tr_detach_data(batch_results[metric_name]))

    def _run_and_log_metrics_at_epoch_end(self, metrics_to_log: List[str]):
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
