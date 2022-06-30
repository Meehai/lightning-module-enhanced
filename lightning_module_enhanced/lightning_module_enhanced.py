"""Generic Pytorch Lightning Graph module on top of a Graph module"""
from typing import Dict, Callable, List, Union, Any, Sequence, Tuple
from copy import deepcopy
from overrides import overrides
import torch as tr
from torch import optim, nn
from torchinfo import summary, ModelStatistics
from pytorch_lightning import Callback, LightningModule
from torchmetrics import Metric
from nwutils.torch import tr_get_data, tr_to_device

from .logger import logger
from .torchmetric_wrapper import TorchMetricWrapper

# pylint: disable=too-many-ancestors, arguments-differ, unused-argument, abstract-method
class LightningModuleEnhanced(LightningModule):
    """
        Pytorch Lightning module enhanced. Callbacks and metrics are called based on the order described here:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#hooks
    """
    def __init__(self, base_model: nn.Module, *args, **kwargs):
        assert isinstance(base_model, nn.Module)
        super().__init__()
        self.save_hyperparameters({"args": args, **kwargs})
        self.base_model = base_model
        self.optimizer: optim.Optimizer = None
        self.scheduler_dict: optim.lr_scheduler._LRScheduler = None
        self.criterion_fn: Callable[[tr.Tensor, tr.Tensor], float] = None
        self._metrics: Dict[str, Metric] = {}
        self._logged_metrics: List[str] = []
        self._summary: ModelStatistics = None
        self.callbacks: List[Callback] = []

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
    def summary(self, **kwargs) -> ModelStatistics:
        """Prints the summary (layers, num params, size in MB), with the help of torchinfo module."""
        self._summary = summary(self.base_model, verbose=0, **kwargs) if self._summary is None else self._summary
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
    def metrics(self) -> List[str]:
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Dict[str, Tuple[Callable, str]]):
        if len(self._metrics) != 0:
            logger.info(f"Overwriting existing metrics: {list(metrics.keys())}")
        self._metrics = {}
        for metric_name, metric_fn in metrics.items():
            assert isinstance(metric_fn, (Metric, Tuple, Callable)), f"Unknown metric type: '{type(metric_fn)}'. " \
                   "Expcted torchmetrics.Metric, Tuple[Callable, str] or Callable."
            if not isinstance(metric_fn, Metric):
                logger.debug(f"Metric '{metric_name}' is a callable. Converting to torchmetrics.Metric.")
                min_or_max = "min"
                if isinstance(metric_fn, Tuple):
                    metric_fn, min_or_max = metric_fn
                assert isinstance(metric_fn, Callable) and isinstance(min_or_max, str) and min_or_max in ("min", "max")
                metric_fn = TorchMetricWrapper(metric_fn, higher_is_better=(min_or_max == "max"))

            self._metrics[metric_name] = metric_fn
        self._metrics["loss"] = TorchMetricWrapper(self.criterion_fn, higher_is_better=False)
        self._metrics["loss"]._enable_grad = True
        self.logged_metrics = list(self._metrics.keys())

    @property
    def logged_metrics(self) -> List[str]:
        return self._logged_metrics

    @logged_metrics.setter
    def logged_metrics(self, logged_metrics: List[str]):
        logger.info(f"Setting the logged metrics to {logged_metrics}")
        diff = set(logged_metrics).difference(self._metrics.keys())
        assert len(diff) == 0, f"Metrics {diff} are not in set metrics: {self._metrics.keys()}"
        self._logged_metrics = logged_metrics

    @overrides
    def on_fit_start(self) -> None:
        # We need to make a copy for all metrics if we have a validation dataloader, such that the global statistics
        #  are unique for each of these datasets.
        new_metrics = self.metrics
        if self.trainer.enable_validation:
            val_metrics = self._clone_all_metrics_with_prefix(prefix="val_")
            new_metrics = {**new_metrics, **val_metrics}
        self._metrics = new_metrics
        return super().on_fit_start()

    @overrides
    def on_test_start(self) -> None:
        test_metrics = self._clone_all_metrics_with_prefix(prefix="test_")
        self._metrics = test_metrics
        return super().on_test_start()

    @overrides
    def training_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs) -> Union[tr.Tensor, Dict[str, Any]]:
        """Training step: returns batch training loss and metrics."""
        return self._generic_batch_step(train_batch)

    @overrides
    def validation_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs):
        """Training step: returns batch validation loss and metrics."""
        return self._generic_batch_step(train_batch, prefix="val_")

    @overrides
    def test_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs):
        """Training step: returns batch validation loss and metrics."""
        return self._generic_batch_step(train_batch, prefix="test_")

    @overrides
    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return self.callbacks

    @overrides
    def training_epoch_end(self, outputs):
        """Computes epoch average train loss and metrics for logging."""
        self._on_epoch_end()

    @overrides
    def test_epoch_end(self, outputs):
        for metric_name in self.metrics.keys():
            metric_fn: Metric = self.metrics[metric_name]
            metric_epoch_result = metric_fn.compute()
            self.log(metric_name, metric_epoch_result, on_epoch=True)
            # Reset the metric after storing this epoch's value
            metric_fn.reset()

    @overrides
    def configure_optimizers(self) -> Dict:
        """Configure the optimizer/scheduler/monitor."""
        assert self.optimizer is not None, "No optimizer has been provided. Set a torch optimizer first."

        if self.scheduler_dict is None:
            return {"optimizer": self.optimizer}

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler_dict
        }

    # Public methods

    def forward(self, *args, **kwargs):
        tr_args = tr_to_device(args, self.device)
        tr_kwargs = tr_to_device(kwargs, self.device)
        return self.base_model.forward(*tr_args, **tr_kwargs)

    def np_forward(self, *args, **kwargs):
        """Forward numpy data to the model, returns whatever the model returns, usually torch data"""
        tr_args = tr_get_data(args)
        tr_kwargs = tr_get_data(kwargs)
        with tr.no_grad():
            y_tr = self.forward(*tr_args, **tr_kwargs)
        return y_tr

    def reset_parameters(self):
        """Resets the parameters of the base model"""
        for layer in self.base_model.children():
            assert hasattr(layer, "reset_parameters")
            layer.reset_parameters()

    def setup_module_for_train(self, train_cfg: Dict):
        """Given a train cfg, prepare this module for training, by setting the required information."""
        from .train_setup import TrainSetup
        TrainSetup(self, train_cfg).setup()

    # Internal methods

    def _generic_batch_step(self, train_batch: Dict, prefix: str = "") -> Dict[str, tr.Tensor]:
        """Generic step for computing the forward pass, loss and metrics."""
        y = self.forward(train_batch["data"])
        gt = tr_to_device(tr_get_data(train_batch["labels"]), self.device)
        outputs = self._get_batch_metrics(y, gt, prefix)
        return outputs

    def _get_batch_metrics(self, y, gt, prefix: str) -> Dict[str, tr.Tensor]:
        """Get batch-level metrics"""
        outputs = {}
        for metric_name in self.logged_metrics:
            prefixed_metric_name = f"{prefix}{metric_name}"
            metric_fn: Metric = self.metrics[prefixed_metric_name]
            # Call the metric and update its state
            metric_output: tr.Tensor = metric_fn.forward(y, gt)
            metric_fn.update(metric_output)
            outputs[prefixed_metric_name] = metric_output
            # Log all the numeric batch metrics. Don't show on pbar.
            if isinstance(metric_output, tr.Tensor) and len(metric_output.shape) == 0:
                self.log(prefixed_metric_name, metric_output, prog_bar=False, on_step=True, batch_size=1)
        return outputs

    # pylint: disable=no-member
    def _on_epoch_end(self):
        """Get epoch-level metrics"""
        # If validation is enabled (for train loops), add "val_" metrics for all logged metrics.
        if self.trainer.enable_validation:
            val_logged_metrics = [f"val_{metric_name}" for metric_name in self.logged_metrics]
            metrics_to_log = [*self.logged_metrics, *val_logged_metrics]
        self._run_and_log_metrics_at_epoch_end(metrics_to_log)

    def _run_and_log_metrics_at_epoch_end(self, metrics_to_log: List[str]):
        """Runs and logs a given list of logged metrics. Assume they all exist in self.metrics"""
        for metric_name in metrics_to_log:
            metric_fn: Metric = self.metrics[metric_name]
            metric_epoch_result = metric_fn.compute()
            self.log(metric_name, metric_epoch_result, on_epoch=True)
            # Reset the metric after storing this epoch's value
            metric_fn.reset()

    def _clone_all_metrics_with_prefix(self, prefix: str):
        """Clones all the existing metris, by ading a prefix"""
        assert len(prefix) > 0 and prefix[-1] == "_"
        new_metrics = {}
        for metric_name in self.metrics:
            assert not metric_name.startswith(prefix), f"This may be a bug, since metric '{metric_name}'" \
                                                       f"already has prefix '{prefix}'"
            new_metrics[f"{prefix}{metric_name}"] = deepcopy(self.metrics[metric_name])
        return new_metrics
