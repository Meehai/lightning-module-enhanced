"""Generic Pytorch Lightning Graph module on top of a Graph module"""
from typing import Dict, Callable, List, Union, Any, Sequence
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
    def metrics(self, metrics: Dict[str, Callable]):
        if len(self._metrics) != 0:
            logger.info(f"Overwriting existing metrics: {list(metrics.keys())}")
        self._metrics = {}
        for metric_name, metric_fn in metrics.items():
            assert isinstance(metric_fn, (Metric, Callable)), \
                f"Unknown metric type: '{type(metric_fn)}'. Expcted torchmetrics.Metric or Callable."
            if not isinstance(metric_fn, Metric):
                logger.debug(f"Metric '{metric_name}' is a callable. Converting to torchmetrics.Metric.")
                metric_fn = TorchMetricWrapper(metric_fn)

            self._metrics[metric_name] = metric_fn
        self._metrics["loss"] = TorchMetricWrapper(self.criterion_fn)
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

    # Pytorch lightning overrides

    @overrides
    def on_train_start(self) -> None:
        # We need to make a copy for all metrics if we have a validation dataloader, such that the global statistics
        #  are unique for each of these datasets.
        new_metrics = self.metrics
        prefix = "val_"
        if len(self.trainer.val_dataloaders) > 0:
            for metric_name in self.logged_metrics:
                new_metrics[f"{prefix}{metric_name}"] = deepcopy(self.metrics[metric_name])
        self._metrics = new_metrics
        return super().on_train_start()

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
        """Computes average test loss and metrics for logging."""
        self._on_epoch_end()

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
            metric_output = metric_fn.update(y, gt)
            outputs[prefixed_metric_name] = metric_output
            if metric_fn.compute_on_step:
                self.log(prefixed_metric_name, outputs[prefixed_metric_name], prog_bar=True, on_step=True)
        return outputs

    # pylint: disable=no-member
    def _on_epoch_end(self):
        """Get epoch-level metrics"""
        logged_metrics = self.logged_metrics
        if len(self.trainer.val_dataloaders) > 0:
            val_logged_metrics = [f"val_{metric_name}" for metric_name in logged_metrics]
            logged_metrics = [*logged_metrics, *val_logged_metrics]
        for metric_name in logged_metrics:
            metric_fn = self.metrics[metric_name]
            metric_epoch_result = metric_fn.compute()
            self.log(metric_name, metric_epoch_result, on_epoch=True)
            # Reset the metric after storing this epoch's value
            metric_fn.reset()

    def reset_parameters(self):
        """Resets the parameters of the base model"""
        for layer in self.base_model.children():
            assert hasattr(layer, "reset_parameters")
            layer.reset_parameters()

    def setup_module_for_train(self, train_cfg: Dict):
        """Given a train cfg, prepare this module for training, by setting the required information."""
        from .train_setup import TrainSetup
        TrainSetup(self, train_cfg).setup()
