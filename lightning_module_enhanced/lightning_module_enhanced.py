"""Generic Pytorch Lightning Graph module on top of a Graph module"""
from typing import Dict, Callable, List, Union, Any, Sequence, Tuple, Optional
from copy import deepcopy
from overrides import overrides
import torch as tr
from torch import optim, nn
from torchinfo import summary, ModelStatistics
from pytorch_lightning import Callback, LightningModule
from torchmetrics import Metric

from nwutils.torch import tr_get_data as to_tensor, tr_to_device as to_device
from .logger import logger

from .metadata_logger import MetadataLogger
from .torchmetric_wrapper import TorchMetricWrapper

# pylint: disable=too-many-ancestors, arguments-differ, unused-argument, abstract-method
class LightningModuleEnhanced(LightningModule):
    """

        Generic pytorch-lightning module for predict-ml models.

        Callbacks and metrics are called based on the order described here:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#hooks

        Takes a :class:`torch.nn.Module` as the underlying model and implements
        all the training/validation/testing logic.

        Attributes:
            base_model: The base :class:`torch.nn.Module`.
            optimizer: The torch optimizer.
            scheduler: The torch scheduler.
            criterion_fn: The model's criterion.
            metrics: The evaluation metrics as dict `name: torch metric`.

        Args:
            base_model: The base :class:`torch.nn.Module`.
    """
    def __init__(self, base_model: nn.Module, *args, **kwargs):
        assert isinstance(base_model, nn.Module)
        super().__init__()
        self.base_model = base_model
        self.optimizer: optim.Optimizer = None
        self.scheduler_dict: optim.lr_scheduler._LRScheduler = None
        self._criterion_fn: Callable[[tr.Tensor, tr.Tensor], tr.Tensor] = None
        self._metrics: Dict[str, Metric] = {}
        self._logged_metrics: List[str] = []
        self._summary: ModelStatistics = None
        self.callbacks: List[Callback] = []
        self.metadata_logger: MetadataLogger = None

        hyper_parameters = {
            "args": args,
            "base_model": base_model.__class__.__name__,
            **kwargs
        }
        self.save_hyperparameters(hyper_parameters, ignore=["base_model"])


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
    def criterion_fn(self) -> Callable[[tr.Tensor, tr.Tensor], tr.Tensor]:
        return self._criterion_fn

    @criterion_fn.setter
    def criterion_fn(self, criterion_fn: Callable[[tr.Tensor, tr.Tensor], tr.Tensor]):
        assert isinstance(criterion_fn, Callable)
        self._criterion_fn = criterion_fn
        if len(self.metrics) == 0:
            self.metrics = {}

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
        assert "loss" in logged_metrics, "When manually settings logged_metrics, " \
                                         f"loss must be present. Got: {logged_metrics}"
        logger.info(f"Setting the logged metrics to {logged_metrics}")
        diff = set(logged_metrics).difference(self._metrics.keys())
        assert len(diff) == 0, f"Metrics {diff} are not in set metrics: {self._metrics.keys()}"
        self._logged_metrics = logged_metrics

    # Overrides on top of the standard pytorch lightning module

    @overrides(check_signature=False)
    def log(self, name: str, value: Any, prog_bar: bool = False, logger: bool = True, on_step: Optional[bool] = None,
            on_epoch: Optional[bool] = None, *args, **kwargs):

        # If it is a single value, call all the loggers
        if isinstance(value, tr.Tensor) and len(value.shape) == 0:
            super().log(name, value, prog_bar, logger, on_step, on_epoch, *args, **kwargs)
        if on_epoch is not True:
            return None
        # Othrerwise, call just the metadata logger, but only if it applies to the epoch since we only care about
        #  epoch metrics
        return self.metadata_logger.save_epoch_metric(name, value, self.trainer.current_epoch)

    @overrides
    def on_fit_start(self) -> None:
        self.metadata_logger = MetadataLogger(self, self.loggers[0].log_dir)
        logger.info(f"Adding metadata logger to this model: {self.metadata_logger}")

        if len(self.metrics) == 0:
            self.metrics = {}

        # We need to make a copy for all metrics if we have a validation dataloader, such that the global statistics
        #  are unique for each of these datasets.
        new_metrics = self.metrics
        if self.trainer.enable_validation:
            val_metrics = self._clone_all_metrics_with_prefix(prefix="val_")
            new_metrics = {**new_metrics, **val_metrics}
        self._metrics = new_metrics
        self.metadata_logger.save_metadata("fit_start_hparams", self.hparams)
        return super().on_fit_start()

    @overrides
    def on_test_start(self) -> None:
        self.metadata_logger = MetadataLogger(self, self.loggers[0].log_dir)
        logger.info(f"Adding metadata logger to this model: {self.metadata_logger}")

        test_metrics = self._clone_all_metrics_with_prefix(prefix="test_")
        self._metrics = test_metrics
        self.metadata_logger.save_metadata("test_start_hparams", self.hparams)
        return super().on_test_start()

    @overrides
    def training_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs) -> Union[tr.Tensor, Dict[str, Any]]:
        """Training step: returns batch training loss and metrics."""
        res = self._generic_batch_step(train_batch)
        return res

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
        breakpoint()
        self._run_and_log_metrics_at_epoch_end(self.metrics.keys())

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
        tr_args = to_device(args, self.device)
        tr_kwargs = to_device(kwargs, self.device)
        return self.base_model.forward(*tr_args, **tr_kwargs)

    def np_forward(self, *args, **kwargs):
        """Forward numpy data to the model, returns whatever the model returns, usually torch data"""
        tr_args = to_tensor(args)
        tr_kwargs = to_tensor(kwargs)
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

    def load_state_from_path(self, path: str):
        """Loads the state dict from a path"""
        # if path is remote (gcs) download checkpoint to a temp dir
        self.load_state_dict(tr.load(path)["state_dict"])
        return self

    # Internal methods

    def _generic_batch_step(self, train_batch: Dict, prefix: str = "") -> Dict[str, tr.Tensor]:
        """Generic step for computing the forward pass, loss and metrics."""
        y = self.forward(train_batch["data"])
        gt = to_device(to_tensor(train_batch["labels"]), self.device)
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
            # Get the metric's epoch result
            metric_epoch_result = metric_fn.compute()
            # Log the metric at the end of the epoch
            self.log(metric_name, metric_epoch_result, on_epoch=True)
            # Reset the metric after storing this epoch's value
            metric_fn.reset()
        self.metadata_logger.save()

    def _clone_all_metrics_with_prefix(self, prefix: str):
        """Clones all the existing metris, by ading a prefix"""
        assert len(prefix) > 0 and prefix[-1] == "_"
        new_metrics = {}
        for metric_name in self.metrics:
            assert not metric_name.startswith(prefix), f"This may be a bug, since metric '{metric_name}'" \
                                                       f"already has prefix '{prefix}'"
            new_metrics[f"{prefix}{metric_name}"] = deepcopy(self.metrics[metric_name])
        return new_metrics
