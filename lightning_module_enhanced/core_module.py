"""Generic Pytorch Lightning Graph module on top of a Graph module"""
from __future__ import annotations
from typing import Dict, Callable, List, Union, Any, Sequence, Tuple
from copy import deepcopy
from overrides import overrides
import torch as tr
import pytorch_lightning as pl
from torch import optim, nn
from torchinfo import summary, ModelStatistics

from .metrics import CoreMetric, CallableCoreMetric
from .logger import logger
from .callbacks import MetadataCallback
from .train_setup import TrainSetup
from .utils import to_tensor, to_device

# pylint: disable=too-many-ancestors, arguments-differ, unused-argument, abstract-method
class CoreModule(pl.LightningModule):
    """

        Generic pytorch-lightning module for predict-ml models.

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
            logged_metrics: The subset of metrics used for this run (train or test)
            callbacks: The list of callbacks for this module
            summary: A summary including parameters, layers and tensor shapes of this module
            metadata_callback: The metadata logger for this run
            device: The torch device this module is residing on
            trainable_params: The number of trainable parameters of this module

        Args:
            base_model: The base :class:`torch.nn.Module`
    """
    def __init__(self, base_model: nn.Module, *args, **kwargs):
        assert isinstance(base_model, nn.Module)
        super().__init__()
        self.base_model = base_model
        self.optimizer: optim.Optimizer = None
        self.scheduler_dict: optim.lr_scheduler._LRScheduler = None
        self._criterion_fn: Callable[[tr.Tensor, tr.Tensor], tr.Tensor] = None
        self._metrics: Dict[str, CoreMetric] = {}
        self._logged_metrics: List[str] = []
        self._summary: ModelStatistics = None
        self._callbacks: List[pl.Callback] = [MetadataCallback()]
        self._metadata_callback = None

        # Store initial hyperparameters in the pl_module and the initial shapes/model name in metadata logger
        self.save_hyperparameters({"args": args, **kwargs}, ignore=["base_model"])

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
    def callbacks(self) -> List[pl.Callback]:
        """Gets the callbacks"""
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: List[pl.Callback]):
        """Sets the callbacks + the default metadata callback"""
        found = False
        for callback in callbacks:
            if isinstance(callback, MetadataCallback):
                metadata_callback = callback
                found = True
        if not found:
            logger.info("Metadata callback added to the model's callbacks")
            metadata_callback = MetadataCallback()
            callbacks.append(metadata_callback)
        self._metadata_callback = metadata_callback
        self._callbacks = callbacks

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
        """Get the criterion function loss(y, gt) -> backpropagable tensor"""
        return self._criterion_fn

    @criterion_fn.setter
    def criterion_fn(self, criterion_fn: Callable[[tr.Tensor, tr.Tensor], tr.Tensor]):
        assert isinstance(criterion_fn, Callable), f"Got '{criterion_fn}'"
        logger.debug(f"Setting criterion to '{criterion_fn}'")
        self._criterion_fn = criterion_fn
        if len(self.metrics) == 0:
            self.metrics = {}

    @property
    def metrics(self) -> Dict[str, CoreMetric]:
        """Gets the list of metric names"""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Dict[str, Tuple[Callable, str]]):
        if len(self._metrics) != 0:
            logger.info(f"Settings metrics to: {list(metrics.keys())}")
        self._metrics = {}

        for metric_name, metric_fn in metrics.items():
            # Our metrics can be a CoreMetric already, a Tuple (callable, min/max) or just a Callable
            assert isinstance(metric_fn, (CoreMetric, Tuple, Callable)), \
                   f"Unknown metric type: '{type(metric_fn)}'. " \
                   "Expcted CoreMetric, Callable or (Callable, \"min\"/\"max\")."

            # If it is not a CoreMetric already (Tuple or Callable), we convert it to CallableCoreMetric
            if isinstance(metric_fn, Callable) and not isinstance(metric_fn, CoreMetric):
                metric_fn = (metric_fn, "min")

            if isinstance(metric_fn, Tuple):
                logger.debug(f"Metric '{metric_name}' is a callable. Converting to CallableCoreMetric.")
                metric_fn, min_or_max = metric_fn
                assert min_or_max in ("min", "max"), f"Got '{min_or_max}'"
                metric_fn = CallableCoreMetric(metric_fn, higher_is_better=(min_or_max == "max"))

            self._metrics[metric_name] = metric_fn
        self._metrics["loss"] = CallableCoreMetric(self.criterion_fn, higher_is_better=False, requires_grad=True)
        self.logged_metrics = list(self._metrics.keys())

    @property
    def logged_metrics(self) -> List[str]:
        """Return the list of logged metrics out of all the defined ones"""
        return self._logged_metrics

    @logged_metrics.setter
    def logged_metrics(self, logged_metrics: List[str]):
        assert "loss" in logged_metrics, "When manually settings logged_metrics, " \
                                         f"loss must be present. Got: {logged_metrics}"
        logger.debug(f"Setting the logged metrics to {logged_metrics}") if len(logged_metrics) > 1 else ()
        diff = set(logged_metrics).difference(self._metrics.keys())
        assert len(diff) == 0, f"Metrics {diff} are not in set metrics: {self._metrics.keys()}"
        self._logged_metrics = logged_metrics

    # Overrides on top of the standard pytorch lightning module
    @property
    def metadata_callback(self):
        """Returns the metadata callback of this module"""
        if self._metadata_callback is None:
            for callback in self.trainer.callbacks:
                if isinstance(callback, MetadataCallback):
                    self._metadata_callback = callback
        return self._metadata_callback

    @overrides
    def on_fit_start(self) -> None:
        if hasattr(self.base_model, "criterion_fn") and self.base_model.criterion_fn is not None:
            logger.info("Base model has criterion_fn attribute. Using these by default")
            self.criterion_fn = self.base_model.criterion_fn
        if hasattr(self.base_model, "metrics") and self.base_model.metrics is not None:
            assert hasattr(self.base_model, "criterion_fn"), "For now, we need both or just criterion_fn to be set"
            logger.info("Base model has metrics attribute. Using these by default")
            self.metrics = self.base_model.metrics

        if len(self.metrics) == 0:
            self.metrics = {}

        # We need to remove the val_ metrics, because if .fit() is called twice, this will create too many copies
        new_metrics = {metric_name: metric_fn for metric_name, metric_fn in self.metrics.items()
                       if not metric_name.startswith("val_")}
        if self.trainer.enable_validation:
            # If we use a validation set, clone all the metrics, so that the statistics don't intefere with each other
            val_metrics = CoreModule._clone_all_metrics_with_prefix(new_metrics, prefix="val_")
            new_metrics = {**new_metrics, **val_metrics}
        self._metrics = new_metrics
        return super().on_fit_start()

    @overrides
    def on_test_start(self) -> None:
        return super().on_test_start()

    @overrides
    def training_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs) -> Union[tr.Tensor, Dict[str, Any]]:
        """Training step: returns batch training loss and metrics."""
        res = self._generic_batch_step(train_batch)
        return res

    @overrides
    def validation_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs):
        """Validation step: returns batch validation loss and metrics."""
        return self._generic_batch_step(train_batch, prefix="val_")

    @overrides
    def test_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs):
        """Testing step: returns batch test loss and metrics. No prefix."""
        return self._generic_batch_step(train_batch)

    @overrides
    def configure_callbacks(self) -> Union[Sequence[pl.Callback], pl.Callback]:
        return self.callbacks

    @overrides
    def training_epoch_end(self, outputs):
        """Computes epoch average train loss and metrics for logging."""
        # If validation is enabled (for train loops), add "val_" metrics for all logged metrics.
        metrics_to_log = self.logged_metrics
        if self.trainer.enable_validation:
            val_logged_metrics = [f"val_{metric_name}" for metric_name in self.logged_metrics]
            metrics_to_log = [*self.logged_metrics, *val_logged_metrics]
        self._run_and_log_metrics_at_epoch_end(metrics_to_log)

    @overrides
    def test_epoch_end(self, outputs):
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
        num_params = len(tuple(self.parameters()))
        if num_params == 0:
            return
        for layer in self.base_model.children():
            if CoreModule(layer).num_params == 0:
                continue
            if not hasattr(layer, "reset_parameters"):
                logger.debug(f"Layer {layer} has params, but no reset_parameters() method. Trying recursively")
                layer = CoreModule(layer)
            layer.reset_parameters()

    def load_state_from_path(self, path: str):
        """Loads the state dict from a path"""
        # if path is remote (gcs) download checkpoint to a temp dir
        logger.info(f"Loading weights and hyperparameters from '{path}'")
        # if str(path).startswith("gs"):
        #     path = update_gcs_path(path)
        ckpt_data = tr.load(path)
        self.load_state_dict(ckpt_data["state_dict"])
        self.save_hyperparameters(ckpt_data["hyper_parameters"])

    def state_dict(self):
        return self.base_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        return self.base_model.load_state_dict(state_dict, strict)

    def setup_module_for_train(self, train_cfg: Dict):
        """Given a train cfg, prepare this module for training, by setting the required information."""
        TrainSetup(self, train_cfg).setup()

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
            metric_fn: CoreMetric = self.metrics[prefixed_metric_name]
            # Call the metric and update its state
            metric_output: tr.Tensor = metric_fn.forward(y, gt)
            metric_fn.batch_update(metric_output)
            outputs[prefixed_metric_name] = metric_output
            # Log all the numeric batch metrics. Don't show on pbar.
            # Don't use any self.log() here. We don't really care about intermediate batch results, only epoch results,
            #  which are handled down.
        return outputs

    def _run_and_log_metrics_at_epoch_end(self, metrics_to_log: List[str]):
        """Runs and logs a given list of logged metrics. Assume they all exist in self.metrics"""
        for metric_name in metrics_to_log:
            metric_fn: CoreMetric = self.metrics[metric_name]
            # Get the metric's epoch result
            metric_epoch_result = metric_fn.epoch_result()
            # Log the metric at the end of the epoch. Only log on pbar the val_loss, loss is tracked by default
            prog_bar = metric_name == "val_loss"

            value_reduced = metric_fn.epoch_result_reduced(metric_epoch_result)
            if value_reduced is not None:
                super().log(metric_name, value_reduced, prog_bar=prog_bar, on_epoch=True)
            # Call the metadata callback for the full result, since it can handle any sort of metrics
            self.metadata_callback.save_epoch_metric(metric_name, metric_epoch_result, self.trainer.current_epoch)
            # Reset the metric after storing this epoch's value
            metric_fn.reset()

    @staticmethod
    def _clone_all_metrics_with_prefix(metrics: Dict[str, CoreMetric], prefix: str) -> Dict[str, CoreMetric]:
        """Clones all the given metrics, by ading a prefix to the name"""
        assert len(prefix) > 0 and prefix[-1] == "_", "Prefix must be of format 'XXX_'"
        new_metrics = {}
        for metric_name, metric_fn in metrics.items():
            if metric_name.startswith(prefix):
                logger.warning(f"This may be a bug, since metric '{metric_name}' already has prefix '{prefix}'")
                continue
            new_metrics[f"{prefix}{metric_name}"] = deepcopy(metric_fn)
        return new_metrics
