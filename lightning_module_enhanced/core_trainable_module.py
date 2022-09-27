"""
CoreTrainableModule is a standalone mixin class used to add the necessary properties to train a model:
    criterion_fn, metrics, optimizer, scheduler & callbacks. It is comaptible with TrainSetup class.
"""
from typing import Dict, Union, Any, Callable, List, Tuple
from torch import optim, nn
import torch as tr
import pytorch_lightning as pl
from .metrics import CoreMetric, CallableCoreMetric
from .callbacks import MetadataCallback
from .logger import logger
from .train_setup import TrainSetup

class CoreTrainableModule(nn.Module):
    """CoreTrainableModule mixin class implementation"""
    def __init__(self):
        super().__init__()
        self._optimizer: optim.Optimizer = None
        self._scheduler_dict: Dict[str, Union[optim.lr_scheduler._LRScheduler, Any]] = None
        self._criterion_fn: Callable[[tr.Tensor, tr.Tensor], tr.Tensor] = None
        self._metrics: Dict[str, CoreMetric] = {}
        # The unique instance of metadata callback. Cannot over overwriten.
        self._metadata_callback = MetadataCallback()
        self._callbacks: List[pl.Callback] = [self._metadata_callback]

    @property
    def metadata_callback(self):
        """Returns the metadata callback of this module"""
        return self._metadata_callback

    @property
    def callbacks(self) -> List[pl.Callback]:
        """Gets the callbacks"""
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: List[pl.Callback]):
        """Sets the callbacks + the default metadata callback"""
        for callback in callbacks:
            assert not isinstance(callback, MetadataCallback), "Metadata callback cannot be overwriten."
        callbacks.append(self._metadata_callback)
        self._callbacks = callbacks

    @property
    def criterion_fn(self) -> Callable:
        """Get the criterion function loss(y, gt) -> backpropagable tensor"""
        return self._criterion_fn

    @criterion_fn.setter
    def criterion_fn(self, criterion_fn: Callable[[tr.Tensor, tr.Tensor], tr.Tensor]):
        assert isinstance(criterion_fn, Callable), f"Got '{criterion_fn}'"
        logger.debug(f"Setting criterion to '{criterion_fn}'")
        self._criterion_fn = CallableCoreMetric(criterion_fn, higher_is_better=False, requires_grad=True)
        self.metrics = {**self.metrics, "loss": self.criterion_fn}

    @property
    def metrics(self) -> Dict[str, CoreMetric]:
        """Gets the list of metric names"""
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Dict[str, Tuple[Callable, str]]):
        if len(self._metrics) != 0:
            logger.info(f"Overwriting existing metrics {list(self.metrics.keys())} to {list(metrics.keys())}")
        self._metrics = {}

        for metric_name, metric_fn in metrics.items():
            # Our metrics can be a CoreMetric already, a Tuple (callable, min/max) or just a Callable
            assert isinstance(metric_fn, (CoreMetric, Tuple, Callable)), \
                   f"Unknown metric type: '{type(metric_fn)}'. " \
                   "Expcted CoreMetric, Callable or (Callable, \"min\"/\"max\")."
            assert not metric_name.startswith("val_"), "metrics cannot start with val_"
            if metric_name == "loss":
                assert isinstance(metric_fn, CallableCoreMetric) and metric_fn.requires_grad is True

            # If it is not a CoreMetric already (Tuple or Callable), we convert it to CallableCoreMetric
            if isinstance(metric_fn, Callable) and not isinstance(metric_fn, CoreMetric):
                metric_fn = (metric_fn, "min")

            if isinstance(metric_fn, Tuple):
                logger.debug(f"Metric '{metric_name}' is a callable. Converting to CallableCoreMetric.")
                metric_fn, min_or_max = metric_fn
                assert min_or_max in ("min", "max"), f"Got '{min_or_max}'"
                metric_fn = CallableCoreMetric(metric_fn, higher_is_better=(min_or_max == "max"), requires_grad=False)
            self._metrics[metric_name] = metric_fn
        if self.criterion_fn is not None:
            self._metrics["loss"] = self.criterion_fn
        logger.info(f"Set module metrics: {list(self.metrics.keys())} ({len(self.metrics)})")

    def setup_module_for_train(self, train_cfg: Dict):
        """Given a train cfg, prepare this module for training, by setting the required information."""
        TrainSetup(self, train_cfg).setup()
