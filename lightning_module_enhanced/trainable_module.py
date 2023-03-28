"""
TrainableModule is a standalone mixin class used to add the necessary properties to train a model:
    criterion_fn, metrics, optimizer, scheduler & callbacks. It is comaptible with TrainSetup class.
"""
from abc import ABC, abstractmethod
from typing import Dict, Union, Any, Callable, List, Tuple, Type
from torch import optim, nn
import torch as tr
import pytorch_lightning as pl
from .metrics import CoreMetric, CallableCoreMetric
from .callbacks import MetadataCallback
from .logger import logger

OptimizerType = Union[optim.Optimizer, List[optim.Optimizer]]
SchedulerType = Union[Dict, List[Dict]]


class TrainableModule(nn.Module, ABC):
    """
    Trainable module abstract class
    Defines the necessary and optional attributes required to train a LME.
    The necessary attributes are: optimizer & criterion.
    The optional attributes are: scheduler, metrics & callbacks.
    """

    @property
    @abstractmethod
    def callbacks(self) -> List[pl.Callback]:
        """The callbacks"""

    @property
    @abstractmethod
    def criterion_fn(self) -> Callable:
        """Get the criterion function loss(y, gt) -> backpropagable tensor"""

    @property
    @abstractmethod
    def metrics(self) -> Dict[str, CoreMetric]:
        """Gets the list of metric names"""

    @property
    @abstractmethod
    def optimizer(self) -> OptimizerType:
        """Returns the optimizer"""

    @property
    @abstractmethod
    def scheduler_dict(self) -> Dict:
        """Returns the scheduler dict"""


# pylint: disable=abstract-method
class TrainableModuleMixin(TrainableModule):
    """TrainableModule mixin class implementation"""

    def __init__(self):
        super().__init__()
        self._optimizer: optim.Optimizer = None
        self._scheduler_dict: Dict[str, Union[optim.lr_scheduler._LRScheduler, Any]] = None
        self._criterion_fn: Callable[[tr.Tensor, tr.Tensor], tr.Tensor] = None
        self._metrics: Dict[str, CoreMetric] = {}
        # The default callbacks that are singletons. Cannot be overwritten and only one instance must exist.
        self._default_callbacks = [MetadataCallback()]
        self._callbacks: List[pl.Callback] = []
        self._metadata_callback = self._default_callbacks[0]

    @property
    def metadata_callback(self):
        """Returns the metadata callback of this module"""
        return self._metadata_callback

    # Required for training
    @property
    def criterion_fn(self) -> Callable:
        """Get the criterion function loss(y, gt) -> backpropagable tensor"""
        if isinstance(self.base_model, TrainableModule):
            return self.base_model.criterion_fn
        return self._criterion_fn

    @criterion_fn.setter
    def criterion_fn(self, criterion_fn: Callable[[tr.Tensor, tr.Tensor], tr.Tensor]):
        assert not isinstance(self.base_model, TrainableModule)
        assert isinstance(criterion_fn, Callable), f"Got '{criterion_fn}'"
        logger.debug(f"Setting criterion to '{criterion_fn}'")
        self._criterion_fn = CallableCoreMetric(criterion_fn, higher_is_better=False, requires_grad=True)
        self.metrics = {**self.metrics, "loss": self.criterion_fn}

    @property
    def optimizer(self) -> OptimizerType:
        """Returns the optimizer"""
        if isinstance(self.base_model, TrainableModule):
            return self.base_model.optimizer

        res = self._optimizer
        if res is not None and len(res) == 1:
            return res[0]
        return res

    @optimizer.setter
    def optimizer(self, optimizer: OptimizerType):
        assert not isinstance(self.base_model, TrainableModule)
        if isinstance(optimizer, optim.Optimizer):
            optimizer = [optimizer]
        for o in optimizer:
            assert isinstance(o, optim.Optimizer), f"Got {o} (type {type(o)})"
        logger.debug(f"Set the optimizer to {optimizer}")
        self._optimizer = optimizer

    @property
    def callbacks(self) -> List[pl.Callback]:
        """Gets the callbacks"""
        if isinstance(self.base_model, TrainableModule):
            return [*self._default_callbacks, *self.base_model.callbacks]
        return [*self._default_callbacks, *self._callbacks]

    @callbacks.setter
    def callbacks(self, callbacks: List[pl.Callback]):
        """Sets the callbacks + the default metadata callback"""
        assert not isinstance(self.base_model, TrainableModule)
        res = []
        for callback in callbacks:
            if callback in self._default_callbacks:
                continue
            res.append(callback)
        new_res = list(set(res))

        if len(res) != len(new_res):
            logger.warning("Duplicates were found in callbacks and removed")

        for callback in new_res:
            for default_callback in self._default_callbacks:
                assert not isinstance(callback, type(default_callback)), f"{callbacks} vs {default_callback}"

        self._callbacks = new_res

    @property
    def metrics(self) -> Dict[str, CoreMetric]:
        """Gets the list of metric names"""
        if isinstance(self.base_model, TrainableModule):
            return self.base_model.metrics
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Dict[str, Tuple[Callable, str]]):
        assert not isinstance(self.base_model, TrainableModule)
        if len(self._metrics) != 0:
            logger.debug(f"Overwriting existing metrics {list(self.metrics.keys())} to {list(metrics.keys())}")
        self._metrics = {}

        for metric_name, metric_fn in metrics.items():
            # Our metrics can be a CoreMetric already, a Tuple (callable, min/max) or just a Callable
            assert isinstance(metric_fn, (CoreMetric, Tuple)), (
                f"Unknown metric type: '{type(metric_fn)}'. "
                'Expcted CoreMetric, or a tuple of form (Callable, "min"/"max").'
            )
            assert not metric_name.startswith("val_"), "metrics cannot start with val_"
            if metric_name == "loss":
                assert isinstance(metric_fn, CallableCoreMetric) and metric_fn.requires_grad is True

            # If we get a tuple, we will assume it's a 2 piece: a callable function (or class) and a
            if isinstance(metric_fn, Tuple):
                logger.debug(f"Metric '{metric_name}' is a callable. Converting to CallableCoreMetric.")
                metric_fn, min_or_max = metric_fn
                assert not isinstance(metric_fn, CoreMetric), "Cannot use tuple syntax with metric instances"
                assert isinstance(metric_fn, Callable), "Cannot use the tuple syntax with non-callables for metrics"
                assert min_or_max in (
                    "min",
                    "max",
                ), f"Got '{min_or_max}', expected 'min' or 'max'"
                metric_fn = CallableCoreMetric(
                    metric_fn,
                    higher_is_better=(min_or_max == "max"),
                    requires_grad=False,
                )
            self._metrics[metric_name] = metric_fn
        if self.criterion_fn is not None:
            self._metrics["loss"] = self.criterion_fn
        logger.debug(f"Set module metrics: {list(self.metrics.keys())} ({len(self.metrics)})")

    @property
    def optimizer_type(self) -> Type[optim.Optimizer]:
        """Returns the optimizer type, instead of the optimizer itself"""
        return type(self.optimizer)

    @property
    def scheduler_dict(self) -> SchedulerType:
        """Returns the scheduler dict"""
        if isinstance(self.base_model, TrainableModule):
            return self.base_model.scheduler_dict
        res = self._scheduler_dict
        if res is not None and len(res) == 1:
            return res[0]
        return res

    @scheduler_dict.setter
    def scheduler_dict(self, scheduler_dict: SchedulerType):
        assert not isinstance(self.base_model, TrainableModule)
        assert isinstance(scheduler_dict, (dict, list))
        if isinstance(scheduler_dict, Dict):
            scheduler_dict = [scheduler_dict]
        for i in range(len(scheduler_dict)):
            assert "scheduler" in scheduler_dict[i]
            assert hasattr(scheduler_dict[i]["scheduler"], "step"), "Scheduler does not have a step method"
        logger.debug(f"Set the scheduler to {scheduler_dict}")
        self._scheduler_dict = scheduler_dict
