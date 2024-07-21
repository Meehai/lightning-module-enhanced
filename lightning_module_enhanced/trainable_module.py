"""
TrainableModule is a standalone mixin class used to add the necessary properties to train a model:
    criterion_fn, metrics, optimizer, scheduler & callbacks.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Union
from torch import optim, nn
import torch as tr
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from .metrics import CoreMetric, CallableCoreMetric
from .callbacks import MetadataCallback, MetricsHistory
from .logger import lme_logger as logger
from .utils import parsed_str_type, make_list

OptimizerType = Union[optim.Optimizer, List[optim.Optimizer]]
_SchedulerType = Dict[str, Union[optim.lr_scheduler.LRScheduler, str]]
SchedulerType = Union[_SchedulerType, List[_SchedulerType], None]
CriterionFnType = CallableCoreMetric


class TrainableModule(nn.Module, ABC):
    """
    Trainable module abstract class
    Defines the necessary and optional attributes required to train a LME.
    The necessary attributes are: optimizer & criterion.
    The optional attributes are: scheduler, metrics & callbacks.
    """

    @property
    @abstractmethod
    def callbacks(self) -> list[pl.Callback]:
        """The callbacks"""

    @property
    @abstractmethod
    def criterion_fn(self) -> Callable:
        """Get the criterion function loss(y, gt) -> backpropagable tensor"""

    @property
    @abstractmethod
    def metrics(self) -> dict[str, CoreMetric]:
        """Gets the list of metric names"""

    @property
    @abstractmethod
    def optimizer(self) -> OptimizerType:
        """Returns the optimizer"""

    @property
    @abstractmethod
    def scheduler(self) -> SchedulerType:
        """Returns the scheduler dict"""

    @property
    @abstractmethod
    def checkpoint_monitors(self) -> list[str]:
        """A subset of the metrics that are used for model checkpointing"""


# pylint: disable=abstract-method
class TrainableModuleMixin(TrainableModule):
    """TrainableModule mixin class implementation"""

    def __init__(self):
        super().__init__()
        self.metadata_callback = MetadataCallback()
        self.metrics_history = MetricsHistory()
        self._optimizer: OptimizerType = None
        self._scheduler: SchedulerType = None
        self._criterion_fn: CriterionFnType = None
        self._metrics: dict[str, CoreMetric] = None
        # The default callbacks that are singletons. Cannot be overwritten and only one instance must exist.
        self._callbacks: list[pl.Callback] = []
        self._checkpoint_monitors = None
        self._lme_reserved_properties = ["criterion_fn", "optimizer", "scheduler", "metrics", "callbacks"]

    @property
    def default_callbacks(self):
        """Returns the list of default callbacks"""
        return [self.metadata_callback, self.metrics_history]

    # Required for training
    @property
    def criterion_fn(self) -> CriterionFnType:
        """Get the criterion function loss(y, gt) -> backpropagable tensor"""
        if self._criterion_fn is None:
            return CallableCoreMetric(TrainableModuleMixin._default_criterion_fn, higher_is_better=False)
        return self._criterion_fn

    @criterion_fn.setter
    def criterion_fn(self, criterion_fn: Callable[[tr.Tensor, tr.Tensor], tr.Tensor]):
        assert isinstance(criterion_fn, Callable), f"Got '{criterion_fn}'"
        logger.info(f"Setting criterion to '{criterion_fn}'")
        self._criterion_fn = CallableCoreMetric(criterion_fn, higher_is_better=False, requires_grad=True)

    @staticmethod
    def _default_criterion_fn(y: tr.Tensor, gt: tr.Tensor):
        raise NotImplementedError("No criterion fn was implemented. Use model.criterion_fn=XXX or a different "
                                  "model.model_algorithm that includes a loss function")

    @property
    def optimizer(self) -> OptimizerType:
        """Returns the optimizer"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: OptimizerType | optim.Optimizer):
        assert isinstance(optimizer, (optim.Optimizer, list)), type(optimizer)
        optimizer_types = (type(x) for x in make_list(optimizer))
        assert all(lambda x: isinstance(x, optim.optimizer) for x in optimizer_types), optimizer_types
        logger.info(f"Set the optimizer to: {', '.join(parsed_str_type(o) for o in make_list(optimizer))}")
        self._optimizer = optimizer

    def _model_ckpt_cbs(self) -> list[pl.Callback]:
        prefix = "val_" if self.trainer.enable_validation else ""
        res = []
        for i, monitor in enumerate(self.checkpoint_monitors):
            # Lightning requires ValueError here, though KeyError would be more appropriate
            if monitor != "loss" and monitor not in self.metrics:
                raise ValueError(f"Checkpoint monitor '{monitor}' not in metrics: {self.metrics.keys()}")

            metric_fn = self.metrics[monitor] if monitor != "loss" else self.criterion_fn
            mode = "max" if metric_fn.higher_is_better else "min"
            ckpt_monitor = f"{prefix}{monitor}"
            filename = "{epoch}-{" + prefix + monitor + ":.2f}"
            # note: save_last=True for len(model_ckpt_cbs)==0 only (i.e. first monitor)
            res.append(ModelCheckpoint(monitor=ckpt_monitor, mode=mode, filename=filename,
                                       save_last=(i == 0), save_on_train_epoch_end=True))
        return res

    @property
    def callbacks(self) -> list[pl.Callback]:
        """Gets the callbacks"""

        # trainer not attached yet, so no model checkpoints are needed.
        try:
            _ = self.trainer
        except RuntimeError:
            return [*self.default_callbacks, *self._callbacks]

        trainer_cbs = [callback for callback in self.trainer.callbacks
                       if isinstance(callback, ModelCheckpoint) and callback.monitor is not None]
        if len(trainer_cbs) > 0:
            logger.debug("ModelCheckpoint callbacks were provided in the Trainer. Not using the checkpoint_monitors!")
            return [*self.default_callbacks, *self._callbacks, *trainer_cbs]
        return [*self.default_callbacks, *self._callbacks, *self._model_ckpt_cbs()]

    @callbacks.setter
    def callbacks(self, callbacks: list[pl.Callback]):
        """Sets the callbacks + the default metadata callback"""
        res = []
        for callback in callbacks:
            if callback in self.default_callbacks:
                continue
            res.append(callback)
        new_res = list(set(res))

        if len(res) != len(new_res):
            logger.debug("Duplicates were found in callbacks and removed")

        for callback in new_res:
            for default_callback in self.default_callbacks:
                assert not isinstance(callback, type(default_callback)), f"{callbacks} vs {default_callback}"

        self._callbacks = new_res
        logger.info(f"Set {len(self.callbacks)} callbacks to the module")

    @property
    def metrics(self) -> dict[str, CoreMetric]:
        """Gets the list of metric names"""
        if self._metrics is None:
            logger.debug("No metrics were set. Returning empty dict")
            return {}
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: dict[str, tuple[Callable, str]]):
        if self._metrics is not None:
            logger.debug(f"Overwriting existing metrics {list(self.metrics.keys())} to {list(metrics.keys())}")
        self._metrics = {}

        assert "loss" not in metrics.keys(), f"'loss' is not a valid metric name {list(metrics.keys())}"
        for metric_name, metric_fn in metrics.items():
            # Our metrics can be a CoreMetric already, a tuple (callable, min/max) or just a Callable
            assert isinstance(metric_fn, (CoreMetric, tuple)), f"Unknown metric type: '{type(metric_fn)}'. Expected " \
                                                               'CoreMetric, or a tuple of form (Callable, "min"/"max").'
            assert not metric_name.startswith("val_"), "metrics cannot start with val_"

            # If we get a tuple, we will assume it's a 2 piece: a callable function (or class) and a
            if isinstance(metric_fn, tuple):
                logger.debug(f"Metric '{metric_name}' is a callable. Converting to CallableCoreMetric.")
                metric_fn, min_or_max = metric_fn
                assert not isinstance(metric_fn, CoreMetric), "Cannot use tuple syntax with metric instances"
                assert isinstance(metric_fn, Callable), "Cannot use the tuple syntax with non-callables for metrics"
                assert min_or_max in ("min", "max"), f"Got '{min_or_max}', expected 'min' or 'max'"
                metric_fn = CallableCoreMetric(metric_fn, higher_is_better=(min_or_max == "max"), requires_grad=False)

            self._metrics[metric_name] = metric_fn
        if len(self.metrics) > 0:
            logger.info(f"Set module metrics: {list(self.metrics.keys())} ({len(self.metrics)})")

    @property
    def scheduler(self) -> SchedulerType:
        """Returns the scheduler dict"""
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler: SchedulerType | dict):
        for sch in make_list(scheduler):
            assert isinstance(sch, dict) and "scheduler" in sch.keys(), \
                'Use model.scheduler={"scheduler": sch, ["monitor": ...]} (or a list of dicts if >1 optimizers)'
            assert hasattr(sch["scheduler"], "step"), f"Scheduler {sch} does not have a step method"
        logger.info(f"Set the scheduler to {scheduler}")
        assert len(make_list(self.optimizer)) == 1, f"Can have scheduler only with 1 optimizer: {self.optimizer}" # TODO
        self._scheduler = scheduler

    @property
    def checkpoint_monitors(self) -> list[str]:
        if self._checkpoint_monitors is None:
            logger.debug("checkpoint_monitors not set. Defaulting to 'loss'")
            self.checkpoint_monitors = ["loss"]

        for monitor in self._checkpoint_monitors:
            if monitor != "loss" and monitor not in self.metrics:
                raise ValueError(f"Monitor '{monitor}' not in metrics: '{self.metrics}'")
        return self._checkpoint_monitors

    @checkpoint_monitors.setter
    def checkpoint_monitors(self, checkpoint_monitors: list[str]) -> list[str]:
        assert "loss" in checkpoint_monitors, f"'loss' must be in checkpoint monitors. Got: {checkpoint_monitors}"
        for monitor in checkpoint_monitors:
            if monitor != "loss" and monitor not in self.metrics:
                raise ValueError(f"Provided monitor: '{monitor}' is not in the metrics: {self.metrics}")
        self._checkpoint_monitors = checkpoint_monitors
        logger.debug(f"Set the checkpoint monitors to: {self._checkpoint_monitors}")
