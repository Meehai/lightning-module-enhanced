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
from .metrics import CoreMetric, CallableCoreMetric, StubMetric
from .callbacks import MetadataCallback
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
        return [self.metadata_callback]

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
        prefix = "val_" if (self._trainer is not None and self.trainer.enable_validation) else ""
        res = []
        for i, monitor in enumerate(self.checkpoint_monitors):
            # note: save_last=True for len(model_ckpt_cbs)==0 only (i.e. first monitor)
            mode = self.metrics[monitor].mode if monitor != "loss" else "min"
            filename = "{epoch}-{" + prefix + monitor + ":.3f}"
            res.append(ModelCheckpoint(monitor=f"{prefix}{monitor}", filename=filename, save_last=(i == 0),
                                       save_on_train_epoch_end=True, mode=mode))
        return res

    @property
    def callbacks(self) -> list[pl.Callback]:
        """Gets the callbacks"""
        if self._trainer is None: # trainer not attached yet, so no model checkpoints are needed.
            return [*self.default_callbacks, *self._callbacks]
        ckpt_cbs = self._model_ckpt_cbs()
        if self._trainer is not None:
            trainer_ckpt_cbs = [callback for callback in self.trainer.callbacks
                               if isinstance(callback, ModelCheckpoint) and callback.monitor is not None]
            if len(trainer_ckpt_cbs) > 0:
                logger.debug("ModelCheckpoint callbacks provided in the Trainer. Not using the checkpoint_monitors!")
                ckpt_cbs = trainer_ckpt_cbs
        return [*self.default_callbacks, *self._callbacks, *ckpt_cbs]

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
            logger.debug2("No metrics were set. Returning empty dict")
            return {}
        try: # for active runs
            res = self._active_run_metrics.get(self._prefix_from_trainer(), self._metrics)
            return {k: v for k, v in res.items() if k != "loss"}
        except Exception:
            return self._metrics

    @metrics.setter
    def metrics(self, metrics: dict[str, CoreMetric | tuple[Callable | None, str]]):
        assert all(isinstance(k, str) for k in metrics.keys()), metrics
        assert all(isinstance(v, (tuple, CoreMetric)) for v in metrics.values()), metrics
        assert "loss" not in metrics.keys(), f"'loss' is not a valid metric name {list(metrics.keys())}"
        assert not any(k.startswith("val_") for k in metrics.keys()), "metrics cannot start with val_"
        if self._metrics is not None:
            logger.debug(f"Overwriting existing metrics {list(self.metrics.keys())} to {list(metrics.keys())}")

        res = {}
        for metric_name, metric_fn in metrics.items():
            # Our metrics can be a CoreMetric already, a tuple (callable, min/max) or just a Callable
            if isinstance(metric_fn, tuple):
                assert len(metric_fn) == 2, f"Expected (None|Callable, 'min'/'max', got: {metric_fn}"
                assert metric_fn[1] in ("min", "max"), f"Expected 'min'/'max', got: {metric_fn[1]}"
                if metric_fn[0] is None:
                    metric_fn = StubMetric(metric_fn[1] == "max")
                    logger.debug(f"Metric: '{metric_name}'. Making a StubMetric. Direction: {metric_fn.mode=}")
                else:
                    metric_fn = CallableCoreMetric(metric_fn[0], higher_is_better=metric_fn[1] == "max")
                    logger.debug(f"Metric: '{metric_name}'. Making a CallableCoreMetric. Direction: {metric_fn.mode=}")
            assert isinstance(metric_fn, CoreMetric), f"At this point we should have only CoreMetrics. Got: {metric_fn}"
            res[metric_name] = metric_fn

        logger.info(f"Set module metrics: {list(res.keys())} ({len(res)})" if len(res) > 0 else "Unset the metrics")
        self._metrics = res

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
        return self._checkpoint_monitors

    @checkpoint_monitors.setter
    def checkpoint_monitors(self, checkpoint_monitors: list[str]) -> list[str]:
        assert "loss" in checkpoint_monitors, f"'loss' must be in checkpoint monitors. Got: {checkpoint_monitors}"
        cm_wo_loss = set(x for x in checkpoint_monitors if x != "loss")
        assert all(x in self.metrics for x in cm_wo_loss), f"Not in metrics: {(cm_wo_loss - self.metrics.keys())}"
        self._checkpoint_monitors = checkpoint_monitors
        logger.debug(f"Set the checkpoint monitors to: {self._checkpoint_monitors}")
