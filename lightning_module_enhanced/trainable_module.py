"""
TrainableModule is a standalone mixin class used to add the necessary properties to train a model:
    criterion_fn, metrics, optimizer, scheduler & callbacks. It is comaptible with TrainSetup class.
"""
from abc import ABC, abstractmethod
from typing import Dict, Union, Any, Callable, List, Tuple, Type
from torch import optim, nn
import torch as tr
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from .metrics import CoreMetric, CallableCoreMetric
from .callbacks import MetadataCallback
from .logger import logger
from .utils import parsed_str_type

OptimizerType = optim.Optimizer | list[optim.Optimizer]
SchedulerType = dict | list[dict]
CriterionFnType = Callable[[tr.Tensor, tr.Tensor], tr.Tensor]


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

    @property
    @abstractmethod
    def checkpoint_monitors(self) -> List[str]:
        """A subset of the metrics that are used for model checkpointing"""


# pylint: disable=abstract-method
class TrainableModuleMixin(TrainableModule):
    """TrainableModule mixin class implementation"""

    def __init__(self):
        super().__init__()
        self._optimizer: optim.Optimizer = None
        self._scheduler_dict: Dict[str, Union[optim.lr_scheduler._LRScheduler, Any]] = None
        self._criterion_fn: CriterionFnType = None
        self._metrics: Dict[str, CoreMetric] = None
        # The default callbacks that are singletons. Cannot be overwritten and only one instance must exist.
        self._callbacks: List[pl.Callback] = []
        self.metadata_callback = MetadataCallback()
        self._checkpoint_monitors = ["loss"]

    @property
    def default_callbacks(self):
        """Returns the list of default callbacks"""
        return [self.metadata_callback]

    # Required for training
    @property
    def criterion_fn(self) -> CriterionFnType:
        """Get the criterion function loss(y, gt) -> backpropagable tensor"""
        assert not isinstance(self.base_model, TrainableModule), "Cannot have nested Trainable Modules"
        if hasattr(self.base_model, "criterion_fn"):
            logger.warning("Base model has a .criterion_fn property. This may be confusing. It will not be used by LME")
        if self._criterion_fn is None:
            return CallableCoreMetric(TrainableModuleMixin._default_criterion_fn, higher_is_better=False)
        return self._criterion_fn

    @criterion_fn.setter
    def criterion_fn(self, criterion_fn: CriterionFnType):
        assert not isinstance(self.base_model, TrainableModule), "Cannot have nested Trainable Modules"
        assert isinstance(criterion_fn, Callable), f"Got '{criterion_fn}'"
        logger.debug(f"Setting criterion to '{criterion_fn}'")
        self._criterion_fn = CallableCoreMetric(criterion_fn, higher_is_better=False, requires_grad=True)
        self.metrics = {**self.metrics, "loss": self.criterion_fn}

    @staticmethod
    def _default_criterion_fn(y: tr.Tensor, gt: tr.Tensor):
        raise NotImplementedError("No criterion fn was implemented. Use model.criterion_fn=XXX or a different "
                                  "model.model_algorithm that includes a loss function")

    @property
    def optimizer(self) -> OptimizerType:
        """Returns the optimizer"""
        assert not isinstance(self.base_model, TrainableModule), "Cannot have nested Trainable Modules"
        if hasattr(self.base_model, "optimizer"):
            logger.warning("Base model has a .optimizer property. This may be confusing as it will not be used by LME")
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: OptimizerType):
        assert not isinstance(self.base_model, TrainableModule), "Cannot have nested Trainable Modules"
        assert isinstance(optimizer, (optim.Optimizer, List)), type(optimizer)
        if isinstance(optimizer, list):
            for o in optimizer:
                assert isinstance(o, optim.Optimizer), f"Got {o} (type {type(o)})"
            logger.debug(f"Set the optimizer to {[parsed_str_type(x) for x in optimizer]}")
        else:
            logger.debug(f"Set the optimizer to {parsed_str_type(optimizer)}")
        self._optimizer = optimizer

    @property
    def callbacks(self) -> List[pl.Callback]:
        """Gets the callbacks"""
        assert not isinstance(self.base_model, TrainableModule), "Cannot have nested Trainable Modules"
        if hasattr(self.base_model, "callbacks"):
            logger.warning("Base model has a .callbacks property. This may be confusing as it will not be used by LME")

        # trainer not attached yet, so no model checkpoints are needed.
        try:
            _ = self.trainer
        except RuntimeError:
            return [*self.default_callbacks, *self._callbacks]

        trainer_cbs = [callback for callback in self.trainer.callbacks
                       if isinstance(callback, ModelCheckpoint) and callback.monitor is not None]
        if len(trainer_cbs) > 0:
            logger.info("ModelCheckpoint callbacks were provided in the Trainer. Not using the checkpoint_monitors!")
            return [*self.default_callbacks, *self._callbacks, *trainer_cbs]

        prefix = "val_" if self.trainer.enable_validation else ""
        model_ckpt_cbs = []
        for monitor in self.checkpoint_monitors:
            # Lightning requires ValueError here, though KeyError would be more appropriate
            if monitor not in self.metrics:
                raise ValueError(f"Checkpoint monitor '{monitor}' not in metrics: {self.metrics.keys()}")

            mode = "max" if self.metrics[monitor].higher_is_better else "min"
            ckpt_monitor = f"{prefix}{monitor}"
            filename = "{epoch}-{" + prefix + monitor + ":.2f}"
            # note: save_last=True for len(model_ckpt_cbs)==0 only (i.e. first monitor)
            model_ckpt_cbs.append(ModelCheckpoint(monitor=ckpt_monitor, mode=mode, filename=filename,
                                                  save_last=(len(model_ckpt_cbs) == 0), save_on_train_epoch_end=True))

        return [*self.default_callbacks, *self._callbacks, *model_ckpt_cbs]

    @callbacks.setter
    def callbacks(self, callbacks: List[pl.Callback]):
        """Sets the callbacks + the default metadata callback"""
        assert not isinstance(self.base_model, TrainableModule), "Nested trainable modules"
        res = []
        for callback in callbacks:
            if callback in self.default_callbacks:
                continue
            res.append(callback)
        new_res = list(set(res))

        if len(res) != len(new_res):
            logger.warning("Duplicates were found in callbacks and removed")

        for callback in new_res:
            for default_callback in self.default_callbacks:
                assert not isinstance(callback, type(default_callback)), f"{callbacks} vs {default_callback}"

        self._callbacks = new_res

    @property
    def metrics(self) -> Dict[str, CoreMetric]:
        """Gets the list of metric names"""
        assert not isinstance(self.base_model, TrainableModule), "Cannot have nested Trainable Modules"
        if hasattr(self.base_model, "metrics"):
            logger.warning("Base model has a .metrics property. This may be confusing as it will not be used by LME")
        if self._metrics is None:
            return {"loss": CallableCoreMetric(self.criterion_fn, higher_is_better=False, requires_grad=True)}
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: Dict[str, Tuple[Callable, str]]):
        assert not isinstance(self.base_model, TrainableModule), "Nested trainable modules"
        if self._metrics is not None:
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
                assert min_or_max in ("min", "max"), f"Got '{min_or_max}', expected 'min' or 'max'"
                metric_fn = CallableCoreMetric(metric_fn, higher_is_better=(min_or_max == "max"), requires_grad=False)

            self._metrics[metric_name] = metric_fn
        if self.criterion_fn is not None:
            self._metrics["loss"] = self.criterion_fn
        logger.debug(f"Set module metrics: {list(self.metrics.keys())} ({len(self.metrics)})")

    @property
    def optimizer_type(self) -> Type[optim.Optimizer]:
        """Returns the optimizer type, instead of the optimizer itself"""
        return type(self.optimizer)

    # TODO: perhaps refactor this since we don't need to have a dict like this with automatic optimization turned off
    @property
    def scheduler_dict(self) -> SchedulerType:
        """Returns the scheduler dict"""
        assert not isinstance(self.base_model, TrainableModule), "Cannot have nested Trainable Modules"
        if hasattr(self.base_model, "scheduler_dict"):
            logger.warning("Base model has a .scheduler_dict property. This may be confusing. Will not be used by LME")
        res = self._scheduler_dict
        if res is not None and len(res) == 1:
            return res[0]
        return res

    @scheduler_dict.setter
    def scheduler_dict(self, scheduler_dict: SchedulerType):
        assert not isinstance(self.base_model, TrainableModule), "Nested trainable modules"
        assert isinstance(scheduler_dict, (dict, list)), scheduler_dict
        if isinstance(scheduler_dict, Dict):
            scheduler_dict = [scheduler_dict]
        for i in range(len(scheduler_dict)):
            assert "scheduler" in scheduler_dict[i]
            assert hasattr(scheduler_dict[i]["scheduler"], "step"), "Scheduler does not have a step method"
        logger.debug(f"Set the scheduler to {scheduler_dict}")
        self._scheduler_dict = scheduler_dict

    @property
    def checkpoint_monitors(self) -> List[str]:
        for monitor in self._checkpoint_monitors:
            if monitor not in self.metrics:
                raise ValueError(f"Monitor '{monitor}' not in metrics: '{self.metrics}'")
        return self._checkpoint_monitors

    @checkpoint_monitors.setter
    def checkpoint_monitors(self, checkpoint_monitors: List[str]) -> List[str]:
        assert "loss" in checkpoint_monitors, f"'loss' must be in checkpoint monitors. Got: {checkpoint_monitors}"
        for monitor in checkpoint_monitors:
            if monitor not in self.metrics:
                raise ValueError(f"Provided monitor: '{monitor}' is not in the metrics: {self.metrics}")
        self._checkpoint_monitors = checkpoint_monitors
        logger.debug(f"Set the checkpoint monitors to: {self._checkpoint_monitors}")
