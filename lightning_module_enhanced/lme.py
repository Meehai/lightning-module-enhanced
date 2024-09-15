"""Generic Pytorch Lightning module on top of a Pytorch nn.Module"""
from __future__ import annotations
from typing import Any, Sequence, Callable, NamedTuple, Dict, Union
from pathlib import Path
import shutil
from overrides import overrides
import torch as tr
import pytorch_lightning as pl
from lightning_fabric.utilities.seed import seed_everything
from lightning_fabric.utilities.exceptions import MisconfigurationException
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn, optim
from torchinfo import summary, ModelStatistics

from .trainable_module import TrainableModuleMixin, TrainableModule
from .active_run_mixin import ActiveRunMixin
from .logger import lme_logger as logger
from .utils import to_tensor, to_device, tr_detach_data, make_list

ModelAlgorithmOutput = NamedTuple("ModelAlgorithmOutput", y=tr.Tensor, metrics=Dict[str, tr.Tensor],
                                  x=Union[None, tr.Tensor, Dict[str, tr.Tensor]], gt=Union[None, tr.Tensor])

# pylint: disable=too-many-ancestors, arguments-differ, unused-argument, abstract-method
class LightningModuleEnhanced(TrainableModuleMixin, ActiveRunMixin, pl.LightningModule):
    """
        Generic pytorch-lightning module for ml models.

        Callbacks and metrics are called based on the order described here:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#hooks

        Takes a :class:`torch.nn.Module` as the underlying model and implements
        all the training/validation/testing logic in a unified way.

        Attributes:
        - base_model: The base :class:`torch.nn.Module`
        - model_algorithm: Handles the `y, metrics, x, gt = model.algorithm(batch)` part of the model
        - cache_results: Keeps the previous results during training so they can be accessed from various callbacks
        - device: The torch device this module is residing on
        - summary: A summary including parameters, layers and tensor shapes of this module
        - num_parames: The total number of parameters of this module
        - num_trainable_params: The number of trainable parameters of this module
        - has_trainer: Checks if the underlying LightningModule is attached to a pl.Trainer instance
        Attributes (inherited from TrainableModuleMixin):
        - optimizer: The optimizer used for training runs
        - scheduler: The oprimizer scheduler used for training runs, as well as the monitored metric
        - criterion_fn: The criterion function used for training runs
        - metrics: The dictionary (Name => CoreMetric) of all implemented metrics for this module
        - callbacks: The list of callbacks for this module
        - metadata_callback: The metadata logger for this run

        Args:
            base_model: The base :class:`torch.nn.Module`
    """
    def __init__(self, base_model: nn.Module, model_algorithm: Callable = None):
        assert isinstance(base_model, nn.Module), f"Expected a nn.Module, got {type(base_model)}"
        if isinstance(base_model, (LightningModuleEnhanced, TrainableModule)):
            raise ValueError("Cannot have nested LME modules. LME must extend only a basic torch nn.Module")
        super().__init__()
        super(nn.Module, self).__init__()
        for prop in self._lme_reserved_properties:
            assert not hasattr(base_model, prop), f"Cannot clash with {self._lme_reserved_properties=}: {prop}"
        self.base_model = base_model
        self.automatic_optimization = False
        self._summary: ModelStatistics | None = None
        self._model_algorithm = model_algorithm if model_algorithm is not None else type(self)._default_algorithm
        self.cache_result = None

    # Getters and setters for properties
    @property
    def device(self) -> tr.device:
        """Gets the device of the model, assuming all parameters are on the same device."""
        return next(self.base_model.parameters()).device if self.is_parametric_model else tr.device("cpu")

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
        if self._summary is None:
            self._summary = summary(self.base_model, verbose=0, depth=3)
        return self._summary

    @property
    def has_trainer(self) -> bool:
        """returns True is the underlying LightningModule has a trainer attached w/o RuntimeError"""
        return self._trainer is not None # pylint: disable=protected-access

    def trainable_params(self, value: bool):
        """Setter only property that makes all the parameters of this module to be trainable or untrainable"""
        logger.debug(f"Setting parameters of the model to '{value}'.")
        for param in self.base_model.parameters():
            param.requires_grad_(value)
        # Reset summary such that it is recomputted if necessary (like for counting num trainable params)
        self._summary = None
    trainable_params = property(None, trainable_params)

    @property
    def is_trainable_model(self) -> bool:
        """retursn true if the model has any trainable (grad=True) parameters"""
        return self.num_trainable_params > 0

    @property
    def is_parametric_model(self) -> bool:
        """returns true if the model has any trainable parameters (grad = True or False) at all"""
        return self.num_params > 0

    @property
    def model_algorithm(self) -> Callable:
        """The model algorithm, used at both training, validation, test and inference."""
        return self._model_algorithm

    @model_algorithm.setter
    def model_algorithm(self, value: Callable):
        assert isinstance(value, Callable), f"Expected a Callable, got {type(value)}"
        self._model_algorithm = value

    # Overrides on top of the standard pytorch lightning module

    @overrides
    def on_fit_start(self) -> None:
        self._setup_active_metrics()
        self._set_metrics_running_model()

    @overrides
    def on_train_start(self) -> None:
        self._copy_loaded_checkpoint_and_metrics() # must be done here so trainer.current_epoch is set

    @overrides
    def on_fit_end(self):
        self._unset_metrics_running_model()

    @overrides
    def on_test_start(self) -> None:
        self._setup_active_metrics()
        self._set_metrics_running_model()

    @overrides
    def on_test_end(self):
        self._unset_metrics_running_model()

    @overrides(check_signature=False)
    def training_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        """Training step: returns batch training loss and metrics."""
        # Warning: if not using lightning's self.optimizers(), and rather trying to user our self.optimizer, will
        # result in checkpoints not being saved.
        # After more digging it's because self.optimizers() doesn't return self.optimizer (the torch optimizer), but
        # rather lightning's warpper on top of it that can be used using other trainer strategies (ddp) and also
        # updates some internal states, like trainer.global_step.
        _opt = self.optimizers()
        opts: list[LightningOptimizer] = _opt if isinstance(_opt, list) else [_opt]
        for opt in opts:
            opt.optimizer.zero_grad()
        y, train_metrics, _, gt = self.model_algorithm(self, batch) # pylint: disable=not-callable
        self.cache_result = tr_detach_data(y)
        loss = train_metrics["loss"] if "loss" in train_metrics else self.criterion_fn(y, gt) # TODO
        self._update_metrics_at_batch_end(train_metrics)
        # Manual optimization like real men. We disable automatic_optimization in the constructor.
        if loss is not None and loss.requires_grad:
            self.manual_backward(loss)
            for opt in opts:
                opt.step()
        else:
            logger.debug(f"Loss is None or doesn't require grad ({loss}). Skipping this batch.")
        return loss

    @overrides
    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        """Validation step: returns batch validation loss and metrics."""
        y, val_metrics, _, gt = self.model_algorithm(self, batch) # pylint: disable=not-callable
        self.cache_result = tr_detach_data(y)
        val_metrics["loss"] = val_metrics["loss"] if "loss" in val_metrics else self.criterion_fn(y, gt) # TODO
        self._update_metrics_at_batch_end(val_metrics)
        return val_metrics["loss"]

    @overrides
    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        """Testing step: returns batch test loss and metrics."""
        y, test_metrics, _, gt = self.model_algorithm(self, batch) # pylint: disable=not-callable
        test_metrics["loss"] = test_metrics["loss"] if "loss" in test_metrics else self.criterion_fn(y, gt) # TODO
        self.cache_result = tr_detach_data(y)
        self._update_metrics_at_batch_end(test_metrics)
        return test_metrics["loss"]

    @overrides
    def predict_step(self, batch: Any, batch_idx: int, *args, dataloader_idx: int = 0, **kwargs) -> Any:
        return self.model_algorithm(self, batch)[0] # pylint: disable=not-callable

    @overrides
    def configure_callbacks(self) -> Sequence[pl.Callback] | pl.Callback:
        return self.callbacks

    @overrides
    def on_train_epoch_start(self):
        self._apply_scheduler_epoch_start()

    @overrides
    def on_train_epoch_end(self) -> None:
        """Computes epoch average train loss and metrics for logging."""
        self._run_and_log_metrics_at_epoch_end()
        self._reset_all_active_metrics()

    @overrides
    def on_test_epoch_end(self):
        self._run_and_log_metrics_at_epoch_end()
        self._reset_all_active_metrics()

    @overrides(check_signature=False)
    def configure_optimizers(self) -> list[optim.Optimizer] | \
            tuple[list[optim.Optimizer], list[optim.lr_scheduler.LRScheduler]]:
        """
        Configure the optimizer(s). We always return a list of optimizers (even if just 1)
        """
        if self.optimizer is None:
            raise ValueError("No optimizer. Use model.optimizer=optim.XXX or add an optimizer property in base model")
        res = make_list(self.optimizer)
        assert all(isinstance(x, optim.Optimizer) for x in res), (type(x) for x in res)
        if(scheduler := self.scheduler) is not None:
            return res, make_list(scheduler)
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

    def reset_parameters(self, seed: int | None = None):
        """Resets the parameters of the base model. Applied recursively as much as possible."""
        if seed is not None:
            seed_everything(seed)
        num_params = len(tuple(self.parameters()))
        if num_params == 0:
            return
        for layer in self.base_model.children():
            if LightningModuleEnhanced(layer).num_params == 0:
                continue

            if not hasattr(layer, "reset_parameters"):
                logger.debug2(f"Layer {layer} has params, but no reset_parameters() method. Trying recursively")
                layer = LightningModuleEnhanced(layer)
                layer.reset_parameters(seed)
            else:
                layer.reset_parameters()

    def lme_metrics(self, y: tr.Tensor, gt: tr.Tensor, include_loss: bool = True) -> dict[str, tr.Tensor]:
        """
        Pass through all the metrics of this batch and call forward. This updates the metric state for this batch
        Parameters:
        - y the output of the model
        - gt the ground truth
        - include_loss Whether to include the loss in the returned metrics. This can be useful when using a different
        model_algorithm, where we want to compute the loss manually as well.
        """
        metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            # Call the metric and update its state
            with (tr.enable_grad if metric_fn.requires_grad else tr.no_grad)():
                metrics[metric_name] = metric_fn.forward(y, gt)
        if self.criterion_fn is not None and include_loss is True:
            metrics["loss"] = self.criterion_fn(y, gt)
        return metrics

    @overrides(check_signature=False)
    def state_dict(self):
        return self.base_model.state_dict()

    @overrides(check_signature=False)
    def load_state_dict(self, *args, **kwargs):
        return self.base_model.load_state_dict(*args, **kwargs)

    @overrides
    def load_from_checkpoint(self, checkpoint_path: Path) -> LightningModuleEnhanced:
        data = tr.load(checkpoint_path)
        self.load_state_dict(data["state_dict"], strict=True)
        loaded_count = ["weights"]
        if "hyper_parameters" in data:
            for k, v in data["hyper_parameters"].items():
                self.hparams[k] = v
            cnt = len(data["hyper_parameters"])
            loaded_count.extend([f"{cnt} hyper_parameter{'s' if cnt > 1 else ''}"] if cnt > 0 else [])
        if "optimizer_states" in data:
            optimizers = make_list(self.optimizer) if self.optimizer is not None else []
            assert (A := len(optimizers)) == (B := len(data["optimizer_states"])), (A, B)
            for optimizer, optimizer_state in zip(optimizers, data["optimizer_states"]):
                optimizer.load_state_dict(optimizer_state)
            loaded_count.extend([f"{A} optimizer{'s' if A > 1 else ''}"] if A > 0 else [])
        if "lr_schedulers" in data:
            schedulers = make_list(self.scheduler) if self.scheduler is not None else []
            assert (A := len(schedulers)) == (B := len(data["lr_schedulers"])), (A, B)
            for scheduler, scheduler_state in zip(schedulers, data["lr_schedulers"]):
                scheduler["scheduler"].load_state_dict(scheduler_state)
            loaded_count.extend([f"{A} scheduler{'s' if A > 1 else ''}"] if A > 0 else [])
        logger.info(f"Loaded state from checkpoint '{checkpoint_path}'. Loaded state for: {', '.join(loaded_count)}")
        return self

    # Private methods

    @staticmethod
    def _default_algorithm(model: LightningModuleEnhanced, batch: Any) -> ModelAlgorithmOutput:
        raise NotImplementedError("Use model.model_algorithm=xxx to define your forward & metrics function")

    def _copy_loaded_checkpoint_and_metrics(self):
        """copies the loaded checkpoint to the log dir. Optionally: old best ckpt and metrics.csv history if exist"""
        if self.trainer.ckpt_path is None or self.trainer.global_rank != 0:
            return
        (new_ckpt_dir := Path(self.logger.log_dir) / "checkpoints").mkdir(exist_ok=True, parents=False)
        shutil.copyfile(self.trainer.ckpt_path, new_ckpt_dir / "loaded.ckpt")
        artifacts = "loaded ckpt"
        old_best_model_path = Path(self.trainer.checkpoint_callback.best_model_path)
        if old_best_model_path.exists() and old_best_model_path.is_file():
            shutil.copyfile(old_best_model_path, new_ckpt_dir / old_best_model_path.name)
            artifacts += ", best ckpt"
        if (old_metrics_csv_path := Path(self.trainer.ckpt_path).parents[1] / "metrics.csv").exists():
            old_metrics = open(old_metrics_csv_path).readlines()[0 : self.trainer.current_epoch + 1]
            open(f"{self.logger.log_dir}/metrics.csv", "w").write("\n".join(old_metrics))
            artifacts += ", metrics.csv history"
        logger.info(f"Loaded artifacts from old dir: {artifacts}")

    def _prefix_from_trainer(self) -> str:
        """returns the prefix: "" (for training), "val_" for validating or "" for testing as well"""
        assert self.trainer.training or self.trainer.validating or self.trainer.testing \
            or self.trainer.sanity_checking, self.trainer.state
        prefix = "" if self.trainer.training or self.trainer.testing else "val_"
        return prefix

    def _apply_scheduler_epoch_start(self):
        if self.scheduler is None or not self.trainer.training:
            return
        if "monitor" not in self.scheduler:
            self.scheduler["scheduler"].step() # TODO: tests for non monitor schedulers (wtf are these even?)
            return

        monitor = self.scheduler["monitor"]
        is_val = monitor.startswith("val_")
        prefix = "val" if is_val else ""
        monitor = monitor[4:] if is_val else monitor

        if is_val and "val_" not in self._active_run_metrics:
            raise MisconfigurationException(f"Monitor: {monitor} but no validation set provided")
        if monitor not in (metrics := self._active_run_metrics[prefix]):
            raise MisconfigurationException(f"Monitor: {monitor} not in metrics: {metrics}")
        if self.trainer.current_epoch == 0:
            return

        metric = self._active_run_metrics[prefix][monitor]
        try:
            self.scheduler["scheduler"].step()
        except Exception:
            self.scheduler["scheduler"].step(metrics=metric.epoch_result_reduced(metric.epoch_result())) # TODO?

    def __getattribute__(self, item: Any) -> Any:
        if item in ("state_dict", "load_state_dict"): # no 'base_model' prefix: for regular model loading compatibility
            return self.base_model.__getattribute__(item)
        try:
            return super().__getattribute__(item)
        except AttributeError:
            if item == "base_model":
                return self.__getattr__("base_model")
            return self.base_model.__getattr__(item)
