"""Generic Pytorch Lightning Graph module on top of a Graph module"""
from __future__ import annotations
from typing import Dict, List, Union, Any, Sequence
from copy import deepcopy
from pathlib import Path
from overrides import overrides
import torch as tr
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from torch import nn
from torchinfo import summary, ModelStatistics

from .trainable_module import TrainableModuleMixin, TrainableModule
from .metrics import CoreMetric
from .logger import logger
from .utils import to_tensor, to_device

# pylint: disable=too-many-ancestors, arguments-differ, unused-argument, abstract-method
class CoreModule(TrainableModuleMixin, pl.LightningModule):
    """

        Generic pytorch-lightning module for ml models.

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
        assert isinstance(base_model, nn.Module), f"Expected a nn.Module, got {type(base_model)}"
        super().__init__()
        self.base_model = base_model
        self._prefixed_metrics: Dict[str, Dict[str, CoreMetric]] = {}
        self._logged_metrics: List[str] = None
        self._summary: ModelStatistics = None

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
    def logged_metrics(self) -> List[str]:
        """Return the list of logged metrics out of all the defined ones"""
        if self._logged_metrics is None:
            logger.debug2("Logged metrics is not set, defaulting to all metrics")
            return list(self.metrics.keys())
        return self._logged_metrics

    @logged_metrics.setter
    def logged_metrics(self, logged_metrics: List[str]):
        logger.debug(f"Setting the logged metrics to {logged_metrics}")
        diff = set(logged_metrics).difference(self.metrics.keys())
        assert len(diff) == 0, f"Metrics {diff} are not in set metrics: {self.metrics.keys()}"
        self._logged_metrics = logged_metrics

    # Overrides on top of the standard pytorch lightning module
    @overrides
    def on_fit_start(self) -> None:
        if self.criterion_fn is None:
            raise ValueError("Criterion must be set before calling trainer.fit()")
        self._prefixed_metrics[""] = self.metrics
        if self.trainer.enable_validation:
            # If we use a validation set, clone all the metrics, so that the statistics don't intefere with each other
            self._prefixed_metrics["val_"] = CoreModule._clone_all_metrics_with_prefix(self.metrics, prefix="val_")
        return super().on_fit_start()

    @overrides
    def on_fit_end(self):
        self._prefixed_metrics = {}

    @overrides
    def on_test_start(self) -> None:
        self._prefixed_metrics[""] = self.metrics
        if self.criterion_fn is None:
            raise ValueError("Criterion must be set before calling trainer.test()")
        return super().on_test_start()

    @overrides
    def on_test_end(self):
        self._prefixed_metrics = {}

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
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return self.forward(batch["data"])

    @overrides
    def configure_callbacks(self) -> Union[Sequence[pl.Callback], pl.Callback]:
        return self.callbacks

    @overrides
    def training_epoch_end(self, outputs):
        """Computes epoch average train loss and metrics for logging."""
        # If validation is enabled (for train loops), add "val_" metrics for all logged metrics.
        self._run_and_log_metrics_at_epoch_end(self.logged_metrics)

    @overrides
    def test_epoch_end(self, outputs):
        self._run_and_log_metrics_at_epoch_end(self.logged_metrics)

    @overrides
    def configure_optimizers(self) -> Dict:
        """Configure the optimizer/scheduler/monitor."""
        if self.optimizer is None:
            raise ValueError("No optimizer has been set. Use model.optimizer=optim.XXX or add an optimizer "
                             "property in base model")

        if isinstance(self.optimizer, list):
            res = [{"optimizer": o} for o in self.optimizer]
        else:
            res = [{"optimizer": self.optimizer}]

        if self.scheduler_dict is None:
            return res

        if isinstance(self.scheduler_dict, list):
            res_scheduler = [{"lr_scheduler": s} for s in self.scheduler_dict]
        else:
            res_scheduler = [{"lr_scheduler": self.scheduler_dict}]

        assert len(res) == len(res_scheduler), "Something is messed up in your configs. Num optimizer: " \
            f"{res} ({len(res)}), schedulers: {res_scheduler} ({len(res_scheduler)})"

        for i in range(len(res)):
            res[i] = {**res[i], **res_scheduler[i]}
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

    def reset_parameters(self, seed: int = None):
        """Resets the parameters of the base model"""
        if seed is not None:
            seed_everything(seed)
        num_params = len(tuple(self.parameters()))
        if num_params == 0:
            return
        for layer in self.base_model.children():
            if CoreModule(layer).num_params == 0:
                continue

            if not hasattr(layer, "reset_parameters"):
                logger.debug2(f"Layer {layer} has params, but no reset_parameters() method. Trying recursively")
                layer = CoreModule(layer)
                layer.reset_parameters(seed)
            else:
                layer.reset_parameters()

    def load_state_from_path(self, path: str) -> CoreModule:
        """Loads the state dict from a path"""
        # if path is remote (gcs) download checkpoint to a temp dir
        logger.info(f"Loading weights and hyperparameters from '{Path(path).absolute()}'")
        ckpt_data = tr.load(path, map_location="cpu")
        self.load_state_dict(ckpt_data["state_dict"])
        self.save_hyperparameters(ckpt_data["hyper_parameters"])
        return self

    def state_dict(self):
        return self.base_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        return self.base_model.load_state_dict(state_dict, strict)

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
            metric_fn: CoreMetric = self._prefixed_metrics[prefix][prefixed_metric_name]
            # Call the metric and update its state
            state = tr.enable_grad if metric_fn.requires_grad else tr.no_grad
            with state():
                metric_output: tr.Tensor = metric_fn.forward(y, gt)
            metric_fn.batch_update(metric_output)
            outputs[prefixed_metric_name] = metric_output
            # Don't use any self.log() here. We don't really care about intermediate batch results, only epoch results,
            #  which are handled in self._run_and_log_metrics_at_epoch_end(metrics).
        return outputs

    def _run_and_log_metrics_at_epoch_end(self, metrics_to_log: List[str]):
        """Runs and logs a given list of logged metrics. Assume they all exist in self.metrics"""
        all_prefixes = self._prefixed_metrics.keys()
        for metric_name in metrics_to_log:
            for prefix in all_prefixes:
                prefixed_metric = f"{prefix}{metric_name}"
                metric_fn: CoreMetric = self._prefixed_metrics[prefix][prefixed_metric]
                # Get the metric's epoch result
                metric_epoch_result = metric_fn.epoch_result()
                # Log the metric at the end of the epoch. Only log on pbar the val_loss, loss is tracked by default
                prog_bar = metric_name == "val_loss"

                value_reduced = metric_fn.epoch_result_reduced(metric_epoch_result)
                if value_reduced is not None:
                    self.log(prefixed_metric, value_reduced, prog_bar=prog_bar, on_epoch=True)
                # Call the metadata callback for the full result, since it can handle any sort of metrics
                self.metadata_callback.save_epoch_metric(prefixed_metric, metric_epoch_result,
                                                         self.trainer.current_epoch)
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
            cloned_metric: CoreMetric = deepcopy(metric_fn)
            cloned_metric.reset()
            new_metrics[f"{prefix}{metric_name}"] = cloned_metric

        return new_metrics
