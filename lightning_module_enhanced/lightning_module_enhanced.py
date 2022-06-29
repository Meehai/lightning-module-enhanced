"""Generic Pytorch Lightning Graph module on top of a Graph module"""
from typing import Dict, Callable, List, Union, Any, Sequence
from overrides import overrides
import torch as tr
from torch import optim, nn
from torchinfo import summary, ModelStatistics
from pytorch_lightning import Callback, LightningModule
from torchmetrics import Metric
from nwutils.torch import tr_get_data, tr_to_device

from .logger import logger

# pylint: disable=too-many-ancestors, arguments-differ, unused-argument, abstract-method
class LightningModuleEnhanced(LightningModule):
    """Pytorch Lightning module enhanced"""
    def __init__(self, base_model: nn.Module, *args, **kwargs):
        assert isinstance(base_model, nn.Module)
        super().__init__()
        self.save_hyperparameters({"args": args, **kwargs})
        self.base_model = base_model
        self.optimizer: optim.Optimizer = None
        self.scheduler_dict: optim.lr_scheduler._LRScheduler = None
        self.criterion_fn: Callable[[tr.Tensor, tr.Tensor], float] = None
        self.metrics: Dict[str, Metric] = {}
        self.logged_metrics: List[str] = []
        self._summary: ModelStatistics = None
        self.callbacks: List[Callback] = []

    # Getters and setters for properties

    @property
    def device(self) -> tr.device:
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
        return self.num_trainable_params > 0

    @trainable_params.setter
    def trainable_params(self, value: bool):
        """Sets all the parameters of this module to trainable or untrainable"""
        logger.debug(f"Setting parameters of the model to '{value}'.")
        for param in self.base_model.parameters():
            param.requires_grad_(value)
        # Reset summary such that it is recomputted if necessary (like for counting num trainable params)
        self._summary = None


    # Pytorch lightning overrides

    @overrides
    def training_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs) -> Union[tr.Tensor, Dict[str, Any]]:
        """Training step: returns batch training loss and metrics."""
        return self._generic_step(train_batch)

    @overrides
    def validation_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs):
        """Training step: returns batch validation loss and metrics."""
        return self._generic_step(train_batch, "val_")

    @overrides
    def test_step(self, train_batch: Dict, batch_idx: int, *args, **kwargs):
        """Training step: returns batch validation loss and metrics."""
        return self._generic_step(train_batch, "test_")

    @overrides
    def configure_callbacks(self) -> Union[Sequence[Callback], Callback]:
        return self.callbacks

    @overrides
    def training_epoch_end(self, outputs):
        """Computes epoch average train loss and metrics for logging."""
        self._on_epoch_end(outputs)

    @overrides
    def validation_epoch_end(self, outputs):
        """Computes epoch average validation loss and metrics for logging."""
        self._on_epoch_end(outputs)

    @overrides
    def test_epoch_end(self, outputs):
        """Computes average test loss and metrics for logging."""
        self._on_epoch_end(outputs)

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

    def _generic_train_step(self, train_batch: Dict):
        """Generic step for computing the forward pass during training."""
        x = train_batch["data"]
        y = self.forward(x)
        return y

    def _generic_get_outputs(self, y, gt, prefix: str):
        loss = self.criterion_fn(y, gt)
        outputs = {f"{prefix}loss": loss}
        for metric_name, metric_callback in self.metrics.items():
            with tr.no_grad():
                outputs[f"{prefix}{metric_name}"] = metric_callback(y, gt)
        for metric_name in self.logged_metrics:
            self.log(f"{prefix}{metric_name}", outputs[f"{prefix}{metric_name}"], prog_bar=True, on_step=True)
        return outputs

    def _generic_step(self, train_batch: Dict, prefix: str = ""):
        """Generic step for computing the forward pass, loss and metrics."""
        y = self._generic_train_step(train_batch)
        gt = tr_to_device(tr_get_data(train_batch["labels"]), self.device)
        return self._generic_get_outputs(y, gt, prefix)

    # pylint: disable=no-member
    def _on_epoch_end(self, outputs):
        keys = outputs[0].keys()
        avgs = {}
        for key in keys:
            if key == "loss":
                breakpoint()
            items = tr.stack([x[key] for x in outputs])
            # Shape: (N, ) or (N, 1) => just average it
            if len(items.shape) == 1 or (len(items.shape) == 2 and items.shape[-1] == 1):
                avgs[key] = items.mean()
                self.log(key, avgs[key], prog_bar=True)
            else:
                avgs[key] = items.sum(dim=0)
                # TODO: Log confusion matrix somehow using lightning
                logger.info(f"{key} => {avgs[key].numpy()}")

    def reset_parameters(self):
        """Resets the parameters of the base model"""
        for layer in self.base_model.children():
            layer.reset_parameters()

    def setup_module_for_train(self, train_cfg: Dict):
        """Given a train cfg, prepare this module for training, by setting the required information."""
        from .train_setup import TrainSetup
        TrainSetup(self, train_cfg).setup()
