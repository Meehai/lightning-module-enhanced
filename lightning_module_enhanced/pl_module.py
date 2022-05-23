"""Generic Pytorch Lightning Graph module on top of a Graph module"""
from typing import Dict, Callable, List, Union, Any
from overrides import overrides
import logging as logger
import torch as tr
from torch import optim, nn
from torchinfo import summary
from pytorch_lightning import Callback, LightningModule
from torchmetrics import Metric
from nwutils.torch import tr_get_data, tr_to_device


# pylint: disable=too-many-ancestors, arguments-differ, unused-argument, abstract-method
class PLModule(LightningModule):
    """Pytorch Lightning module class for Graphs"""
    def __init__(self, base_model: nn.Module, *args, **kwargs):
        assert isinstance(base_model, nn.Module)
        super().__init__()
        self.save_hyperparameters({"args": args, **kwargs})
        self.base_model = base_model
        self.optimizer = None # TODO getter/setter with string, type and instance options.
        self.scheduler = []
        self.criterion_fn: Callable[[tr.Tensor, tr.Tensor], float]
        self.metrics: Dict[str, Metric] = {}
        self.logged_metrics: List[str] = []
        self._summary = None
        self.callbacks: List[Callback] = []

    @overrides
    def configure_callbacks(self):
        return self.callbacks

    @property
    def device(self):
        return next(self.base_model.parameters()).device

    def generic_train_step(self, train_batch: Dict, prefix: str):
        """Generic step for computing the forward pass during training."""
        x = train_batch["data"]
        y = self.forward(x)
        return y

    def _generic_get_outputs(self, y, gt, prefix: str):
        loss = self.criterion_fn(y, gt)
        outputs = {f"{prefix}loss": loss}
        y = y.detach()
        for metric_name, metric_callback in self.metrics.items():
            outputs[f"{prefix}{metric_name}"] = metric_callback(y, gt)
        for metric_name in self.logged_metrics:
            self.log(f"{prefix}{metric_name}", outputs[f"{prefix}{metric_name}"], prog_bar=True, on_step=True)
        return outputs

    def _generic_step(self, train_batch: Dict, prefix: str = ""):
        """Generic step for computing the forward pass, loss and metrics."""
        y = self.generic_train_step(train_batch, prefix)
        gt = tr_to_device(tr_get_data(train_batch["labels"]), self.device)
        return self._generic_get_outputs(y, gt, prefix)

    def forward(self, *args, **kwargs):
        """Model's forward pass."""
        return self.base_model.forward(*args, **kwargs)

    def np_forward(self, *args, **kwargs):
        """Forward numpy data to the model, returns whatever the model returns, usually torch data"""
        tr_args = tr_to_device(tr_get_data(args), self.device)
        tr_kwargs = tr_to_device(tr_get_data(kwargs), self.device)
        with tr.no_grad():
            y_tr = self.base_model.forward(*tr_args, **tr_kwargs)
        return y_tr

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

    # pylint: disable=no-member
    def _on_epoch_end(self, outputs):
        keys = outputs[0].keys()
        avgs = {}
        for key in keys:
            items = tr.stack([x[key] for x in outputs])
            # Shape: (N, ) or (N, 1) => just average it
            if len(items.shape) == 1 or (len(items.shape) == 2 and items.shape[-1] == 1):
                avgs[key] = items.mean()
                self.log(key, avgs[key], prog_bar=True)
            else:
                avgs[key] = items.sum(dim=0)
                # TODO: Log confusion matrix somehow using lightning
                logger.info(f"{key} => {avgs[key].numpy()}")

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

    def configure_optimizers(self) -> Dict:
        """Configure the optimizer/scheduler/monitor."""
        if self.optimizer is None:
            raise ValueError("No optimizer has been provided. Set a torch optimizer first.")
        if self.scheduler:
            assert len(self.scheduler) == 2
            return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler[0], "monitor": self.scheduler[1]}
        return {"optimizer": self.optimizer}

    def reset_parameters(self):
        """Resets the parameters of the base model"""
        for layer in self.base_model.children():
            layer.reset_parameters()

    def summary(self, **kwargs) -> str:
        """Prints the summary (layers, num params, size in MB), with the help of torchinfo module."""
        self._summary = summary(self, verbose=0, **kwargs) if self._summary is None else self._summary
        return self._summary

    def get_num_params(self) -> int:
        """Returns the total number of parameters of this module"""
        return self.summary().total_params

    def get_num_trainable_params(self) -> int: #pylint: disable=no-self-use
        """Returns the trainable number of parameters of this module"""
        return self.summary().trainable_params

    def set_trainable_params(self, value: bool):
        """Sets all the parameters of this module to trainable or untrainable"""
        logger.debug(f"Setting parameters of {self.base_model} to '{value}'.")
        for param in self.base_model.parameters():
            param.requires_grad_(value)

    def setup_module_for_train(self, train_cfg: Dict):
        """Given a train cfg, prepare this module for training, by setting the required information
            TODO: create a train_cfg object and make this more customizable
        """
        optimizer_type = {
            "adamw": optim.AdamW,
            "adam": optim.Adam,
            "sgd": optim.SGD,
            "rmsprop": optim.RMSprop
        }[train_cfg["optimizer"]["type"]]
        self.optimizer = optimizer_type(self.base_model.parameters(), **train_cfg["optimizer"]["args"])
