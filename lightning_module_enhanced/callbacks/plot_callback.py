"""Module to create a plot callback for train and/or validation for a Lightning Module"""
from typing import Callable, Any
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pathlib import Path
import torch as tr
from ..lightning_module_enhanced import LightningModuleEnhanced

class PlotCallbackGeneric(Callback):
    """Plot callback impementation. For each train/validation epoch, create a dir under logger_dir/pngs/epoch_X"""
    def __init__(self, plot_callback: Callable):
        self.plot_callback = plot_callback

    def get_out_dir(self, trainer: Trainer, dir_name: str) -> Path:
        logger = trainer.logger
        out_dir = Path(f"{logger.log_dir}/pngs/{dir_name}/{trainer.current_epoch}")
        out_dir.mkdir(exist_ok=True, parents=True)
        return out_dir
    
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModuleEnhanced,
                                outputs, batch, batch_idx, dataloader_idx):
        if batch_idx != 0:
            return
        out_dir = self.get_out_dir(trainer, "validation")
        with tr.no_grad():
            y = pl_module.forward(batch)
        self.plot_callback(model=pl_module, batch=batch, y=y, out_dir=out_dir)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModuleEnhanced,
                                outputs, batch, batch_idx, dataloader_idx):
        if batch_idx != 0:
            return
        out_dir = self.get_out_dir(trainer, "train")
        with tr.no_grad():
            y = pl_module.forward(batch)
        self.plot_callback(model=pl_module, batch=batch, y=y, out_dir=out_dir)

class PlotCallback(PlotCallbackGeneric):
    """Above implementation + assumption about data/labels keys"""
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModuleEnhanced,
                                outputs, batch: Any, batch_idx: int, dataloader_idx: int):
        if batch_idx != 0:
            return
        out_dir = self.get_out_dir(trainer, "validation")

        x, gt = batch["data"], batch["labels"]
        with tr.no_grad():
            y = pl_module.forward(x)

        self.plot_callback(x=x, y=y, gt=gt, out_dir=out_dir)

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModuleEnhanced,
                                outputs, batch: Any, batch_idx: int, dataloader_idx: int):
        if batch_idx != 0:
            return
        out_dir = self.get_out_dir(trainer, "train")

        x, gt = batch["data"], batch["labels"]
        with tr.no_grad():
            y = pl_module.forward(x)

        self.plot_callback(x=x, y=y, gt=gt, out_dir=out_dir)
