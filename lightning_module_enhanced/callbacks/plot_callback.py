"""Module to create a plot callback for train and/or validation for a Lightning Module"""
from typing import Callable
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pathlib import Path
import torch as tr
from ..lightning_module_enhanced import LightningModuleEnhanced

class PlotCallback(Callback):
    """Plot callback impementation. For each train/validation epoch, create a dir under logger_dir/pngs/epoch_X"""
    def __init__(self, plot_callback: Callable):
        self.plot_callback = plot_callback

    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModuleEnhanced,
                                outputs, batch, batch_idx, dataloader_idx):
        if batch_idx != 0:
            return
        logger = trainer.logger
        out_dir = Path(f"{logger.log_dir}/pngs/validation/{trainer.current_epoch}")
        out_dir.mkdir(exist_ok=True, parents=True)

        x, gt = batch["data"], batch["labels"]
        with tr.no_grad():
            y = pl_module.forward(x)
        breakpoint()
        
        self.plot_callback(x=x, y=y, gt=gt, out_dir=out_dir)
