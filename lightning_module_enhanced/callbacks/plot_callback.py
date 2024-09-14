"""Module to create a plot callback for train and/or validation for a Lightning Module"""
from __future__ import annotations
from typing import Callable
from pathlib import Path
from overrides import overrides
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..logger import lme_logger as logger

class PlotCallbackGeneric(Callback):
    """Plot callback impementation. For each train/validation epoch, create a dir under logger_dir/pngs/epoch_X"""
    def __init__(self, plot_callback: Callable, mkdir: bool = True):
        self.plot_callback = plot_callback
        self.mkdir = mkdir

    @staticmethod
    def _get_out_dir(trainer: Trainer, dir_name: str, mkdir: bool) -> Path | None:
        """Gets the output directory as '/path/to/log_dir/pngs/train_or_val/epoch_N/' """
        if len(trainer.loggers) == 0:
            return None
        out_dir = Path(f"{trainer.loggers[0].log_dir}/pngs/{dir_name}/{trainer.current_epoch + 1}")
        if mkdir:
            out_dir.mkdir(exist_ok=True, parents=True)
        return out_dir

    def _get_prediction(self, pl_module: LightningModule):
        assert hasattr(pl_module, "cache_result") and pl_module.cache_result is not None
        y = pl_module.cache_result
        return y

    def _do_call(self, trainer: Trainer, pl_module: LightningModule, batch, batch_idx, key: str):
        if batch_idx != 0:
            return
        if len(trainer.loggers) == 0:
            logger.debug("No lightning logger found. Not calling PlotCallbacks()")
            return
        try:
            prediction = self._get_prediction(pl_module)
        except Exception:
            logger.debug("No prediction yet, somehow called before model_algorithm. Returning")
            return

        out_dir = PlotCallbackGeneric._get_out_dir(trainer, key, self.mkdir)
        self.plot_callback(model=pl_module, batch=batch, y=prediction, out_dir=out_dir)

    @rank_zero_only
    @overrides
    # pylint: disable=unused-argument
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                                outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._do_call(trainer, pl_module, batch, batch_idx, "validation")

    @rank_zero_only
    @overrides
    # pylint: disable=unused-argument
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs, batch, batch_idx: int, unused: int = 0):
        self._do_call(trainer, pl_module, batch, batch_idx, "train")

    @rank_zero_only
    @overrides
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._do_call(trainer, pl_module, batch, batch_idx, "test")

class PlotCallback(PlotCallbackGeneric):
    """Above implementation + assumption about data/labels keys"""
    @overrides
    def _do_call(self, trainer, pl_module, batch, batch_idx, key):
        if batch_idx != 0:
            return
        if len(trainer.loggers) == 0:
            logger.debug("No lightning logger found. Not calling PlotCallbacks()")
            return
        try:
            prediction = self._get_prediction(pl_module)
        except Exception:
            logger.debug("No prediction yet, somehow called before model_algorithm. Returning")
            return

        out_dir = PlotCallbackGeneric._get_out_dir(trainer, key, self.mkdir)
        x, gt = batch["data"], batch["labels"]
        self.plot_callback(x=x, y=prediction, gt=gt, out_dir=out_dir, model=pl_module)
