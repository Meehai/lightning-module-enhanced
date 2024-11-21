"""Module to create a plot callback for train and/or validation for a Lightning Module"""
# pylint: disable=too-many-positional-arguments
from __future__ import annotations
from typing import Callable
from pathlib import Path
from overrides import overrides
import torch as tr
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..logger import lme_logger as logger

class PlotCallbackGeneric(Callback):
    """Plot callback impementation. For each train/validation epoch, create a dir under logger_dir/pngs/epoch_X"""
    def __init__(self, plot_callback: Callable, mode: str = "first"):
        assert mode in ("first", "random"), mode
        self.plot_callback = plot_callback
        self.mode = mode

    @staticmethod
    def _get_out_dir(trainer: Trainer, dir_name: str) -> Path | None:
        """Gets the output directory as '/path/to/log_dir/pngs/train_or_val/epoch_N/' """
        if len(trainer.loggers) == 0:
            return None
        out_dir = Path(f"{trainer.logger.log_dir}/pngs/{dir_name}/{trainer.current_epoch + 1}")
        return out_dir

    def _get_prediction(self, pl_module: LightningModule):
        assert hasattr(pl_module, "cache_result") and pl_module.cache_result is not None
        y = pl_module.cache_result
        return y

    def _check_if_should_skip(self, trainer: Trainer, batch_idx: int, n_batches: int) -> bool:
        if self.mode == "first" or trainer.state.stage == "sanity_check":
            return batch_idx != 0
        tr.manual_seed(trainer.current_epoch) # hopefully guaranteed to give us the same permutation during an epoch
        n_batches = n_batches[0] if isinstance(n_batches, list) and len(n_batches) == 1 else n_batches # wtf?
        return batch_idx != tr.randperm(n_batches)[0].item()

    def _do_call(self, pl_module: LightningModule, batch: dict, batch_idx: int, key: str, n_batches: int):
        if self._check_if_should_skip(pl_module.trainer, batch_idx, n_batches):
            return
        assert len(pl_module.trainer.loggers) > 0, "No lightning logger found. Cannot use PlotCallbackGeneric."
        try:
            prediction = self._get_prediction(pl_module)
        except Exception as e:
            logger.debug(f"Exception {e}. No prediction yet, somehow called before model_algorithm. Returning.")
            return

        out_dir = PlotCallbackGeneric._get_out_dir(pl_module.trainer, key)
        self.plot_callback(model=pl_module, batch=batch, y=prediction, out_dir=out_dir)

    @rank_zero_only
    @overrides
    # pylint: disable=unused-argument
    def on_validation_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                                outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._do_call(pl_module, batch, batch_idx, "validation", trainer.num_val_batches)

    @rank_zero_only
    @overrides
    # pylint: disable=unused-argument
    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                           outputs, batch, batch_idx: int, unused: int = 0):
        self._do_call(pl_module, batch, batch_idx, "train", trainer.num_training_batches)

    @rank_zero_only
    @overrides
    def on_test_batch_end(self, trainer: Trainer, pl_module: LightningModule,
                          outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        self._do_call(pl_module, batch, batch_idx, "test", trainer.num_test_batches)

class PlotCallback(PlotCallbackGeneric):
    """Above implementation + assumption about data/labels keys"""
    @overrides
    def _do_call(self, pl_module: LightningModule, batch: dict, batch_idx: int, key: str, n_batches: int):
        if self._check_if_should_skip(pl_module.trainer, batch_idx, n_batches):
            return
        assert len(pl_module.trainer.loggers) > 0, "No lightning logger found. Cannot use PlotCallback."
        try:
            prediction = self._get_prediction(pl_module)
        except Exception:
            logger.debug("No prediction yet, somehow called before model_algorithm. Returning")
            return

        out_dir = PlotCallbackGeneric._get_out_dir(pl_module.trainer, key)
        x, gt = batch["data"], batch["labels"]
        self.plot_callback(x=x, y=prediction, gt=gt, out_dir=out_dir, model=pl_module)
