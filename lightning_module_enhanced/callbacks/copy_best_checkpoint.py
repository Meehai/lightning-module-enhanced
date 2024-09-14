"""CopyBestCheckpoint module"""
from pathlib import Path
import shutil
from overrides import overrides
from pytorch_lightning import Trainer, Callback, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from ..logger import lme_logger as logger

class CopyBestCheckpoint(Callback):
    """Callback to store the best epoch with a special name used throughout the library"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_checkpoint_loss_callback = None

    @rank_zero_only
    @overrides
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for callback in trainer.callbacks:
            if not isinstance(callback, ModelCheckpoint):
                continue
            if callback.monitor is not None and callback.monitor != "val_loss":
                continue
            self.model_checkpoint_loss_callback = callback
        assert self.model_checkpoint_loss_callback is not None, "Not found ModelCheckpoint with 'val_loss' monitor"

    @rank_zero_only
    @overrides
    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        assert self.model_checkpoint_loss_callback is not None
        in_dir = Path(pl_module.logger.log_dir).absolute() / "checkpoints"
        out_file = in_dir / "model_best.ckpt"
        in_file = self.model_checkpoint_loss_callback.best_model_path
        if in_file == "":
            logger.debug("No best model was stored, just last model path (probably no validation set used).")
            in_file = self.model_checkpoint_loss_callback.last_model_path
        # assert not out_file.exists(), f"Out file '{out_file}' already exists..."
        in_file = Path(in_file).absolute()
        shutil.copyfile(in_file, out_file)
