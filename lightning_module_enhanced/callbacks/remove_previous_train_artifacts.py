"""RemovePreviousTrainArtifacts module"""
from pathlib import Path
import os
from pytorch_lightning import Callback, LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class RemovePreviousTrainArtifacts(Callback):
    """Callback to remove the previous trains artifacts (*.tfevents files), if any available"""
    @rank_zero_only
    def on_train_start(self, trainer: LightningModule, pl_module: LightningModule) -> None:
        """Clean up the .tfevents files, besides the last one."""
        RemovePreviousTrainArtifacts._on_start(Path(pl_module.logger.log_dir))

    @rank_zero_only
    def on_test_start(self, trainer: LightningModule, pl_module: LightningModule) -> None:
        RemovePreviousTrainArtifacts._on_start(Path(pl_module.logger.log_dir))

    @staticmethod
    def _on_start(log_dir: Path):
        tfevents_files = sorted([str(x) for x in log_dir.glob("*tfevents*")])
        if len(tfevents_files) > 1:
            for file in tfevents_files[0: -1]:
                os.unlink(file)
