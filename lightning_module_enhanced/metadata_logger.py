import json
from pathlib import Path
from pytorch_lightning import LightningModule
import torch as tr

class MetadataLogger:
    """Metadata Logger for a CoreModule. Stores various information about a training."""
    def __init__(self, model: LightningModule, log_dir: str):
        self.model = model
        self.log_dir = Path(log_dir).absolute()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file_path = self.log_dir / "metadata.json"
        self.metadata = {
            "epoch_metrics": {},
            "hparams_current": None,
        }

        if self.log_file_path.exists():
            with open(self.log_file_path, "r") as fp:
                json.load(fp)

    def save_metadata(self, key, value):
        """Adds a key->value pair to the current metadata"""
        self.metadata[key] = value

    def save_epoch_metric(self, key: str, value: tr.Tensor, epoch: int):
        """Adds a epoch metric to the current metadata"""
        if not key in self.metadata["epoch_metrics"]:
            self.metadata["epoch_metrics"][key] = {}
        if epoch != 0:
            # Epoch 0 can sometimes have a validation sanity check fake epoch
            assert epoch not in self.metadata["epoch_metrics"][key], f"Cannot overwrite existing epoch metric '{key}'"
        self.metadata["epoch_metrics"][key][epoch] = value.tolist()

    def save(self):
        # Always use the current hparams such that, for test modes, we get the loaded stats
        metadata = {**self.metadata}
        metadata["hparams_current"] = self.model.hparams
        with open(self.log_file_path, "w") as fp:
            json.dump(metadata, fp)

    def __str__(self):
        return f"Metadata Logger. Log dir: '{self.log_dir}'"
