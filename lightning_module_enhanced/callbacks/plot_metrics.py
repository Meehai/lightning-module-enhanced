"""Module to plots metrics"""
from typing import Dict, List, Any
from overrides import overrides
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

class PlotMetrics(Callback):
    """Plot metrics implementation"""
    def __init__(self):
        self.history: Dict[str, List[float]] = None
        self.current_epoch_results: Dict[str, List[float]] = None

    def _generic_on_batch_end(self, prefix, outputs):
        assert prefix in ("train", "val")
        metrics = outputs.keys()
        # Sometimes they are "val_loss", sometimes (for train) just "loss".
        unprefixed = ["_".join(metric.split("_")[1: ]) if metric.startswith(prefix) else metric for metric in metrics]
        if self.current_epoch_results is None:
            self.current_epoch_results = {u_metric: {"train": [], "val": []} for u_metric in unprefixed}
        for unprefixed_metric, metric in zip(unprefixed, metrics):
            value = float(outputs[metric].detach().to("cpu"))
            self.current_epoch_results[unprefixed_metric][prefix].append(value)

    @overrides
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._generic_on_batch_end(prefix="val", outputs=outputs)

    @overrides
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=False):
        self._generic_on_batch_end(prefix="train", outputs=outputs)

    @staticmethod
    def _do_plot(metric_history, metric, out_file):
        plt.figure()
        _range = range(1, len(metric_history["train"]) + 1)
        plt.plot(_range, metric_history["train"], label="train")
        if None not in metric_history["val"]:
            plt.plot(_range, metric_history["val"], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(out_file)
        plt.close()

    def _generic_on_epoch_end(self, out_dir, current_epoch):
        if self.history is None:
            self.history = {metric: {"train": [], "val": []} for metric in self.current_epoch_results.keys()}
        if current_epoch < len(self.history["loss"]["train"]):
            return
        assert self.current_epoch_results is not None
        # Validation sanity check stuff
        if len(self.current_epoch_results["loss"]["train"]) == 0:
            return

        for metric in self.current_epoch_results.keys():
            train_mean = np.mean(self.current_epoch_results[metric]["train"])
            val_res = self.current_epoch_results[metric]["val"]
            val_mean = np.mean(val_res) if len(val_res) > 0 else None
            assert len(self.history[metric]["train"]) == len(self.history[metric]["val"]) == current_epoch
            self.history[metric]["train"].append(train_mean)
            self.history[metric]["val"].append(val_mean)

            out_file = f"{out_dir}/{metric}.png"
            PlotMetrics._do_plot(self.history[metric], metric, out_file)
        self.current_epoch_results = None

    @overrides
    def on_validation_epoch_end(self, trainer, pl_module):
        self._generic_on_epoch_end(trainer.logger.log_dir, trainer.current_epoch)

    @overrides
    def on_train_epoch_end(self, trainer, pl_module):
        # Call _generic_on_epoch_end only if there is no validation dataset.
        if trainer.val_dataloaders is not None and len(trainer.val_dataloaders) > 0:
            return
        self._generic_on_epoch_end(trainer.logger.log_dir, trainer.current_epoch)

    @overrides
    def state_dict(self) -> Dict[str, Any]:
        return {"history": self.history}

    @overrides
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.history = state_dict["history"]
