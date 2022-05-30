"""Module to plots metrics"""
from typing import Dict, List, Any
from overrides import overrides
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt


class PlotMetrics(Callback):
    """Plot metrics implementation"""
    def __init__(self):
        self.history: Dict[str, List[float]] = None
        self.current_epoch_history: Dict[str, List[float]] = None

    @overrides
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.current_epoch_history is None:
            self.current_epoch_history = {metric: [] for metric in outputs.keys()}
        for metric in outputs.keys():
            value = float(outputs[metric].detach().to("cpu"))
            self.current_epoch_history[metric].append(value)

    @overrides
    def on_validation_epoch_end(self, trainer, pl_module):
        assert self.current_epoch_history is not None
        if self.history is None:
            self.history = {metric: [] for metric in self.current_epoch_history.keys()}
        for metric in self.current_epoch_history.keys():
            current_count = len(self.current_epoch_history[metric])
            value = sum(self.current_epoch_history[metric]) / current_count
            self.history[metric].append(value)
            out_file = f"{trainer.logger.log_dir}/{metric}.png"

            plt.figure()
            plt.plot(range(1, len(self.history[metric]) + 1), self.history[metric])
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.savefig(out_file)
            plt.close()
        self.current_epoch_history = None

    @overrides
    def state_dict(self) -> Dict[str, Any]:
        return {"history": self.history}

    @overrides
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.history = state_dict["history"]
