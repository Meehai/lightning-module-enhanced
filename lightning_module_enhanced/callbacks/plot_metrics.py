"""Module to plots metrics"""
from typing import Dict, List, Any
from overrides import overrides
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from ..logger import logger

class PlotMetrics(Callback):
    """Plot metrics implementation"""
    def __init__(self):
        self.history: Dict[str, List[float]] = None

    # pylint: disable=protected-access
    def _plot_best_dot(self, ax: plt.Axes, pl_module, metric_name):
        """Plot the dot. We require to know if the metric is max or min typed."""
        metric = pl_module.metrics[metric_name]
        metric_history = self.history[metric_name]
        scores = metric_history["val"] if metric_history["val"][0] is not None else metric_history["train"]
        metric_x = np.argmax(scores) if metric.higher_is_better else np.argmin(scores)
        metric_y = scores[metric_x]
        ax.annotate(f"Epoch {metric_x + 1}\nMax {metric_y:.2f}", xy=(metric_x + 1, metric_y))
        ax.plot([metric_x + 1], [metric_y], "o")

    def _do_plot(self, pl_module, metric_name: str, out_file: str):
        """Plot the figure with the metric"""
        fig = plt.figure()
        ax = fig.gca()
        metric_history = self.history[metric_name]
        _range = range(1, len(metric_history["train"]) + 1)
        ax.plot(_range, metric_history["train"], label="train")
        if None not in metric_history["val"]:
            ax.plot(_range, metric_history["val"], label="validation")
        self._plot_best_dot(ax, pl_module, metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        fig.legend()
        fig.savefig(out_file)
        plt.close(fig)

    @overrides
    def on_fit_start(self, trainer, pl_module) -> None:
        self.history = None

    @overrides
    def on_train_epoch_end(self, trainer, pl_module):
        relevant_metrics = {k: v for k, v in trainer.logged_metrics.items()
                            if not (k.endswith("_epoch") or k.endswith("_step"))}
        non_val_metrics = list(filter(lambda name: not name.startswith("val_"), relevant_metrics.keys()))
        if self.history is None:
            self.history = {metric: {"train": [], "val": []} for metric in non_val_metrics}

        for metric_name in non_val_metrics:
            if metric_name not in self.history:
                logger.warning(f"Metric '{metric_name}' not in original metrics, probably added afterwards. Skipping")
                continue
            self.history[metric_name]["train"].append(relevant_metrics[metric_name].item())
            val_number = relevant_metrics[f"val_{metric_name}"].item() if trainer.enable_validation else None
            self.history[metric_name]["val"].append(val_number)

            out_file = f"{trainer.loggers[0].log_dir}/{metric_name}.png"
            self._do_plot(pl_module, metric_name, out_file)

    @overrides
    def state_dict(self) -> Dict[str, Any]:
        return {"history": self.history}

    @overrides
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.history = state_dict["history"]
