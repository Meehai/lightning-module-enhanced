"""Module to plots metrics"""
from __future__ import annotations
from typing import Any
from overrides import overrides
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import numpy as np
from ..logger import lme_logger as logger

def _norm(x):
    return np.clip(x, -2 * np.sign(np.median(x)) * np.median(x), 2 * np.sign(np.median(x)) * np.median(x))

class PlotMetrics(Callback):
    """Plot metrics implementation"""
    def _plot_best_dot(self, ax: plt.Axes, pl_module: Any, metric_name: str):
        """Plot the dot. We require to know if the metric is max or min typed."""
        metric = pl_module.metrics[metric_name] if metric_name != "loss" else pl_module.criterion_fn
        metric_history = pl_module.metrics_history.history[metric_name]
        scores = metric_history["val"] if metric_history["val"][0] is not None else metric_history["train"]
        metric_x = np.argmax(scores) if metric.higher_is_better else np.argmin(scores)
        metric_y = scores[metric_x]
        ax.annotate(f"Epoch {metric_x + 1}\nMax {metric_y:.2f}", xy=(metric_x + 1, metric_y))
        ax.plot([metric_x + 1], [metric_y], "o")

    def _do_plot(self, pl_module: Any, metric_name: str, out_file: str):
        """Plot the figure with the metric"""
        fig = plt.figure()
        ax = fig.gca()
        metric_history = pl_module.metrics_history.history[metric_name]
        _range = range(1, len(metric_history["train"]) + 1)
        ax.plot(_range, _norm(metric_history["train"]), label="train")
        if None not in metric_history["val"]:
            ax.plot(_range, _norm(metric_history["val"]), label="validation")
        self._plot_best_dot(ax, pl_module, metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        fig.legend()
        fig.savefig(out_file)
        plt.close(fig)

    @overrides
    def on_train_epoch_end(self, trainer: Trainer, pl_module: Any):
        if len(trainer.loggers) == 0:
            logger.debug("No lightning logger found. Not calling PlotMetrics()")
            return

        expected_metrics: list[str] = [*list(pl_module.metrics.keys()), "loss"]
        for metric_name in expected_metrics:
            out_file = f"{trainer.loggers[0].log_dir}/{metric_name}.png"
            self._do_plot(pl_module, metric_name, out_file)
