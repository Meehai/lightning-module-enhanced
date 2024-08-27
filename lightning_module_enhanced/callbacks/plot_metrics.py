"""Module to plots metrics"""
from __future__ import annotations
from typing import Any
from overrides import overrides
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
import matplotlib.pyplot as plt
import numpy as np
from ..logger import lme_logger as logger
from .metrics_history import MetricsHistory

def _norm(x, metric_name: str):
    x = np.array(x)
    assert not np.isnan(x).any(), f"You have NaNs in your metric ({metric_name}): {x}"
    median = np.median(x)
    return x.clip(-2 * np.sign(median) * median, 2 * np.sign(median) * median)

class PlotMetrics(Callback):
    """Plot metrics implementation"""
    def _plot_best_dot(self, ax: plt.Axes, pl_module: Any, metric_name: str, higher_is_better: bool):
        """Plot the dot. We require to know if the metric is max or min typed."""
        metric_history = pl_module.metrics_history.history[metric_name]
        if len(metric_history["train"]) == 0:
            logger.debug2(f"No metrics yet for '{metric_name}'")
            return
        scores = metric_history["val"] if metric_history["val"][0] is not None else metric_history["train"]
        metric_x = np.argmax(scores) if higher_is_better else np.argmin(scores)
        metric_y = scores[metric_x]
        ax.annotate(f"Epoch {metric_x + 1}\nMax {metric_y:.2f}", xy=(metric_x + 1, metric_y))
        ax.plot([metric_x + 1], [metric_y], "o")

    def _do_plot(self, pl_module: Any, metric_name: str, out_file: str):
        """Plot the figure with the metric"""
        fig = plt.figure()
        ax = fig.gca()
        metrics_history: MetricsHistory = pl_module.metrics_history
        metric_history = metrics_history.history[metric_name]
        x_plot = np.arange(len(metric_history["train"])) + 1
        ax.plot(x_plot, _norm(metric_history["train"], metric_name), label="train")
        if None not in metric_history["val"]:
            ax.plot(x_plot, _norm(metric_history["val"], metric_name), label="validation")
        self._plot_best_dot(ax, pl_module, metric_name, metrics_history.higher_is_better[metric_name])
        ax.set_xlabel("Epoch")
        name_trimmed = metric_name if len(metric_name) < 35 else f"{metric_name[0: 25]}...{metric_name[-7:]}"
        ax.set_title(f"{name_trimmed}({'↑' if metrics_history.higher_is_better[metric_name] else '↓'})")
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
