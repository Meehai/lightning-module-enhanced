"""Module to plots metrics"""
from __future__ import annotations
from typing import Any
from pathlib import Path
import csv
from overrides import overrides
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
import numpy as np

from ..logger import lme_logger as logger

def _norm(x, metric_name: str):
    x = np.array(x)
    assert not np.isnan(x).any(), f"You have NaNs in your metric ({metric_name}): {x}"
    median = np.median(x)
    return x.clip(-2 * np.sign(median) * median, 2 * np.sign(median) * median)

class PlotMetrics(pl.Callback):
    """Plot metrics implementation"""
    def __init__(self):
        self.log_dir = None

    def _plot_best_dot(self, ax: plt.Axes, scores: list[float], higher_is_better: bool):
        """Plot the dot. We require to know if the metric is max or min typed."""
        metric_x = np.argmax(scores) if higher_is_better else np.argmin(scores)
        metric_y = scores[metric_x]
        ax.annotate(f"Epoch {metric_x + 1}\nMax {metric_y:.2f}", xy=(metric_x + 1, metric_y))
        ax.plot([metric_x + 1], [metric_y], "o")

    def _do_plot(self, pl_module: Any, csv_data: list[dict[str, float]], metric_name: str, out_file: str):
        """Plot the figure with the metric"""
        ax = (fig := plt.figure()).gca()
        x_plot = range(1, len(csv_data) + 1)
        higher_is_better = pl_module.metrics[metric_name].higher_is_better if metric_name != "loss" else False
        train_y = [row[metric_name] for row in csv_data]
        val_y = [row[f"val_{metric_name}"] for row in csv_data] if f"val_{metric_name}" in csv_data[0].keys() else None
        ax.plot(x_plot, _norm(train_y, metric_name), label="train")
        if val_y is not None:
            ax.plot(x_plot, _norm(val_y, metric_name), label="validation")
        self._plot_best_dot(ax, train_y if val_y is None else val_y, higher_is_better)
        ax.set_xlabel("Epoch")
        name_trimmed = metric_name if len(metric_name) < 35 else f"{metric_name[0: 25]}...{metric_name[-7:]}"
        ax.set_title(f"{name_trimmed}({'↑' if higher_is_better else '↓'})")
        fig.legend()
        fig.savefig(out_file)
        plt.close(fig)

    @overrides
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert any(isinstance(logger, CSVLogger) for logger in trainer.loggers), trainer.loggers
        if self.log_dir is None: # cache it at epoch 1 before it hangs. TODO: check why it hangs and make minimal repro
            self.log_dir = trainer.log_dir # IF I ACCESS TRAINER IN THE METHOD BELOW ON DDP IT HANGS FOR NO REASON ?!

    @rank_zero_only
    @overrides
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: Any):
        if not Path(f"{self.log_dir}/metrics.csv").exists():
            logger.debug(f"No metrics.csv found in log dir: '{self.log_dir}'. Skipping this epoch")
            return
        csv_data = [{k: float(v) for k, v in row.items()}
                    for row in csv.DictReader(open(f"{self.log_dir}/metrics.csv"))]
        expected_metrics: list[str] = [*list(pl_module.metrics.keys()), "loss"]
        for metric_name in expected_metrics:
            if len(csv_data) == 0 or metric_name not in csv_data[0]:
                logger.debug(f"'{metric_name}' not in {list(csv_data[0])}")
                continue
            self._do_plot(pl_module, csv_data, metric_name, out_file=f"{self.log_dir}/{metric_name}.png")
