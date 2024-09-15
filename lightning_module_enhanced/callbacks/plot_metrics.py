"""Module to plots metrics"""
from __future__ import annotations
from typing import Any
from pathlib import Path
from overrides import overrides
import pandas as pd
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import matplotlib.pyplot as plt
import numpy as np
from ..logger import lme_logger as logger

def _norm(x, metric_name: str):
    x = np.array(x)
    assert not np.isnan(x).any(), f"You have NaNs in your metric ({metric_name}): {x}"
    median = np.median(x)
    return x.clip(-2 * np.sign(median) * median, 2 * np.sign(median) * median)

class PlotMetrics(Callback):
    """Plot metrics implementation"""
    def _plot_best_dot(self, ax: plt.Axes, scores: pd.Series, higher_is_better: bool):
        """Plot the dot. We require to know if the metric is max or min typed."""
        metric_x = np.argmax(scores) if higher_is_better else np.argmin(scores)
        metric_y = scores[metric_x]
        ax.annotate(f"Epoch {metric_x + 1}\nMax {metric_y:.2f}", xy=(metric_x + 1, metric_y))
        ax.plot([metric_x + 1], [metric_y], "o")

    def _do_plot(self, pl_module: Any, df: pd.DataFrame, metric_name: str, out_file: str):
        """Plot the figure with the metric"""
        ax = (fig := plt.figure()).gca()
        x_plot = range(1, len(df) + 1)
        higher_is_better = pl_module.metrics[metric_name].higher_is_better if metric_name != "loss" else False
        ax.plot(x_plot, _norm((train_y := df[metric_name]), metric_name), label="train")
        if (val_y := df.get(f"val_{metric_name}")) is not None:
            ax.plot(x_plot, _norm(val_y, metric_name), label="validation")
        self._plot_best_dot(ax, train_y if val_y is None else val_y, higher_is_better)
        ax.set_xlabel("Epoch")
        name_trimmed = metric_name if len(metric_name) < 35 else f"{metric_name[0: 25]}...{metric_name[-7:]}"
        ax.set_title(f"{name_trimmed}({'↑' if higher_is_better else '↓'})")
        fig.legend()
        fig.savefig(out_file)
        plt.close(fig)

    @rank_zero_only
    @overrides
    def on_train_epoch_start(self, trainer: Trainer, pl_module: Any):
        assert any(isinstance(logger, CSVLogger) for logger in trainer.loggers), trainer.loggers
        if not Path(f"{trainer.log_dir}/metrics.csv").exists():
            logger.debug(f"No metrics.csv found in log dir: '{trainer.log_dir}'. Skipping this epoch")
            return
        df = pd.read_csv(f"{trainer.log_dir}/metrics.csv")
        expected_metrics: list[str] = [*list(pl_module.metrics.keys()), "loss"]
        for metric_name in expected_metrics:
            if metric_name not in df:
                logger.warning(f"'{metric_name}' not in {list(df.columns)}")
                continue
            self._do_plot(pl_module, df, metric_name, out_file=f"{trainer.log_dir}/{metric_name}.png")
