"""MetricsHistory callback. Keeps track of the metrics throughout a train run"""
from __future__ import annotations
from typing import Any
from overrides import overrides
import pytorch_lightning as pl

from ..logger import logger

class MetricsHistory(pl.Callback):
    """MetricsHistory callback implementation"""
    def __init__(self):
        self.history: dict[str, list[float]] = None
        self.expected_metrics = []

    @overrides
    def on_fit_start(self, trainer: pl.Trainer, pl_module: Any) -> None:
        # we need to not reset for Trainer().fit(ckpt_path=...) [NGC] or if we reuse the same trainer.
        # We need to check both conditions BECAUSE the trainer's state is updated AFTER this call, but brfore
        # on_train_epoch_start. See here: https://github.com/Lightning-AI/lightning/issues/17712
        self.expected_metrics = [*list(pl_module.metrics.keys()), "loss"]
        if trainer.current_epoch == 0 and trainer.ckpt_path is None:
            self.history = {metric_name: {"train": [], "val": []} for metric_name in self.expected_metrics}

    @overrides
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Any):
        assert self.history is not None, "self.history is None: on_train_epoch_end somehow called before on_fit_start."

        for metric_name in self.expected_metrics:
            if metric_name not in self.history:
                logger.warning(f"Metric '{metric_name}' not in original metrics, probably added afterwards. Skipping")
                continue
            metric = pl_module._active_run_metrics[""][metric_name] # pylint: disable=protected-access
            metric_score = metric.epoch_result_reduced(metric.epoch_result())
            if metric_score is None:
                logger.debug2(f"Metric '{metric_name}' cannot be reduced to a single number. Skipping")
                continue
            self.history[metric_name]["train"].append(metric_score.item())

            if trainer.enable_validation:
                val_metric = pl_module._active_run_metrics["val_"][metric_name] # pylint: disable=protected-access
                val_metric_score = val_metric.epoch_result_reduced(val_metric.epoch_result())
                self.history[metric_name]["val"].append(val_metric_score.item())

    @overrides
    def state_dict(self) -> dict[str, Any]:
        return {"history": self.history, "expected_metrics": self.expected_metrics}

    @overrides
    def load_state_dict(self, state_dict: dict[str, Any]):
        self.history = state_dict["history"]
        self.expected_metrics = state_dict["expected_metrics"]
