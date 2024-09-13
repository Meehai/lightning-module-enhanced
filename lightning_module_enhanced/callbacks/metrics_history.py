"""MetricsHistory callback. Keeps track of the metrics throughout a train run"""
from __future__ import annotations
from typing import Any
from overrides import overrides
import pytorch_lightning as pl

from ..logger import lme_logger as logger
from ..metrics import CoreMetric

# TODO: remove the call in active_run to metadata_callback and use this callback in there too for metrics tracking
# TODO: make this store both reduced metrics and non-reduced metrics [as 2 separate dicts]
class MetricsHistory(pl.Callback):
    """MetricsHistory callback implementation"""
    def __init__(self):
        self.history: dict[str, dict[str, list[float]]] = None
        self.expected_metrics = []
        self.higher_is_better: dict[str, bool] = None

    def _setup_metrics(self, pl_module: "LME"):
        self.expected_metrics = [*list(pl_module.metrics.keys()), "loss"]
        if self.history is not None:
            assert len(self.history) > 0
            for k in self.history.keys():
                assert len(self.history[k]["train"]) > 0
            assert (a := set(self.expected_metrics)) == (b := set(self.history.keys())), (a, b)
        # see test_metrics_history_2 as to why we reset here
        if pl_module.trainer.enable_validation: # TODO: what about fine-tuning where previous train was w/o val?
            self.history = {metric_name: {"train": [], "val": []} for metric_name in self.expected_metrics}
        else:
            self.history = {metric_name: {"train": []} for metric_name in self.expected_metrics}
        self.higher_is_better = {name: metric.higher_is_better for name, metric in pl_module.metrics.items()}
        self.higher_is_better["loss"] = False

    @overrides
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: "LME"):
        if trainer.current_epoch == 0:
            self._setup_metrics(pl_module)
        assert self.history is not None, "self.history is None: on_train_epoch_end somehow called before on_fit_start."

        for metric_name in self.expected_metrics:
            if metric_name not in self.history:
                logger.debug(f"Metric '{metric_name}' not in original metrics, probably added afterwards. Skipping")
                continue
            metric: CoreMetric = pl_module._active_run_metrics[""][metric_name] # pylint: disable=protected-access
            metric_score = metric.epoch_result_reduced(metric.epoch_result())
            if metric_score is None:
                logger.debug2(f"Metric '{metric_name}' cannot be reduced to a single number. Skipping")
                continue
            self.history[metric_name]["train"].append(metric_score.item())

            if trainer.enable_validation:
                val_metric: CoreMetric = pl_module._active_run_metrics["val_"][metric_name] # pylint: disable=all
                val_metric_score = val_metric.epoch_result_reduced(val_metric.epoch_result())
                if "val" not in self.history[metric_name]:
                    logger.debug(f"Patching validation for '{metric_name}'. Possibly fine-tuning an old non-val model")
                    self.history[metric_name]["val"] = [None] * (len(self.history[metric_name]["train"]) - 1)
                self.history[metric_name]["val"].append(val_metric_score.item())
                assert len(self.history[metric_name]["val"]) == len(self.history[metric_name]["train"])

    @overrides
    def state_dict(self) -> dict[str, Any]:
        return {"history": self.history, "expected_metrics": self.expected_metrics,
                "higher_is_better": self.higher_is_better}

    @overrides
    def load_state_dict(self, state_dict: dict[str, Any]):
        self.history = state_dict["history"]
        self.expected_metrics = state_dict["expected_metrics"]
        self.higher_is_better = state_dict["higher_is_better"]

    def __repr__(self):
        return f"""[MetricsHistory]. { {k: f"{len(v['train'])} entries" for k, v in self.history.items()} }"""

    def __getitem__(self, key: str):
        return self.history[key]
