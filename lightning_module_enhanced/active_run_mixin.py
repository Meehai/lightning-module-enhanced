"""ActiveRunMixin. Helper class to track stuff during runs (training, testing, predicting)"""
from __future__ import annotations
from copy import deepcopy
import torch as tr
from torch import nn
from .metrics import CoreMetric
from .utils import tr_detach_data

# pylint: disable=abstract-method
class ActiveRunMixin(nn.Module):
    """
    Helper class to keep track of properties that are updated during a run started via Trainer.fit, Trainer.test or
    Trainer.predict.
    Note: "loss" keys is always included here, as it's supposed that any Trainer active run must have a loss function.
    """
    def __init__(self):
        super().__init__()
        # Updated during the epochs of an actieve run (i.e. Trainer.fit, Trainer.test or Trainer.predict).
        self._active_run_metrics: dict[str, dict[str, CoreMetric]] = {}

    def _setup_active_metrics(self):
        """sets up self.active_run_metrics based on metrics for this train run. Called at on_fit_start."""
        assert len(self._active_run_metrics) == 0, "TODO: add breakpoint here to understand if/where it's hit"
        self._active_run_metrics = {"": {"loss": self.criterion_fn, **self.metrics}}
        if hasattr(self, "trainer") and self.trainer.enable_validation:
            self._active_run_metrics["val_"] = deepcopy(self._active_run_metrics[""])

    def _reset_all_active_metrics(self):
        """ran at epoch end to reset the metrics"""
        for prefix in self._active_run_metrics.keys():
            for metric in self._active_run_metrics[prefix].values():
                metric.reset()

    def _set_metrics_running_model(self):
        """ran at fit/test start to set the running model"""
        for prefix in self._active_run_metrics.keys():
            for metric in self._active_run_metrics[prefix].values():
                metric.running_model = lambda: self

    def _unset_metrics_running_model(self):
        """ran at fit/test end to unset the running model"""
        for prefix in self._active_run_metrics.keys():
            for metric in self._active_run_metrics[prefix].values():
                metric.running_model = None
        self._active_run_metrics = {}

    def _update_metrics_at_batch_end(self, batch_results: dict[str, tr.Tensor | None]):
        assert isinstance(batch_results, dict), f"Expected dict, got {type(batch_results)}"
        batch_results_detach: dict[str, tr.Tensor | None] = tr_detach_data(batch_results)
        run_metrics: dict[str, CoreMetric] = self._active_run_metrics[self._prefix_from_trainer()] # .metrics no loss

        if (bres := set(batch_results.keys())) != (expected_metrics := set(run_metrics.keys())):
            raise ValueError(f"Expected metrics: {sorted(expected_metrics)} vs. this batch: {sorted(bres)}")

        for metric_name, metric in run_metrics.items():
            metric.batch_update(batch_results_detach[metric_name])

    def _run_and_log_metrics_at_epoch_end(self):
        """Runs and logs a given list of logged metrics. Assume they all exist in self.metrics"""
        all_prefixes = self._active_run_metrics.keys()
        metrics_to_log = list(self._active_run_metrics[""].keys())
        for metric_name in metrics_to_log:
            for prefix in all_prefixes:
                metric_fn: CoreMetric = self._active_run_metrics[prefix][metric_name]
                metric_epoch_result = metric_fn.epoch_result()
                # Log the metric at the end of the epoch. Only log on pbar the val_loss, loss is tracked by default
                prog_bar = (metric_name == "loss" and prefix == "val_")

                value_reduced = metric_fn.epoch_result_reduced(metric_epoch_result)
                if value_reduced is not None:
                    assert (a := value_reduced.device) == (b := self.device), f"Expected {a}, got {b} ({metric_name})"
                    self.log(f"{prefix}{metric_name}", value_reduced, prog_bar=prog_bar, on_epoch=True)
                # Call the metadata callback for the full result, since it can handle any sort of metrics
                if self.trainer.global_rank == 0 and metric_epoch_result is not None: # TODO: remove this
                    self.metadata_callback.log_epoch_metric(metric_name, metric_epoch_result,
                                                            self.trainer.current_epoch, prefix)
