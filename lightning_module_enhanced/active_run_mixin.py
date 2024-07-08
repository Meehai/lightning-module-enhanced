"""ActiveRunMixin. Helper class to track stuff during runs (training, testing, predicting)"""
from __future__ import annotations
from copy import deepcopy
from torch import nn
from .metrics import CoreMetric, CallableCoreMetric
from .utils import tr_detach_data

def _make_stub_metric() -> CoreMetric:
    return CallableCoreMetric(metric_fn=lambda y, _: (y - y).mean(), higher_is_better=False)

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

    def _setup_active_metrics(self, metrics: list[str]):
        """sets up self.active_run_metrics based on metrics for this train run. Called at on_fit_start"""
        if len(self.metrics) == 0:
            self.metrics = {metric_name: _make_stub_metric() for metric_name in metrics}

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

    def _active_run_batch_updates(self, batch_results: dict):
        prefix = self._prefix_from_trainer()
        for metric_name, metric in self._active_run_metrics[prefix].items():
            metric.batch_update(tr_detach_data(batch_results[metric_name]))

    def _run_and_log_metrics_at_epoch_end(self):
        """Runs and logs a given list of logged metrics. Assume they all exist in self.metrics"""
        all_prefixes = self._active_run_metrics.keys()
        metrics_to_log = list(self._active_run_metrics[""].keys())
        for metric_name in metrics_to_log:
            for prefix in all_prefixes:
                metric_fn: CoreMetric = self._active_run_metrics[prefix][metric_name]
                # Get the metric's epoch result
                metric_epoch_result = metric_fn.epoch_result()
                # Log the metric at the end of the epoch. Only log on pbar the val_loss, loss is tracked by default
                prog_bar = (metric_name == "loss" and prefix == "val_")

                value_reduced = metric_fn.epoch_result_reduced(metric_epoch_result)
                if value_reduced is not None:
                    self.log(f"{prefix}{metric_name}", value_reduced, prog_bar=prog_bar, on_epoch=True)
                # Call the metadata callback for the full result, since it can handle any sort of metrics
                self.metadata_callback.log_epoch_metric(metric_name, metric_epoch_result,
                                                        self.trainer.current_epoch, prefix)
