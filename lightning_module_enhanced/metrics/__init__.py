"""Init module for metrics"""
from .core_metric import CoreMetric
from .callable_core_metric import CallableCoreMetric
from .stub_metric import StubMetric

from .multi_class_accuracy import MultiClassAccuracy
from .multi_class_confusion_matrix import MultiClassConfusionMatrix
from .multi_class_f1_score import MultiClassF1Score
