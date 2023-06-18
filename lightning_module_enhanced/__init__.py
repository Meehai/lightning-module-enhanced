"""Init module"""
import logging

# pylint: disable=reimported
from .core_module import CoreModule as LME, CoreModule as LightningModuleEnhanced
from .trainable_module import TrainableModule
from .train_setup import TrainSetup

# disable seed messages from pytorch lightning
logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.CRITICAL)
