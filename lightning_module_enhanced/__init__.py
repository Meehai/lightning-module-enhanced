"""Init module"""
import logging
import warnings

# pylint: disable=reimported
from .lme import LightningModuleEnhanced as LME, LightningModuleEnhanced as LightningModuleEnhanced
from .trainable_module import TrainableModule
from .train_setup import TrainSetup

# disable seed messages from pytorch lightning
logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric")
