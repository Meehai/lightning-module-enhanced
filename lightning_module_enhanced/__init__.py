"""Init module"""
import logging
import warnings
from lightning_fabric.utilities.warnings import PossibleUserWarning

# pylint: disable=reimported
from .lme import LightningModuleEnhanced as LME, ModelAlgorithmOutput
from .trainable_module import TrainableModule

# disable seed messages from pytorch lightning
logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric")
warnings.filterwarnings("ignore", ".*does not have many workers.*")
