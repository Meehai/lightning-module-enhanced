"""ReduceLROnPlateauWithBurnIn scheduler"""
from torch.optim.lr_scheduler import ReduceLROnPlateau
from overrides import overrides

from ..logger import logger

class ReduceLROnPlateauWithBurnIn(ReduceLROnPlateau):
    """Reduce LR on Plateau with Burn-in epochs. Same as ReduceLROnPateau, but skips N epochs for burn-in."""
    def __init__(self, *args, burn_in_epochs: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.burn_in_epochs = burn_in_epochs
        assert burn_in_epochs >= 1

    @overrides
    def step(self, metrics, epoch=None):
        assert epoch is None
        if self.last_epoch < self.burn_in_epochs:
            logger.debug2(f"Epoch {self.last_epoch} is less than burn in epoch {self.burn_in_epochs}. Returning early")
            self.last_epoch += 1
            return
        super().step(metrics, epoch)
