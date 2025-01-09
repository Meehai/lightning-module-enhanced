"""MinMax Learning rate linear scheduler"""
# pylint: disable=too-many-positional-arguments
import torch as tr
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

class CyclesListLR(LRScheduler):
    """A list of cycles [l, r] and their number of epochs/steps."""
    def __init__(self, optimizer: Optimizer, cycles: list[tuple[float, float, int]]):
        assert len(cycles) > 0 and all(len(x) == 3 for x in cycles), cycles
        all_lrs = []
        for l, r, steps in cycles:
            cycle_lrs = tr.linspace(l, r, steps + 1)[1:].tolist()
            all_lrs.extend(cycle_lrs)
        self.lr_cycles = [round(x, 6) for x in all_lrs]
        assert all(0 < x < 1 for x in self.lr_cycles), self.lr_cycles
        super().__init__(optimizer)

    def get_lr(self):
        lr = self.lr_cycles[self._step_count % len(self.lr_cycles)]
        return [lr for _ in self.optimizer.param_groups]
