"""Warmup cosine Scheduler"""
# pylint: disable=too-many-positional-arguments
import torch as tr
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

class MinMaxLR(LRScheduler):
    """Linearly interpolates between optimizer's LR and +/- delta LR in n_steps (epochs)."""
    def __init__(self, optimizer: Optimizer, min_lr: float, max_lr: int, n_steps: int, warmup_steps: int):
        assert n_steps >= 2
        self.warmup_steps = warmup_steps
        self.n_steps = n_steps
        group_lrs = [group["lr"] for group in optimizer.param_groups]
        assert all(lr == group_lrs[0] for lr in group_lrs), group_lrs
        self.initial_lr = group_lrs[0]
        assert min_lr < self.initial_lr < max_lr, (min_lr, self.initial_lr, max_lr)
        p1 = tr.linspace(self.initial_lr, min_lr, n_steps + 1)[1:].tolist()
        p2 = tr.linspace(min_lr, self.initial_lr, n_steps + 1)[1:].tolist()
        p3 = tr.linspace(self.initial_lr, max_lr, n_steps + 1)[1:].tolist()
        p4 = tr.linspace(max_lr, self.initial_lr, n_steps + 1)[1:].tolist()
        assert len(p1) > 0 and len(p2) > 0 and len(p3) > 0 and len(p4) > 0, (p1, p2, p3, p4)
        self.lr_cycles = [round(x, 6) for x in [*p1, *p2, *p3, *p4]]
        assert all(0 < x < 1 for x in self.lr_cycles), self.lr_cycles
        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            return [self.initial_lr for _ in self.optimizer.param_groups]
        ix = self._step_count - self.warmup_steps - 1
        lr = self.lr_cycles[ix % len(self.lr_cycles)]
        return [lr for _ in self.optimizer.param_groups]
