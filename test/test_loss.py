from lightning_module_enhanced.loss import batch_weighted_ce, batch_weighted_bce
import torch as tr
from torch.nn import functional as F


def test_batch_weighted_ce_1():
    y = tr.randn(200, 300, 8)
    gt = F.one_hot(tr.randint(0, 7, size=(200, 300)), num_classes=8).type(tr.float)
    loss = batch_weighted_ce(y, gt)
    assert len(loss.shape) == 0 and loss > 0


def test_batch_weighted_ce_2():
    y = tr.randn(200, 300, 8)
    gt = F.one_hot(tr.randint(0, 7, size=(200, 300)), num_classes=8).type(tr.float)
    loss = batch_weighted_ce(y, gt, reduction="none")
    assert len(loss.shape) == 2 and loss.shape == y.shape[0:2]
    assert loss.mean() == batch_weighted_ce(y, gt)
