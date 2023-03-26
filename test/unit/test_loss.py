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
    assert tr.allclose(loss.mean(), batch_weighted_ce(y, gt))

def test_batch_weighted_ce_3():
    y = tr.randn(200, 300, 1)
    gt = F.one_hot(tr.randint(0, 2, size=(200, 300)), num_classes=8).type(tr.float)
    try:
        loss = batch_weighted_ce(y, gt, reduction="none")
        raise ValueError("shouldn't reach here")
    except AssertionError:
        pass

def test_batch_weighted_bce_1():
    y = tr.randn(100, 1).sigmoid()
    gt = tr.randint(0, 2, (100, 1), dtype=tr.float32)
    loss = batch_weighted_bce(y, gt)
    assert len(loss.shape) == 0 and loss > 0

def test_batch_weighted_bce_2():
    y = tr.randn(100, 200, 1).sigmoid()
    gt = tr.randint(0, 2, (100, 200, 1), dtype=tr.float32)
    loss = batch_weighted_bce(y, gt, reduction="none")
    assert loss.shape == y.shape
    assert tr.allclose(loss.mean(), batch_weighted_bce(y, gt))
