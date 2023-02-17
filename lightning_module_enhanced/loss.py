"""loss functions module for various losses that are general but not supported in other frameworks"""
import torch as tr
from torch.nn import functional as F


def batch_weighted_ce(y: tr.Tensor, gt: tr.Tensor, reduction: str = "mean", **kwargs) -> tr.Tensor:
    """
    Batch-weighted cross-entropy loss. Uses same inputs as F.ce() but flattens first. Takes care of nans/infs as well.
    There should be no way all the weights are zero. At the end, we multiply by the number of classes s.t. the final
    number is in the same range as F.ce(). F.ce() can be thought as F.ce(y, gt, tr.ones(C)). Without doing that,
    we end up with F.ce(y, gt, tr.ones(C)/C). We also subtract from C the number of classes with 0 values in the batch
    """
    C = y.shape[-1]
    y_flat = y.reshape(-1, C)
    gt_flat = gt.reshape(-1, C)
    denom = 1 / gt_flat.sum(dim=0)
    finite_mask = tr.isfinite(denom)
    batch_weights = denom / denom[finite_mask].sum()
    batch_weights[~finite_mask] = 0
    batch_weights *= C - (~finite_mask).sum()
    loss = F.cross_entropy(y_flat, gt_flat, weight=batch_weights, reduction=reduction, **kwargs)
    if reduction == "none":
        loss = loss.reshape(y.shape[0:-1])
    return loss


def batch_weighted_bce(y: tr.Tensor, gt: tr.Tensor) -> tr.Tensor:
    """Batch-weighted cross-entropy loss."""
    # we may need to do some shape update like the one above
    assert False, "TODO"
    pos_weight = (gt != 0).sum() / len(gt)
    batch_weights = (gt == 0) * pos_weight + (gt != 0) * (1 - pos_weight)
    loss = F.binary_cross_entropy(y, gt, reduction="none")
    return (loss * batch_weights).mean()
