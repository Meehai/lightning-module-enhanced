"""loss functions module for various losses that are general but not supported in other frameworks"""
import torch as tr
import numpy as np
from torch.nn import functional as F


def batch_weighted_ce(y: tr.Tensor, gt: tr.Tensor, reduction: str = "mean", **kwargs) -> tr.Tensor:
    """
    Batch-weighted cross-entropy loss. Uses same inputs as F.ce() but flattens first. Takes care of nans/infs as well.
    There should be no way all the weights are zero. At the end, we multiply by the number of classes s.t. the final
    number is in the same range as F.ce(). F.ce() can be thought as F.ce(y, gt, tr.ones(C)). Without doing that,
    we end up with F.ce(y, gt, tr.ones(C)/C). We also subtract from C the number of classes with 0 values in the batch
    """
    assert gt.dtype in (tr.long, tr.float), f"GT must be float or index tesor (long). Got: {gt.dtype}"
    assert y.dtype == tr.float, f"Predictions must only be float. Got: {y.dtype}"
    C = y.shape[-1]
    assert C > 1, f"At least two classes required for cross entropy. Got: {C}. y: {y.shape}. gt: {gt.shape}"
    if gt.dtype == tr.long:
        sum_classes = tr.zeros(C, dtype=tr.long)
        gt_flat = gt.reshape(-1)
        ix, counts = gt.unique(return_counts=True)
        sum_classes[ix] = counts
    else:
        gt_flat = gt.reshape(-1, C)
        sum_classes = gt_flat.sum(dim=0)

    y_flat = y.reshape(-1, C)
    n_invalid = (sum_classes == 0).sum()
    denom = (1 / sum_classes).nan_to_num(0, 0, 0)
    batch_weights = denom / denom.sum()
    batch_weights = batch_weights * (C - n_invalid)
    loss = F.cross_entropy(y_flat, gt_flat, weight=batch_weights, reduction=reduction, **kwargs)
    if reduction == "none":
        loss = loss.reshape(y.shape[0:-1])
    return loss


def batch_weighted_bce(y: tr.Tensor, gt: tr.Tensor, reduction: str = "mean", **kwargs) -> tr.Tensor:
    """Batch-weighted cross-entropy loss. Must have same shape and y must be AFTER sigmoid, so [0:1] range"""
    assert y.shape == gt.shape, f"y shape: {y.shape} vs gt shape: {gt.shape}"
    pos_weight = (gt != 0).sum() / np.prod(gt.shape)
    batch_weights = (gt == 0) * pos_weight + (gt != 0) * (1 - pos_weight)
    res = F.binary_cross_entropy(y, gt, weight=batch_weights, reduction=reduction, **kwargs)
    return res
