import functools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config.config import configurable
from detectron2.layers import nonzero_tuple
from detectron2.utils.registry import Registry

from .sampling import random_choice

__all__ = ["MultiPosCrossEntropy", "build_track_loss", "ROI_TRACK_LOSS_REGISTRY"]

ROI_TRACK_LOSS_REGISTRY = Registry("ROI_TRACK_LOSS")


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / max(1, len(loss))
    elif reduction_enum == 2:
        return loss.sum()


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / max(1, avg_factor)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


@weighted_loss
def l2_loss(pred, target):
    """L2 loss.
    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)**2
    return loss


@ROI_TRACK_LOSS_REGISTRY.register()
class MultiPosCrossEntropy(nn.Module):
    @configurable
    def __init__(self, loss_weight, reduction):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction

    @classmethod
    def from_config(cls, cfg):
        return {
            "loss_weight": cfg.MODEL.QDTRACK.ROI_TRACK_LOSS.WEIGHT,
            "reduction": "mean",    # TODO
        }

    def forward(self, pred, label, avg_factor=None):
        # a more numerical stable implementation.
        pos_inds = (label == 1)
        neg_inds = (label == 0)
        pred_pos = pred * pos_inds.float()
        pred_neg = pred * neg_inds.float()
        # use -inf to mask out unwanted elements.
        pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
        pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

        _pos_expand = pred_pos[:, :, None]
        _neg_expand = pred_neg[:, None, :]
        x = torch.nn.functional.pad((_neg_expand - _pos_expand).flatten(1), (0, 1), "constant", 0)
        loss = torch.logsumexp(x, dim=1)

        loss = weight_reduce_loss(
            loss, reduction=self.reduction, avg_factor=avg_factor)

        return self.loss_weight * loss


@ROI_TRACK_LOSS_REGISTRY.register()
class L2Loss(nn.Module):
    @configurable
    def __init__(self, loss_weight, reduction, pos_margin, neg_margin, hard_mining, neg_pos_ratio):
        super().__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction

        self.pos_margin = pos_margin
        self.neg_margin = neg_margin

        self.hard_mining = hard_mining
        self.neg_pos_ratio = neg_pos_ratio

    @classmethod
    def from_config(cls, cfg):
        return {
            "loss_weight": cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.WEIGHT,
            "reduction": "mean",    # TODO
            "pos_margin": cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.POS_MARGIN,
            "neg_margin": cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.NEG_MARGIN,
            "hard_mining": cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.HARD_MINING,
            "neg_pos_ratio": cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.NEG_POS_RATIO,
        }

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
    ):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
        """
        pred, weight, avg_factor = self.update_weight(pred, target, weight,
                                                      avg_factor)
        loss_bbox = self.loss_weight * l2_loss(
            pred, target, weight, reduction=self.reduction, avg_factor=avg_factor)
        return loss_bbox

    def update_weight(self, pred, target, weight, avg_factor):
        if weight is None:
            weight = target.new_ones(target.size())
        pos_inds = target == 1
        neg_inds = target == 0

        if self.pos_margin > 0:
            pred[pos_inds] -= self.pos_margin
        if self.neg_margin > 0:
            pred[neg_inds] -= self.neg_margin
        pred = torch.clamp(pred, min=0, max=1)

        num_pos = int(pos_inds.sum().item())
        num_neg = int(neg_inds.sum().item())
        if self.neg_pos_ratio > 0 and num_neg / max(1, num_pos) > self.neg_pos_ratio:
            num_neg = num_pos * self.neg_pos_ratio
            neg_idx = nonzero_tuple(neg_inds)

            if self.hard_mining:
                costs = l2_loss(pred, target.float(), reduction='none')[neg_idx[0], neg_idx[1]].detach()
                samp_idx = costs.topk(int(num_neg))[1]
            else:
                samp_idx = random_choice(np.arange(len(neg_idx[0])), num_neg)
            neg_idx = (neg_idx[0][samp_idx], neg_idx[1][samp_idx])

            new_neg_inds = neg_inds.new_zeros(neg_inds.size()).bool()
            new_neg_inds[neg_idx[0], neg_idx[1]] = True

            invalid_neg_inds = torch.logical_xor(neg_inds, new_neg_inds)
            weight[invalid_neg_inds] = 0.0

        avg_factor = (weight > 0).sum()
        return pred, weight, avg_factor


def build_track_loss(cfg, name):
    """
    Build a track loss defined by `cfg.MODEL.QDTRACK.ROI_TRACK_LOSS.NAME`.
    """
    return ROI_TRACK_LOSS_REGISTRY.get(name)(cfg)
