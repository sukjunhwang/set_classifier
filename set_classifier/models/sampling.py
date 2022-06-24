from random import random
import numpy as np
import torch

from detectron2.layers import nonzero_tuple

__all__ = ["subsample_labels_for_track"]


def random_choice(gallery, num):
    assert len(gallery) >= num

    is_tensor = isinstance(gallery, torch.Tensor)
    if not is_tensor:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = 'cpu'
        gallery = torch.tensor(gallery, dtype=torch.long, device=device)
    perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
    rand_inds = gallery[perm]
    if not is_tensor:
        rand_inds = rand_inds.cpu().numpy()
    return rand_inds


def _subsample_positive_labels(
    gt_ids: torch.Tensor, pos_idxs: torch.Tensor, num_pos_samples: int
):
    if pos_idxs.numel() <= num_pos_samples:
        return pos_idxs

    unique_gt_ids = gt_ids[pos_idxs].unique()
    num_gts = len(unique_gt_ids)
    num_per_gt = int(round(num_pos_samples / float(num_gts)) + 1)
    sampled_inds = []
    for i in unique_gt_ids:
        inds = nonzero_tuple(gt_ids == i.item())[0]
        if inds.numel() == 0:
            continue
        if len(inds) > num_per_gt:
            inds = random_choice(inds, num_per_gt)
        sampled_inds.append(inds)
    sampled_inds = torch.cat(sampled_inds)
    if len(sampled_inds) < num_pos_samples:
        num_extra = num_pos_samples - len(sampled_inds)
        extra_inds = np.array(list(set(pos_idxs.cpu()) - set(sampled_inds.cpu())))
        if len(extra_inds) > num_extra:
            extra_inds = random_choice(extra_inds, num_extra)
        extra_inds = torch.from_numpy(extra_inds).to(gt_ids.device).long()
        sampled_inds = torch.cat([sampled_inds, extra_inds])
    elif len(sampled_inds) > num_pos_samples:
        sampled_inds = random_choice(sampled_inds, num_pos_samples)
    return sampled_inds


def _subsample_negative_labels(
    gt_ids: torch.Tensor, neg_idxs: torch.Tensor, num_neg_samples: int
):
    if len(neg_idxs) <= num_neg_samples:
        return neg_idxs
    else:
        return random_choice(neg_idxs, num_neg_samples)


def subsample_labels_for_track(
    gt_ids: torch.Tensor, matched_labels: torch.Tensor,
    num_samples: int, positive_fraction: float, neg_pos_ratio: float,
):
    pos_idxs = nonzero_tuple(matched_labels == 1)[0]
    neg_idxs = nonzero_tuple(matched_labels == 0)[0]

    num_expected_pos = int(num_samples * positive_fraction)
    sampled_pos_idxs = _subsample_positive_labels(gt_ids, pos_idxs, num_expected_pos)
    # We found that sampled indices have duplicated items occasionally.
    # (may be a bug of PyTorch)
    sampled_pos_idxs = sampled_pos_idxs.unique()

    num_sampled_pos = sampled_pos_idxs.numel()
    num_expected_neg = num_samples - num_sampled_pos
    if neg_pos_ratio >= 0:
        neg_upper_bound = int(neg_pos_ratio * max(1, num_sampled_pos))
        if num_expected_neg > neg_upper_bound:
            num_expected_neg = neg_upper_bound
    sampled_neg_idxs = _subsample_negative_labels(gt_ids, neg_idxs, num_expected_neg)
    sampled_neg_idxs = sampled_neg_idxs.unique()

    return sampled_pos_idxs, sampled_neg_idxs
