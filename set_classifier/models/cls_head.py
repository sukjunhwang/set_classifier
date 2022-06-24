import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.utils.registry import Registry

from detectron2.projects.set_classifier.data.datasets import LVIS_CLS_CNT

from .transformer import SequencePredictor
from .misc import MLP

__all__ = ["build_cls_head", "ROI_CLS_HEAD_REGISTRY"]

ROI_CLS_HEAD_REGISTRY = Registry("ROI_CLS_HEAD")
ROI_CLS_HEAD_REGISTRY.__doc__ = """
Registry for cls heads, which predicts instance representation vectors given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_CLS_HEAD_REGISTRY.register()
class ClsHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, num_classes, channel_size,
        ins_head_on, seq_head_on, include_bg,
        seq_batch_size, seq_length_range, seq_dim,
        num_heads, num_enc_layers,
        cls_ins_weight, cls_pair_weight,
        cls_seq_weight, cls_seq_aux_weight,
        use_cls_cnt
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ins_head_on = ins_head_on
        self.seq_head_on = seq_head_on
        self.include_bg = include_bg

        if self.ins_head_on:
            K = self.num_classes + (1 if self.include_bg else 0)
            self.cls_ins_head = MLP(channel_size, channel_size, K, 1)
            nn.init.normal_(self.cls_ins_head.layers[-1].weight, std=0.01)
            nn.init.constant_(self.cls_ins_head.layers[-1].bias, 0)

        self.seq_batch_size = seq_batch_size
        self.seq_length_range = seq_length_range
        max_min = seq_length_range[1] - seq_length_range[0]
        assert self.seq_batch_size % max_min == 0, \
            "Batch size {} should be divided by seq_length_range {}".format(
                self.seq_batch_size, max_min
            )

        triangle = torch.triu(torch.ones((max_min, max_min)))
        sample_slots = torch.cat(
            (triangle, torch.ones(max_min, seq_length_range[0])), dim=1
        )
        sample_slots = sample_slots.repeat(self.seq_batch_size // max_min, 1)

        self.insert_idx = nonzero_tuple(sample_slots)
        self.sample_size = int(sample_slots.sum().item())

        self.cls_ins_weight = cls_ins_weight
        self.cls_pair_weight = cls_pair_weight
        self.cls_seq_weight = cls_seq_weight
        self.cls_seq_aux_weight = cls_seq_aux_weight
        self.cls_seq_aux_on = (cls_seq_aux_weight > 0.0)

        if self.seq_head_on:
            self.cls_seq_head = SequencePredictor(
                in_channels=channel_size, d_model=seq_dim, out_channels=num_classes,
                nhead=num_heads, num_encoder_layers=num_enc_layers,
                return_seq_ins=(True, self.cls_seq_aux_on),
            )

        self.use_cls_cnt = use_cls_cnt
        if self.use_cls_cnt and self.seq_head_on:
            self.register_buffer(
                'cls_cnt', torch.tensor(LVIS_CLS_CNT, dtype=torch.float),
            )

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "channel_size": cfg.MODEL.ROI_BOX_HEAD.FC_DIM,
            "ins_head_on": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.INS_HEAD_ON,
            "seq_head_on": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_HEAD_ON,
            "include_bg": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.INCLUDE_BG,
            "cls_ins_weight": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.INS_LOSS_WEIGHT,
            "cls_pair_weight": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.PAIR_LOSS_WEIGHT,
            "cls_seq_weight": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_LOSS_WEIGHT,
            "cls_seq_aux_weight": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_AUX_LOSS_WEIGHT,
            "seq_batch_size": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_BATCH_SIZE,
            "seq_length_range": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_LENGTH_RANGE,
            "seq_dim": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_DIM,
            "num_heads": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.NUM_HEADS,
            "num_enc_layers": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.NUM_ENC_LAYERS,
            "use_cls_cnt": cfg.MODEL.QDTRACK.ROI_CLS_HEAD.USE_CLS_CNT,
        }

    def inference(self, proposals, cls_features):
        num_inst_per_image = [len(p) for p in proposals]
        cls_features = cls_features.split(num_inst_per_image, dim=0)

        ret_proposals = []
        for proposals_per_image, cls_features_per_image in zip(
            proposals, cls_features
        ):
            proposals_per_image.cls_feats = cls_features_per_image

            ret_proposals.append(proposals_per_image)

        return ret_proposals

    def losses(self, embeds, instances):
        num_roi = len(embeds)

        gt_classes = torch.cat([ins.gt_classes for ins in instances])
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]

        if self.include_bg:
            valid_inds = nonzero_tuple(gt_classes >= 0)[0]
            ins_embeds = embeds[valid_inds]
            ins_gt_classes = gt_classes[valid_inds]
        else:
            ins_embeds = embeds[fg_inds]
            ins_gt_classes = gt_classes[fg_inds]

        seq_embeds = embeds[fg_inds]
        seq_gt_classes = gt_classes[fg_inds]

        loss_cls = {}
        if self.ins_head_on:
            loss_cls_ins = self.loss_instance(ins_embeds, ins_gt_classes) / max(num_roi, 1)
            loss_cls["loss_cls_ins"] = loss_cls_ins * self.cls_ins_weight
        if self.seq_head_on:
            loss_cls_seq = self.loss_tracklet(seq_embeds, seq_gt_classes)
            loss_cls.update(loss_cls_seq)
        return loss_cls

    @autocast(enabled=False)
    def loss_instance(self, embeds, gt_classes):
        pred_logits = self.cls_ins_head(embeds.float())
        if len(embeds) == 0:
            return pred_logits.sum() * 0.0

        return cross_entropy(pred_logits, gt_classes, reduction="sum")

    @autocast(enabled=False)
    def loss_tracklet(self, embeds, gt_classes):
        embeds = embeds.float()
        N, C = embeds.shape
        if N == 0:
            # When there is no instance in a given batch.
            _dummy = embeds.new_zeros(1, 1, embeds.shape[-1]) + embeds.sum()
            seq_pred_logits, ins_pred_logits = self.cls_seq_head(_dummy)

            loss = {"loss_cls_seq": seq_pred_logits.sum() * 0.0}
            if self.cls_seq_aux_on:
                loss["loss_cls_seq_aux"] = ins_pred_logits.sum() * 0.0
            return loss

        if self.use_cls_cnt:
            # TODO the line below would be very important.
            sample_prob = 1 / ((self.cls_cnt)[gt_classes] ** 0.5)
        else:
            sample_prob = torch.ones((len(gt_classes),), dtype=torch.float, device=embeds.device)

        # Add buffers to make the chunk be the size of total_sample_size.
        sample_idx = torch.multinomial(sample_prob, self.sample_size, replacement=True)
        sample_gt_classes = gt_classes[sample_idx]
        sample_embeds = embeds[sample_idx]

        origin_idx = sample_idx.new_zeros(self.seq_batch_size, self.seq_length_range[1]) - 1
        origin_idx[self.insert_idx[0], self.insert_idx[1]] = sample_idx

        gt_classes = sample_gt_classes.new_zeros(self.seq_batch_size, self.seq_length_range[1]) - 1
        gt_classes[self.insert_idx[0], self.insert_idx[1]] = sample_gt_classes

        input_embeds = sample_embeds.new_zeros(self.seq_batch_size, self.seq_length_range[1], C)
        input_embeds[self.insert_idx[0], self.insert_idx[1]] = sample_embeds

        mask = (gt_classes == -1)

        # Assign gt distribution by the proportion of gt classes
        _gt_classes = gt_classes[:, None, :].repeat(1, self.num_classes, 1)
        arange_classes = torch.arange(self.num_classes, device=embeds.device)[None, :, None]
        gt_classes_cnt = (_gt_classes == arange_classes).sum(dim=2).float()
        gt_distribution = gt_classes_cnt / (~mask).sum(dim=1, keepdims=True)

        # forward into the sequence head.
        seq_pred_logits, ins_pred_logits = self.cls_seq_head(input_embeds, mask=mask)

        # Cross-entropy
        loss_cls_seq = -F.log_softmax(seq_pred_logits, 1) * gt_distribution
        loss_cls_seq = loss_cls_seq.sum() / len(input_embeds)

        losses = {"loss_cls_seq": loss_cls_seq * self.cls_seq_weight}

        if self.cls_seq_aux_on:
            # Auxiliary Loss
            origin_idx = (
                origin_idx[:, :, None] == torch.arange(N, device=origin_idx.device)[None, None, :]
            )
            origin_cnt = origin_idx.sum(dim=(0,1))
            element_weight = (origin_idx / (origin_cnt[None, None, :] + 1e-6)).sum(dim=2)

            loss_cls_seq_aux = F.cross_entropy(
                ins_pred_logits.flatten(0,1), gt_classes.flatten(), reduction='none', ignore_index=-1)
            loss_cls_seq_aux = (loss_cls_seq_aux * element_weight.flatten()).sum() / N

            losses.update({"loss_cls_seq_aux": loss_cls_seq_aux * self.cls_seq_aux_weight})

        return losses

    @autocast(enabled=False)
    def loss_pair(self, embeds, instances):
        embeds = embeds.float()
        if len(embeds) == 0:
            return {"loss_cls_pair": self.cls_ins_head(embeds).sum() * 0.0}

        num_instances = [len(x1)+len(x2) for x1, x2 in zip(instances[::2], instances[1::2])]
        gt_ids = [torch.cat((x1.gt_ids, x2.gt_ids)) for x1, x2 in zip(instances[::2], instances[1::2])]

        pred_logits = self.cls_ins_head(embeds)
        pred_logits_split = torch.split(pred_logits.detach(), num_instances)

        centroid_logits = []
        for _ids, _pred_logits in zip(gt_ids, pred_logits_split):
            unique_id_match = torch.unique(_ids)[:, None] == _ids[None]
            _centroid_logits = (
                (unique_id_match.float() @ _pred_logits) / unique_id_match.sum(dim=1, keepdims=True)
            )

            # IDs should be contiguously mapped.
            # e.g., _ids = [10, 11, 12, 15]
            # Shape of _centroid_dists would be (4, K), and indexing by _ids is invalid.
            # Thus map [10, 11, 12, 15] to [0, 1, 2, 3] by the below line.
            _ids_contiguous = unique_id_match.T.nonzero()[:,1]

            centroid_logits.append(_centroid_logits[_ids_contiguous])
        centroid_logits = torch.cat(centroid_logits)

        loss_pair = F.kl_div(
            F.log_softmax(pred_logits, dim=1), F.softmax(centroid_logits, dim=1),
            reduction="batchmean"
        )
        return {"loss_cls_pair": loss_pair * self.cls_pair_weight}


def build_cls_head(cfg):
    """
    Build a cls head defined by `cfg.MODEL.QDTRACK.ROI_CLS_HEAD.NAME`.
    """
    name = cfg.MODEL.QDTRACK.ROI_CLS_HEAD.NAME
    return ROI_CLS_HEAD_REGISTRY.get(name)(cfg)
