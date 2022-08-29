from random import randint
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.utils.registry import Registry

from .embed_head import build_embed_head
from .track_loss import build_track_loss
from .transformer import SequencePredictor
from .misc import MLP

__all__ = ["QDTrackHead", "build_track_head", "ROI_TRACK_HEAD_REGISTRY"]

ROI_TRACK_HEAD_REGISTRY = Registry("ROI_TRACK_HEAD")
ROI_TRACK_HEAD_REGISTRY.__doc__ = """
Registry for track heads, which predicts instance representation vectors given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def cal_similarity(key_embeds,
                   ref_embeds,
                   method='dot_product',
                   temperature=-1):
    assert method in ['dot_product', 'cosine']

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
        return torch.mm(key_embeds, ref_embeds.t())
    elif method == 'dot_product':
        if temperature > 0:
            dists = cal_similarity(key_embeds, ref_embeds, method='cosine')
            dists /= temperature
            return dists
        else:
            return torch.mm(key_embeds, ref_embeds.t())


def track_head_inference(instances, track_ins_features):
    num_insances = [len(p) for p in instances]
    track_ins_features = torch.split(track_ins_features, num_insances)

    for track_ins_features_per_image, instances_per_image in zip(
        track_ins_features, instances
    ):
        instances_per_image.track_ins_feats = track_ins_features_per_image


@ROI_TRACK_HEAD_REGISTRY.register()
class QDTrackHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, sampling_frame_num, track_embed_head,
        loss_track, loss_track_aux,
    ):
        super().__init__()
        self.sampling_frame_num = sampling_frame_num
        self.track_embed_head = track_embed_head
        channel_size = self.track_embed_head._output_size
        self.track_out_layer = MLP(channel_size, channel_size, channel_size, 1)

        self.loss_track = loss_track
        self.loss_track_aux = loss_track_aux

    @classmethod
    def from_config(cls, cfg, input_shape):
        track_embed_head = cls._init_embed_head(cfg, input_shape)

        loss_track_name = cfg.MODEL.QDTRACK.ROI_TRACK_LOSS.NAME
        loss_track = build_track_loss(cfg, loss_track_name)

        loss_track_aux_name = cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.NAME
        loss_track_aux = build_track_loss(cfg, loss_track_aux_name)

        return {
            "sampling_frame_num": cfg.INPUT.SAMPLING_FRAME_NUM,
            "track_embed_head": track_embed_head,
            "loss_track": loss_track,
            "loss_track_aux": loss_track_aux,
        }

    @classmethod
    def _init_embed_head(cls, cfg, input_shape):
        if not cfg.MODEL.QDTRACK.TRACK_ON:
            return {"track_head": None}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        return build_embed_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

    def forward(self, pos_features, pos_instances, neg_features=None, neg_instances=None):
        pos_embeds = F.relu(self.track_embed_head(pos_features))
        pos_track_embeds = self.track_out_layer(pos_embeds)

        if neg_features is not None:
            neg_embeds = F.relu(self.track_embed_head(neg_features))
            neg_track_embeds = self.track_out_layer(neg_embeds)

        if self.training:
            losses = {}
            losses.update(
                self.losses_track(
                    pos_track_embeds, pos_instances, neg_track_embeds, neg_instances
                )
            )
            return losses
        else:
            track_head_inference(pos_instances, pos_track_embeds)
            return pos_instances

    def forward_seq_test(self, pos_embeds, mask):
        _, seq_pred = self.track_seq_head(pos_embeds, mask=mask)
        seq_pred = self.ins_pred_layer(seq_pred)
        seq_pred = torch.bmm(seq_pred, seq_pred.permute(0, 2, 1))

        valid = ~mask
        valid_sequence = valid[:, None] & valid[..., None]
        valid_len = valid.sum(dim=1)

        seq_pred = seq_pred.sigmoid()
        pred_scores = (seq_pred * valid_sequence).sum(dim=2) / (valid_len[:, None] + 1e-6)
        pred_scores = pred_scores.sum(dim=1) / (valid_len + 1e-6)

        return pred_scores

    @autocast(enabled=False)
    def losses_track(self, pos_embeds, pos_instances, neg_embeds, neg_instances):
        pos_embeds = pos_embeds.float()
        neg_embeds = neg_embeds.float()

        pos_num_instances = [len(x) for x in pos_instances]
        neg_num_instances = [len(x) for x in neg_instances]

        pos_ids = [x.gt_ids for x in pos_instances]
        neg_ids = [x.gt_ids for x in neg_instances]

        key_ids = pos_ids
        _ref_ids = [torch.cat((p, n)) for p, n in zip(pos_ids, neg_ids)]
        ref_ids = []
        for i in range(0, len(_ref_ids), 2):
            ref_ids.append(_ref_ids[i+1])
            ref_ids.append(_ref_ids[i])

        targets, weights = self.get_sim_targets(key_ids, ref_ids)

        pos_embeds = torch.split(pos_embeds, pos_num_instances)
        neg_embeds = torch.split(neg_embeds, neg_num_instances)

        # Assuming only pairs of frames are taken into the batch
        key_embeds = pos_embeds
        _ref_embeds = [torch.cat((p, n)) for p, n in zip(pos_embeds, neg_embeds)]
        ref_embeds = []
        for i in range(0, len(_ref_embeds), 2):
            ref_embeds.append(_ref_embeds[i+1])
            ref_embeds.append(_ref_embeds[i])

        dists, cos_dists = self.get_sim_distances(key_embeds, ref_embeds)

        return self.get_sim_loss(dists, cos_dists, targets, weights)

    def get_sim_targets(self, key_ids, ref_ids):
        targets = [(k[:,None] == r[None]).float() for k, r in zip(key_ids, ref_ids)]
        weights = [(t.sum(dim=1) > 0.0).float() for t in targets]

        return targets, weights

    def get_sim_distances(self, key_embeds, ref_embeds):
        dists, cos_dists = [], []
        for _key_embeds, _ref_embeds in zip(key_embeds, ref_embeds):
            # Dot product similarity
            # NOTE check if softmax_temp is necessary
            dist = cal_similarity(
                _key_embeds, _ref_embeds, method='dot_product')
            dists.append(dist)

            # Cosine similarity
            cos_dist = cal_similarity(
                _key_embeds, _ref_embeds, method='cosine')
            cos_dists.append(cos_dist)

        return dists, cos_dists

    def get_sim_loss(self, dists, cos_dists, targets, weights):
        losses = dict()

        loss_track = 0.
        loss_track_aux = 0.
        for _dists, _cos_dists, _targets, _weights in zip(
                dists, cos_dists, targets, weights):
            loss_track += self.loss_track(
                _dists, _targets, avg_factor=_weights.sum())
            loss_track_aux += self.loss_track_aux(_cos_dists, _targets)
        losses['loss_track'] = loss_track / max(1, len(dists))

        if self.loss_track_aux is not None:
            losses['loss_track_aux'] = loss_track_aux / max(1, len(dists))

        return losses


def build_track_head(cfg, input_shape):
    """
    Build a track head defined by `cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.NAME`.
    """
    name = cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.NAME
    return ROI_TRACK_HEAD_REGISTRY.get(name)(cfg, input_shape)
