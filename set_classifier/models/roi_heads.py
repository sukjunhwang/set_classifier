import copy
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, ROIHeads, StandardROIHeads, select_foreground_proposals

from .cls_head import build_cls_head
from .track_head import build_track_head
from .sampling import subsample_labels_for_track
from .fast_rcnn import FastRCNNOutputLayersSeq

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class QDTrackROIHeads(StandardROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        freeze_detector: bool = False,
        track_head: Optional[nn.Module] = None,
        track_proposal_matcher: Optional[object] = None,
        track_batch_size_per_image: Optional[int] = 256,
        track_positive_fraction: Optional[float] = 0.5,
        track_neg_pos_ratio: Optional[float] = 3.0,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            mask_in_features=mask_in_features,
            mask_pooler=mask_pooler,
            mask_head=mask_head,
            train_on_pred_boxes=train_on_pred_boxes,
            **kwargs,
        )

        self.freeze_detector = freeze_detector
        self.track_on = track_head is not None
        if self.track_on:
            self.track_head = track_head
            self.track_proposal_matcher = track_proposal_matcher
            self.track_batch_size_per_image = track_batch_size_per_image
            self.track_positive_fraction = track_positive_fraction
            self.track_neg_pos_ratio = track_neg_pos_ratio

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["freeze_detector"] = cfg.MODEL.QDTRACK.FREEZE_DETECTOR

        if cfg.MODEL.QDTRACK.TRACK_ON:
            ret.update(cls._init_track_head(cfg, input_shape))
            ret["track_batch_size_per_image"] = cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.BATCH_SIZE_PER_IMAGE
            ret["track_neg_pos_ratio"] = cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.NEG_POS_RATIO
            ret["track_positive_fraction"] = cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.POSITIVE_FRACTION
            ret["track_proposal_matcher"] = Matcher(
                cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.IOU_THRESHOLDS,
                cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.IOU_LABELS,
                allow_low_quality_matches=False,
            )
        return ret

    @classmethod
    def _init_track_head(cls, cfg, input_shape):
        if not cfg.MODEL.QDTRACK.TRACK_ON:
            return {"track_head": None}

        track_head = build_track_head(cfg, input_shape)
        return {"track_head": track_head}

    @torch.no_grad()
    def label_and_sample_proposals_for_track(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        sampled_pos_proposals = []
        sampled_neg_proposals = []

        num_pos_samples = []
        num_neg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.track_proposal_matcher(match_quality_matrix)

            has_gt = len(targets_per_image) > 0
            gt_ids = targets_per_image.gt_ids[matched_idxs] if has_gt else (torch.zeros_like(matched_idxs) - 1)
            gt_classes = targets_per_image.gt_classes[matched_idxs] if has_gt else (torch.zeros_like(matched_idxs) - 1)

            sampled_pos_idxs, sampled_neg_idxs = subsample_labels_for_track(
                gt_ids, matched_labels, self.track_batch_size_per_image, self.track_positive_fraction, self.track_neg_pos_ratio
            )

            gt_pos_ids, gt_neg_ids = gt_ids[sampled_pos_idxs], gt_ids[sampled_neg_idxs]
            gt_classes = gt_classes[sampled_pos_idxs]

            # Set target attributes of the sampled proposals:
            pos_proposals_per_image = proposals_per_image[sampled_pos_idxs]
            pos_proposals_per_image.gt_ids = gt_pos_ids
            pos_proposals_per_image.gt_classes = gt_classes

            neg_proposals_per_image = proposals_per_image[sampled_neg_idxs]
            neg_proposals_per_image.gt_ids = torch.zeros_like(gt_neg_ids) - 1 # Assign -1 as gt_id for all negative samples

            num_pos_samples.append(sampled_pos_idxs.numel())
            num_neg_samples.append(sampled_neg_idxs.numel())
            sampled_pos_proposals.append(pos_proposals_per_image)
            sampled_neg_proposals.append(neg_proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("track_head/num_pos_samples", np.mean(num_pos_samples))
        storage.put_scalar("track_head/num_neg_samples", np.mean(num_neg_samples))

        return sampled_pos_proposals, sampled_neg_proposals

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            box_proposals = self.label_and_sample_proposals(copy.deepcopy(proposals), targets)
            if self.track_on:
                track_proposals = self.label_and_sample_proposals_for_track(
                    copy.deepcopy(proposals), targets
                )
        del targets

        if self.training:
            losses = {}
            if not self.freeze_detector:
                losses.update(self._forward_box(features, box_proposals))
                losses.update(self._forward_mask(features, box_proposals))
            if self.track_on:
                losses.update(self._forward_track(features, *track_proposals))
            return box_proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_track(features, instances)
        return instances

    def _forward_track(self, features, pos_instances, neg_instances=None):
        if not self.track_on:
            return {} if self.training else pos_instances

        features = [features[f] for f in self.box_in_features]
        pos_boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in pos_instances]
        pos_features = self.box_pooler(features, pos_boxes)
        if neg_instances is not None:
            neg_boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in neg_instances]
            neg_features = self.box_pooler(features, neg_boxes)
        else:
            neg_features = None

        return self.track_head(pos_features, pos_instances, neg_features, neg_instances)


@ROI_HEADS_REGISTRY.register()
class QDTrackROIHeadsSeq(QDTrackROIHeads):
    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        cls_head: Optional[nn.Module] = None,
        cls_predictor: Optional[nn.Module] = None,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        freeze_detector: bool = False,
        track_head: Optional[nn.Module] = None,
        track_proposal_matcher: Optional[object] = None,
        track_batch_size_per_image: Optional[int] = 256,
        track_positive_fraction: Optional[float] = 0.5,
        track_neg_pos_ratio: Optional[float] = 3.0,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
        """
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_head,
            box_predictor=box_predictor,
            mask_in_features=mask_in_features,
            mask_pooler=mask_pooler,
            mask_head=mask_head,
            freeze_detector=freeze_detector,
            track_head=track_head,
            track_proposal_matcher=track_proposal_matcher,
            track_batch_size_per_image=track_batch_size_per_image,
            track_positive_fraction=track_positive_fraction,
            track_neg_pos_ratio=track_neg_pos_ratio,
            **kwargs,
        )
        self.cls_head = cls_head
        self.cls_predictor = cls_predictor

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update(cls._init_cls_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_cls_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        cls_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        cls_predictor = build_cls_head(cfg)

        return {"cls_head": cls_head, "cls_predictor": cls_predictor}

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret["box_predictor"]

        ret["box_predictor"] = FastRCNNOutputLayersSeq(cfg, ret["box_head"].output_shape)
        return ret

    def _forward_box(self, features, box_proposals):
        features = [features[f] for f in self.box_in_features]
        _box_features = self.box_pooler(features, [x.proposal_boxes for x in box_proposals])
        box_features = self.box_head(_box_features)
        cls_features = self.cls_head(_box_features)

        box_predictions = self.box_predictor(box_features)
        del box_features, _box_features

        if self.training:
            losses = {}
            losses.update(self.cls_predictor.losses(cls_features, box_proposals))
            losses.update(self.box_predictor.losses(box_predictions, box_proposals))
            return losses
        else:
            pred_instances = self.cls_predictor.inference(box_proposals, cls_features)
            pred_instances, _ = self.box_predictor.inference(box_predictions, pred_instances)
            return pred_instances

    def _forward_track(self, features, pos_instances, neg_instances=None):
        if not self.track_on:
            return {} if self.training else pos_instances

        features = [features[f] for f in self.box_in_features]
        pos_boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in pos_instances]
        pos_features = self.box_pooler(features, pos_boxes)
        if neg_instances is not None:
            neg_boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in neg_instances]
            neg_features = self.box_pooler(features, neg_boxes)
        else:
            neg_boxes, neg_features = None, None

        if self.training:
            losses = self.track_head(pos_features, pos_instances, neg_features, neg_instances)
            if self.cls_predictor.ins_head_on and self.cls_predictor.cls_pair_weight > 0.0:
                losses.update(
                    self.cls_predictor.loss_pair(self.cls_head(pos_features), pos_instances)
                )
            return losses
        else:
            return self.track_head(pos_features, pos_instances)


@ROI_HEADS_REGISTRY.register()
class QDTrackROIHeadsSeqClsFT(QDTrackROIHeadsSeq):
    def _forward_box(self, features, box_proposals):
        features = [features[f] for f in self.box_in_features]
        _box_features = self.box_pooler(features, [x.proposal_boxes for x in box_proposals])
        cls_features = self.cls_head(_box_features)

        if self.training:
            del _box_features

            losses = {}
            losses.update(self.cls_predictor.losses(cls_features, box_proposals))
            return losses
        else:
            _box_features = self.box_head(_box_features)
            box_predictions = self.box_predictor(_box_features)
            del _box_features

            cls_logits = self.cls_predictor.cls_ins_head(cls_features)
            pred_instances = self.cls_predictor.inference(box_proposals, cls_logits, cls_features)
            pred_instances, _ = self.box_predictor.inference(box_predictions, pred_instances)
            return pred_instances

    def _forward_track(self, features, pos_instances, neg_instances=None):
        if not (self.track_on and (self.cls_predictor.ins_head_on and self.cls_predictor.cls_pair_weight > 0.0)):
            return {} if self.training else pos_instances

        features = [features[f] for f in self.box_in_features]
        pos_boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in pos_instances]
        pos_features = self.box_pooler(features, pos_boxes)

        if self.training:
            return self.cls_predictor.loss_pair(self.cls_head(pos_features), pos_instances)
        else:
            return self.track_head(pos_features, pos_instances)
