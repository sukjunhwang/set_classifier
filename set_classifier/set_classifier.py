import numpy as np
from typing import Counter, Dict, List, Optional, Tuple
import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.layers import nonzero_tuple

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from .models import TaoTracker

__all__ = ["QDTrack"]


@META_ARCH_REGISTRY.register()
class QDTrack(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        freeze_detector: bool = False,
        cls_finetune: bool = False,
        track_on: bool = False,
        is_tao: bool = False,
        test_topk_per_image: int = 300,
        score_thresh_test: float = 0.05,
        k_values: tuple = (2, 3.5, 3.5),
        match_score_thr: float = 0.5,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.k_values = k_values

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

        self.tracker = TaoTracker(
            match_score_thr=match_score_thr,
        )
        self.track_on = track_on
        self.is_tao = is_tao
        self.test_topk_per_image = test_topk_per_image
        self.score_thresh_test = score_thresh_test

        if freeze_detector:
            for name, p in self.named_parameters():
                if "track" not in name:
                    p.requires_grad_(False)
        if cls_finetune:
            for name, p in self.named_parameters():
                if not ("cls_head" in name or "cls_predictor" in name):
                    p.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "freeze_detector": cfg.MODEL.QDTRACK.FREEZE_DETECTOR,
            "cls_finetune": cfg.MODEL.QDTRACK.CLS_FINETUNE,
            "track_on": cfg.MODEL.QDTRACK.TRACK_ON,
            "is_tao": cfg.DATASETS.TEST[0].startswith("tao"),
            "test_topk_per_image" : cfg.TEST.DETECTIONS_PER_IMAGE,
            "score_thresh_test": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "k_values": cfg.MODEL.QDTRACK.K_VALUES,
            "match_score_thr": cfg.MODEL.QDTRACK.MATCH_SCORE_THR,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            if self.track_on and self.is_tao:
                return self.inference_track(batched_inputs)
            else:
                return self.inference_det(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = []
            for video_inputs in batched_inputs:
                for frame_instances in video_inputs["instances"]:
                    gt_instances.append(frame_instances.to(self.device))
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference_det(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, _ = self.proposal_generator(images, features, None)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        results, _ = self.roi_heads(images, features, proposals, None)

        return self.detection_postprocess(results, batched_inputs, images.image_sizes)

    def inference_track(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        assert len(batched_inputs) == 1
        self.tracker.reset()

        images = self.preprocess_image(batched_inputs)
        num_frames = len(images.tensor)
        for frame_idx in range(num_frames):
            frame = ImageList(images.tensor[[frame_idx]], [images.image_sizes[frame_idx]])
            features = self.backbone(frame.tensor)

            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(frame, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(frame, features, proposals, None)

            _detection_results = self.detection_postprocess(results, batched_inputs, frame.image_sizes)
            _detection_results = _detection_results[0]["instances"]

            self.tracker.match(
                bboxes=_detection_results.pred_boxes,
                labels=_detection_results.pred_classes,
                scores=_detection_results.scores,
                cls_feats=_detection_results.cls_feats,
                track_ins_feats=_detection_results.track_ins_feats,
                frame_id=frame_idx,
            )

        return self.tracking_postprocess(
            self.tracker.tracklets, self.roi_heads.cls_predictor.cls_seq_head
        )

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(self.normalizer(frame.to(self.device)))
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def detection_postprocess(self, instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        NOTE it outputs List[Instances].
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def tracking_postprocess(self, tracklets, clip_cls_predictor):
        M = self.roi_heads.cls_predictor.seq_length_range[1]
        C_C = list(tracklets.items())[0][1]["cls_feats"][0].shape[-1]
        max_len = max([len(t["scores"]) for _, t in tracklets.items()] + [M])

        mask = torch.ones((len(tracklets), max_len), dtype=torch.bool, device=self.device)
        cls_feats = torch.zeros((len(tracklets), max_len, C_C), dtype=torch.float, device=self.device)

        tracklet_scores = []
        tracklet_lengths = []
        for t_i, (id, tracklet) in enumerate(tracklets.items()):
            assert id != -1, "ID == -1 appeared. Not expected."
            L = len(tracklet["scores"])
            tracklet_scores.append(sum(tracklet["scores"]) / L)

            mult = max(1, M // L)
            mask[t_i, :L*mult] = False
            cls_feats[t_i, :L*mult] = torch.cat(tracklet['cls_feats'] * mult)
            tracklet_lengths.append(L)
        tracklet_lengths = torch.tensor(tracklet_lengths, device=self.device)

        clip_cls_logits = clip_cls_predictor(cls_feats, mask=mask)[0]
        clip_cls_scores = F.softmax(clip_cls_logits, dim=1)

        len_scores = tracklet_lengths / max_len

        k1, k2, k3 = self.k_values
        k_all = sum([k1, k2, k3])

        out_tracklets = []
        for i, (_, tracklet) in enumerate(tracklets.items()):
            valid_idx = nonzero_tuple(clip_cls_scores[i] > 0.001)[0].cpu().tolist()
            cls_scores = ((
                (clip_cls_scores[i] ** k1) * (tracklet_scores[i] ** k2) * (len_scores[i] ** k3)
            ) ** (1/k_all)).cpu().tolist()
            for v_i in valid_idx:
                out_tracklet = {}
                out_tracklet["label"] = v_i
                out_tracklet["score"] = cls_scores[v_i]
                out_tracklet["bboxes"] = tracklet["bboxes"]
                out_tracklet["frame_idxs"] = tracklet["frame_ids"]
                out_tracklets.append(out_tracklet)

        out_tracklets = sorted(out_tracklets, key=lambda x: x["score"], reverse=True)
        out_tracklets = out_tracklets[:300]

        return out_tracklets
