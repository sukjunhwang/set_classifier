from collections import defaultdict
from math import exp

import torch
import torch.nn.functional as F

from detectron2.layers import nonzero_tuple
from detectron2.structures import pairwise_iou

from .track_head import cal_similarity


class TaoTracker(object):

    def __init__(self,
                 init_score_thr=0.001,
                 obj_score_thr=0.001,
                 match_score_thr=0.5,
                 memo_frames=10,
                 momentum_embed=0.8,
                 momentum_obj_score=0.5,
                 obj_score_diff_thr=1.0,
                 distractor_nms_thr=0.3,
                 distractor_score_thr=0.5,
                 match_metric='bisoftmax',
                 match_with_cosine=True,):
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr

        self.memo_frames = memo_frames
        self.momentum_embed = momentum_embed
        self.momentum_obj_score = momentum_obj_score
        self.obj_score_diff_thr = obj_score_diff_thr
        self.distractor_nms_thr = distractor_nms_thr
        self.distractor_score_thr = distractor_score_thr
        assert match_metric in ['bisoftmax', 'cosine']
        self.match_metric = match_metric
        self.match_with_cosine = match_with_cosine

        self.reset()

    def reset(self):
        self.num_tracklets = 0
        self.tracklets = dict()
        # for analysis
        self.pred_tracks = defaultdict(lambda: defaultdict(list))
        self.gt_tracks = defaultdict(lambda: defaultdict(list))

    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(
        self, ids, bboxes, labels, scores, cls_feats, track_ins_feats, frame_id
    ):
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, label, score, cls_feat, track_ins_feat in zip(
            ids[tracklet_inds],
            bboxes[tracklet_inds],
            labels[tracklet_inds],
            scores[tracklet_inds],
            cls_feats[tracklet_inds],
            track_ins_feats[tracklet_inds],
        ):
            id = int(id)
            if id in self.tracklets:
                self.tracklets[id]['bboxes'].append(bbox)
                self.tracklets[id]['labels'].append(label)
                self.tracklets[id]['scores'].append(score)
                self.tracklets[id]['cls_feats'].append(cls_feat[None])
                self.tracklets[id]['track_ins_feats'] = (
                    (1 - self.momentum_embed) * self.tracklets[id]['track_ins_feats'] + self.momentum_embed * track_ins_feat
                )
                self.tracklets[id]['frame_ids'].append(frame_id)
            else:
                self.tracklets[id] = dict(
                    bboxes=[bbox],
                    labels=[label],
                    scores=[score],
                    cls_feats=[cls_feat[None]],
                    track_ins_feats=track_ins_feat,
                    frame_ids=[frame_id])

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            if frame_id - v['frame_ids'][-1] >= self.memo_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

    @property
    def memo(self):
        memo_ids = []
        memo_labels = []
        memo_scores = []
        memo_track_ins_feats = []
        for k, v in self.tracklets.items():
            memo_ids.append(k)
            memo_labels.append(v['labels'][-1].view(1, 1))
            memo_scores.append(v['scores'][-1].view(1, 1))
            memo_track_ins_feats.append(v['track_ins_feats'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)

        memo_track_ins_feats = torch.cat(memo_track_ins_feats, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_scores = torch.cat(memo_scores, dim=0).squeeze(1)
        return memo_labels, memo_scores, memo_track_ins_feats, memo_ids.squeeze(0)

    def init_tracklets(self, ids, obj_scores):
        new_objs = (ids == -1) & (obj_scores > self.init_score_thr).cpu()
        num_new_objs = new_objs.sum()
        ids[new_objs] = torch.arange(
            self.num_tracklets,
            self.num_tracklets + num_new_objs,
            dtype=torch.long)
        self.num_tracklets += num_new_objs
        return ids

    def match(self,
              bboxes,
              labels,
              scores,
              cls_feats,
              track_ins_feats,
              frame_id,
              temperature=-1,
              **kwargs):
        # all objects is valid here
        valid_inds = torch.ones((len(bboxes),), dtype=torch.bool, device=bboxes.device)

        # nms
        low_inds = nonzero_tuple(scores < self.distractor_score_thr)[0]
        cat_same = labels[low_inds].view(-1, 1) == labels.view(1, -1)
        ious = pairwise_iou(bboxes[low_inds], bboxes)
        sims = ious * cat_same
        for i, ind in enumerate(low_inds):
            if (sims[i, :ind] > self.distractor_nms_thr).any():
                valid_inds[ind] = False
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        scores = scores[valid_inds]
        cls_feats = cls_feats[valid_inds]
        track_ins_feats = track_ins_feats[valid_inds]

        # match if buffer is not empty
        if len(bboxes) > 0 and not self.empty:
            memo_labels, memo_scores, memo_track_ins_feats, memo_ids = self.memo

            sims = cal_similarity(
                track_ins_feats,
                memo_track_ins_feats,
                method='dot_product',
                temperature=temperature)
            cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
            exps = torch.exp(sims) * cat_same
            d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
            t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
            sim_scores = (d2t_scores + t2d_scores) / 2

            cos_scores = cal_similarity(track_ins_feats, memo_track_ins_feats, method='cosine')
            cos_scores = 0.5 * cos_scores + 0.5
            cos_scores = cos_scores * cat_same
            if self.match_with_cosine:
                sim_scores = (sim_scores + cos_scores) / 2

            obj_score_diffs = torch.abs(scores.view(-1, 1) - memo_scores.view(1, -1))

            num_objs = len(bboxes)
            ids = torch.full((num_objs, ), -1, dtype=torch.long)
            for i in range(num_objs):
                if scores[i] < self.obj_score_thr:
                    continue

                conf, memo_ind = torch.max(sim_scores[i, :], dim=0)
                obj_score_diff = obj_score_diffs[i, memo_ind]
                if (conf > self.match_score_thr) and (obj_score_diff < self.obj_score_diff_thr):
                    ids[i] = memo_ids[memo_ind]
                    sim_scores[:i, memo_ind] = 0
                    sim_scores[i + 1:, memo_ind] = 0

                    scores[i] = self.momentum_obj_score * scores[i] + (1 - self.momentum_obj_score) * memo_scores[memo_ind]
        else:
            ids = torch.full((len(bboxes), ), -1, dtype=torch.long)
        # init tracklets
        ids = self.init_tracklets(ids, scores)
        self.update_memo(
            ids, bboxes, labels, scores, cls_feats, track_ins_feats, frame_id
        )
