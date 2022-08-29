# -*- coding: utf-8 -*-
from pickle import FALSE
from detectron2.config import CfgNode as CN


def add_track_config(cfg):
    """
    Add config for QDT.
    """
    cfg.MODEL.QDTRACK = CN()
    cfg.MODEL.QDTRACK.TRACK_ON = True
    cfg.MODEL.QDTRACK.FREEZE_DETECTOR = False
    cfg.MODEL.QDTRACK.CLS_FINETUNE = False
    cfg.MODEL.QDTRACK.K_VALUES = (2, 3.5, 3.5)
    cfg.MODEL.QDTRACK.MATCH_SCORE_THR = 0.5

    # Track Head
    cfg.MODEL.QDTRACK.ROI_TRACK_HEAD = CN()
    cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.NAME = "QDTrackHead"
    cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.IOU_THRESHOLDS = [0.3, 0.7]
    cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.IOU_LABELS = [0, -1, 1]

    cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.POSITIVE_FRACTION = 0.5
    cfg.MODEL.QDTRACK.ROI_TRACK_HEAD.NEG_POS_RATIO = 3.0

    cfg.MODEL.QDTRACK.ROI_TRACK_LOSS = CN()
    cfg.MODEL.QDTRACK.ROI_TRACK_LOSS.NAME = "MultiPosCrossEntropy"
    cfg.MODEL.QDTRACK.ROI_TRACK_LOSS.WEIGHT = 0.25

    cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS = CN()
    cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.NAME = "L2Loss"
    cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.WEIGHT = 1.0
    cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.POS_MARGIN = 0.0
    cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.NEG_MARGIN = 0.1
    cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.HARD_MINING = True
    cfg.MODEL.QDTRACK.ROI_TRACK_AUX_LOSS.NEG_POS_RATIO = 3.0

    # Embed Head
    cfg.MODEL.QDTRACK.ROI_EMBED_HEAD = CN()
    cfg.MODEL.QDTRACK.ROI_EMBED_HEAD.NAME = "QDTrackEmbedHead"
    cfg.MODEL.QDTRACK.ROI_EMBED_HEAD.NUM_FC = 1
    cfg.MODEL.QDTRACK.ROI_EMBED_HEAD.FC_DIM = 1024
    cfg.MODEL.QDTRACK.ROI_EMBED_HEAD.NUM_CONV = 4
    cfg.MODEL.QDTRACK.ROI_EMBED_HEAD.CONV_DIM = 256
    cfg.MODEL.QDTRACK.ROI_EMBED_HEAD.NORM = "GN"
    cfg.MODEL.QDTRACK.ROI_EMBED_HEAD.OUTPUT_DIM = 256

    # Class Head
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD = CN()
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.NAME = "ClsHead"
    # Class Head - INS
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.INS_HEAD_ON = True
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.INCLUDE_BG = False
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.INS_LOSS_WEIGHT = 0.5
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.PAIR_LOSS_WEIGHT = 0.1
    # Class Head - SEQ
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_HEAD_ON = True
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_LOSS_WEIGHT = 0.05
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_AUX_LOSS_WEIGHT = 0.02
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_BATCH_SIZE = 256
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_LENGTH_RANGE = (16, 32)
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.SEQ_DIM = 512
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.NUM_HEADS = 8
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.NUM_ENC_LAYERS = 3
    cfg.MODEL.QDTRACK.ROI_CLS_HEAD.USE_CLS_CNT = True

    # Data Configurations
    cfg.INPUT.AUGMENTATIONS = []
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 1
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False

    # Visualization Configurations
    cfg.TEST.VISUALIZE = False
    cfg.TEST.VIS_OUTDIR = "visualized"
    cfg.TEST.VIS_THRES = 0.3

    cfg.DATASETS.DATASET_RATIO = (1.0,)