# -*- coding: utf-8 -*-
import os

from detectron2.data.datasets.lvis_v0_5_categories import LVIS_CATEGORIES as LVIS_V0_5_CATEGORIES

from .lvis import register_lvis_instances, get_lvis_instances_meta
from .tao import register_tao_instances
from .tao_categories import TAO_CATEGORIES

# ==== Predefined splits for TAO ===========
_PREDEFINED_SPLITS_TAO = {
    "tao_train"         : ("tao/frames/", "tao/annotations/train_ours.json",         TAO_CATEGORIES),
    "tao_val"           : ("tao/frames/", "tao/annotations/validation_ours.json",    TAO_CATEGORIES),
    "tao_test"          : ("tao/frames/", "tao/annotations/test_482_ours.json",          TAO_CATEGORIES),
    "tao_train_full"    : ("tao/frames/", "tao/annotations/train.json",         None),
    "tao_val_full"      : ("tao/frames/", "tao/annotations/validation.json",    None),
    "tao_test_full"     : ("tao/frames/", "tao/annotations/test.json",          None),
}


def register_all_tao(root):
    for key, (image_root, json_file, class_list) in _PREDEFINED_SPLITS_TAO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_tao_instances(
            key,
            get_lvis_instances_meta(key, class_list),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            class_list,
        )


# ==== Predefined splits for LVIS ===========
_PREDEFINED_SPLITS_LVIS = {
    "lvis_tao_merge_coco_train" : ("coco/", "lvis/lvis_v0.5_coco2017_train.json",   TAO_CATEGORIES),
    "lvis_tao_train"            : ("coco/", "lvis/lvis_v0.5_train.json",            TAO_CATEGORIES),
    "lvis_tao_val"              : ("coco/", "lvis/lvis_v0.5_val.json",              TAO_CATEGORIES),
    "lvis_tao_test"             : ("coco/", "lvis/lvis_v0.5_image_info_test.json",  TAO_CATEGORIES),
}


def register_all_lvis(root):
    for key, (image_root, json_file, class_list) in _PREDEFINED_SPLITS_LVIS.items():
        register_lvis_instances(
            key,
            get_lvis_instances_meta(key, class_list),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            class_list,
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_tao(_root)
    register_all_lvis(_root)
