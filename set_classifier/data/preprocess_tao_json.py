import os
import json

from tao.toolkit.tao import Tao


def preprocess_tao_json(file_path, out_file_path):
    tao = Tao(file_path)
    json_file = open(file_path, "r")
    out_file = open(out_file_path, "w")

    raw = json.load(json_file)

    out = {}
    out['videos'] = raw['videos'].copy()
    out['annotations'] = raw['annotations'].copy()
    out['tracks'] = raw['tracks'].copy()
    out['info'] = raw['info'].copy()
    out['categories'] = raw['categories'].copy()
    out['licenses'] = raw['licenses'].copy()
    out['images'] = []

    for video in raw['videos']:
        img_infos = tao.vid_img_map[video['id']]
        for img_info in img_infos:
            img_info['neg_category_ids'] = video['neg_category_ids']
            img_info['not_exhaustive_category_ids'] = video['not_exhaustive_category_ids']
            out['images'].append(img_info)

    json.dump(out, out_file)


if __name__ == "__main__":
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    train_path = os.path.join(_root, "tao/annotations/train.json")
    train_out_path = os.path.join(_root, "tao/annotations/train_ours.json")
    val_path = os.path.join(_root, "tao/annotations/validation.json")
    val_out_path = os.path.join(_root, "tao/annotations/validation_ours.json")
    test_path = os.path.join(_root, "tao/annotations/test.json")
    test_out_path = os.path.join(_root, "tao/annotations/test_ours.json")

    preprocess_tao_json(train_path, train_out_path)
    preprocess_tao_json(val_path, val_out_path)
    preprocess_tao_json(test_path, test_out_path)
