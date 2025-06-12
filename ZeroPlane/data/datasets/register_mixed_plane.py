# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os
import os.path as osp

from os.path import join as pjoin

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg
from detectron2.utils.file_io import PathManager

import random
import cv2
from matplotlib import pyplot as plt
from detectron2.utils.visualizer import Visualizer
import numpy as np
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

PLANE_ID_COLORS = {}
for i in range(len(COCO_CATEGORIES)):
    PLANE_ID_COLORS[i] = COCO_CATEGORIES[i]["color"]

eval_indoor = True
TRAIN_NUM = 560022

MAX_NUM_PLANES = 20

if eval_indoor:
    VAL_NUM = 654

else:
    VAL_NUM = 356


def get_metadata(num=167771):
    meta = {}

    stuff_classes = [i for i in range(num)]
    stuff_colors = [PLANE_ID_COLORS[i] for i in range(num)]  # coco visualization colors

    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    return meta


def load_single_mixed_plane_json(json_file):
    """
    Args:
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]
        npz_file = ann["npz_file_name"]
        segments_info = ann["segments_info"]
        ret.append(
            {
                "npz_file_name": npz_file,
                "image_id": image_id,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No *.npz files found in {os.path.split(npz_file)[0]}!"
    assert PathManager.isfile(ret[0]["npz_file_name"]), ret[0]["npz_file_name"]

    return ret


def register_single_mixed_plane_annos_seg(name,  metadata, plane_seg_json):

    plane_seg_name = "single_" + name

    DatasetCatalog.register(
        plane_seg_name,
        lambda: load_single_mixed_plane_json(plane_seg_json),
    )

    MetadataCatalog.get(plane_seg_name).set(
        stuff_classes=metadata["stuff_classes"],
        stuff_colors=metadata["stuff_colors"],
        json_file = plane_seg_json,
        evaluator_type = "mixed_plane_seg",
        ignore_label = MAX_NUM_PLANES,
        # **metadata,
    )


def register_all_single_mixed_plane_annos_seg(json_root):

    for split in ['train', 'val']:
        name = "mixed_plane_seg" + "_" + split

        if split == 'train':
            plane_seg_json = pjoin(json_root, 'all_mixed_plane_len' + str(TRAIN_NUM) + '_' + split + '.json')

        elif split == 'val':
            if eval_indoor:
                plane_seg_json = pjoin(osp.dirname(json_root), 'nyuv2_plane/nyuv2_plane_len654_test.json')

            else:
                plane_seg_json = pjoin(osp.dirname(json_root), "parallel_domain_plane/parallel_domain_plane_len" + str(VAL_NUM) + "_" + split + ".json")

        register_single_mixed_plane_annos_seg(
            name,
            get_metadata(num=MAX_NUM_PLANES), #! num_queries
            plane_seg_json,
        )

_root = os.path.join(os.getenv("DETECTRON2_DATASETS", "with_origin_img_plane_datasets"), 'mixed_datasets') # 26
register_all_single_mixed_plane_annos_seg(_root)


def plot_samples(dataset_name, n = 1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):

        data = np.load(s["npz_file_name"])
        img = data["image"]
        plane_seg = data["segmentation"]

        s["sem_seg"] = plane_seg

        visualizer = Visualizer(img[:, :, ::-1], metadata = dataset_custom_metadata)
        out = visualizer.draw_dataset_dict(s)

        cv2.imwrite(dataset_name + "_" + s["image_id"] + ".png", out.get_image()[:, :, ::-1])


if __name__ == "__main__":
    plot_samples("single_mixed_plane_seg_train", n = 3)
