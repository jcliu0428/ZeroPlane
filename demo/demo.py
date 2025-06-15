# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
from pathlib import Path

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from detectron2.engine import default_setup

from ZeroPlane import add_ZeroPlane_config
from predictor import VisualizationDemo

from ZeroPlane.utils.disp import visualizationBatch


def get_coordinate_map(K, device, h=192, w=256, oh=480, ow=640):
    # Calculate K_inv * xy1, taking into account that the image has been scaled (oh/ow->h/w),
    # and if there are any other image processing steps, they should be included in the calculation.
    K_inv = np.linalg.inv(np.array(K))

    K = torch.FloatTensor(K).to(device)
    K_inv = torch.FloatTensor(K_inv).to(device)

    x = torch.arange(w, dtype=torch.float32).view(1, w) / w * ow
    y = torch.arange(h, dtype=torch.float32).view(h, 1) / h * oh

    x = x.to(device)
    y = y.to(device)
    xx = x.repeat(h, 1)
    yy = y.repeat(1, w)
    xy1 = torch.stack((xx, yy, torch.ones((h, w), dtype=torch.float32).to(device)))  # (3, h, w)
    xy1 = xy1.view(3, -1)  # (3, h*w)

    k_inv_dot_xy_1 = torch.matmul(K_inv, xy1)  # (3, h*w)

    return k_inv_dot_xy_1


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_ZeroPlane_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # default_setup(cfg, args)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="PlaneRecTR demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/PlaneRecTRScanNetV1/PlaneRecTR_R50_demo.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
        default=None
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--fx",
        type=float,
        help="camera K",
    )

    parser.add_argument(
        "--fy",
        type=float,
        help="camera K",
    )

    parser.add_argument(
        "--ox",
        type=float,
        help="camera K",
    )

    parser.add_argument(
        "--oy",
        type=float,
        help="camera K",
    )

    parser.add_argument(
        "--original-w",
        type=float,
        default=640,
        help="original width corresponding to camera K",
    )

    parser.add_argument(
        "--original-h",
        type=float,
        default=480,
        help="original height corresponding to camera K",
    )

    parser.add_argument(
        "--resize_w",
        type=int,
        default=256,
        help="resize image width before feeding into the model",
    )

    parser.add_argument(
        "--resize_h",
        type=int,
        default=192,
        help="resize image height before feeding into the model",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    print('cfg:', cfg)

    demo = VisualizationDemo(cfg)

    if args.input is None:
        args.input = ['./demo/0_d2_image.png']

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        anchors = {}
        anchors['anchor_normals'] = torch.tensor(np.load('./cluster_anchor/new_mixed_normal_anchors_7.npy')).to(cfg.MODEL.DEVICE)
        anchors['anchor_offsets'] = torch.tensor(np.load('./cluster_anchor/new_mixed_offset_anchors_20.npy')).to(cfg.MODEL.DEVICE)

        # K = np.asarray([
        #     [args.fx, 0, args.ox],
        #     [0, args.fy, args.oy],
        #     [0, 0, 1]]
        # )

        # specify the input intrinsic corresponding to (origin_w, origin_h) image size.
        K = np.asarray([[518.86, 0, 325.58],
                        [0, 519.47, 253.74],
                        [0, 0, 1]])

        for idx, path in tqdm.tqdm(enumerate(args.input), disable=not args.output):
            img_name = Path(path).stem
            img = read_image(path, format="RGB")

            img = cv2.resize(img, (args.resize_w, args.resize_h)) # please resize to a smaller resolution if you encounter memory issues

            k_inv_dot_xy_1 = get_coordinate_map(K, torch.device("cpu"), h=args.resize_h, w=args.resize_w, oh=args.original_h, ow=args.original_w).numpy()
            k_inv_dot_xy_1 = torch.tensor(k_inv_dot_xy_1).to(cfg.MODEL.DEVICE)

            start_time = time.time()
            predictions = demo.run_on_image(img, anchors, k_inv_dot_xy_1)

            sem_seg = predictions["sem_seg"].argmax(dim=0).cpu() # torch.Size([192, 256]) # sem_seg 21, 192, 256
            pred = np.array(sem_seg, dtype=int)  # (192, 256)
            plane_depth = predictions["planes_depth"].cpu().numpy()

            # K = [[args.fx, 0, args.ox],
            # [0, args.fy, args.oy],
            # [0, 0, 1]]

            vis_dicts = {
                        'image': img[:,:,::-1], #->BGR
                        'segmentation': pred,
                        'depth_predplane': plane_depth,
                        # Calculate K_inv * xy1, taking into account that the image has been scaled (oh/ow->h/w),
                        # and if there are any other image processing steps, they should be included in the calculation.
                        'K_inv_dot_xy_1': k_inv_dot_xy_1.cpu().numpy()
            }

            if args.output:
                os.makedirs(args.output, exist_ok=True)

                visualizationBatch(root_path=args.output, idx=str(idx), info=img_name, data_dict=vis_dicts,
                                   num_queries=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                                   save_image=True,
                                   save_segmentation=True,
                                   save_depth=True,
                                   save_ply=True,
                                   save_cloud=False)
