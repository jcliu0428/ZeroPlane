import copy
import numpy as np
import os
import os.path as osp
import cv2
import torch

from detectron2.data import detection_utils as utils
from detectron2.config import configurable

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    PolygonMasks,
    polygons_to_bitmask,
)
import torchvision.transforms as transforms


import logging
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Boxes, Instances

from ...utils.plane_utils import make_pixel_normal_map

__all__ = ['SingleNYUv2PlaneDatasetMapper']


def dataset_precompute_K_inv_dot_xy_1(K_inv, image_h=192, image_w=256):

    x = torch.arange(image_w, dtype=torch.float32).view(1, image_w)
    y = torch.arange(image_h, dtype=torch.float32).view(image_h, 1)

    xx = x.repeat(image_h, 1)
    yy = y.repeat(1, image_w)
    xy1 = torch.stack((xx, yy, torch.ones((image_h, image_w), dtype=torch.float32)))  # (3, image_h, image_w)

    # xy1 = xy1.view(3, -1)  # (3, image_h*image_w)
    xy1 = xy1.numpy()

    K_inv_dot_xy_1 = np.einsum('ij,jkl->ikl', K_inv, xy1) # (3, 3) *(3, image_h, image_w) -> (3, image_h, image_w)

    return K_inv_dot_xy_1, xy1


class SingleNYUv2PlaneDatasetMapper():
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by PlaneRecTR.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        normal_class_num,
        offset_class_num,
        use_coupled_anchor,
        backbone,
        large_resolution_input=False,
        large_resolution_eval=False,
        dino_input_h=196,
        dino_input_w=252
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[NYUv2SinglePlaneDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train

        self.backbone = backbone

        self.use_coupled_anchor = use_coupled_anchor

        if self.use_coupled_anchor:
            self.anchor_normal_divide_offset = np.load('./cluster_anchor/new_mixed_normal_divide_offset_anchors_7.npy')

        else:
            self.anchor_normals = np.load('./cluster_anchor/new_mixed_normal_anchors_{}.npy'.format(normal_class_num))
            self.anchor_offsets = np.load('./cluster_anchor/new_mixed_offset_anchors_{}.npy'.format(offset_class_num))

        self.large_resolution_input = large_resolution_input
        self.large_resolution_eval = large_resolution_eval

        self.dino_input_h = dino_input_h
        self.dino_input_w = dino_input_w

    @classmethod
    def from_config(cls, cfg, is_train=True):

        tfm_gens = []

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,  # RGB
            'normal_class_num': cfg.MODEL.MASK_FORMER.NORMAL_CLS_NUM,
            'offset_class_num': cfg.MODEL.MASK_FORMER.OFFSET_CLS_NUM,
            'use_coupled_anchor': cfg.MODEL.MASK_FORMER.USE_COUPLED_ANCHOR,
            'backbone': cfg.MODEL.BACKBONE.NAME,
            'large_resolution_input': cfg.INPUT.LARGE_RESOLUTION_INPUT,
            'large_resolution_eval': cfg.INPUT.LARGE_RESOLUTION_EVAL,
            'dino_input_h': cfg.INPUT.DINO_INPUT_HEIGHT,
            'dino_input_w': cfg.INPUT.DINO_INPUT_WIDTH
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        data = np.load(dataset_dict["npz_file_name"])

        if self.large_resolution_input:
            image = data['raw_image']

        else:
            image = data["image"]

        origin_h, origin_w = image.shape[:2]

        utils.check_image_size(dataset_dict, image)

        # the intrinsic corresponding to (640, 480) size
        intrinsic = np.asarray([[518.86, 0, 325.58],
                                [0, 519.47, 253.74],
                                [0, 0, 1]])

        if not self.large_resolution_eval:
            intrinsic[0] = intrinsic[0] * 256 / 640
            intrinsic[1] = intrinsic[1] * 192 / 480

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if self.use_coupled_anchor:
            dataset_dict['anchor_normal_divide_offset'] = torch.as_tensor(self.anchor_normal_divide_offset)

        else:
            dataset_dict['anchor_normals'] = torch.as_tensor(self.anchor_normals)
            dataset_dict['anchor_offsets'] = torch.as_tensor(self.anchor_offsets)

        if not self.is_train:
            if self.backbone == 'DPT_DINOv2':
                image = cv2.resize(image, (self.dino_input_w, self.dino_input_h))
                dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            intrinsic_inv = np.linalg.inv(intrinsic)

            if self.large_resolution_eval:
                dataset_K_inv_dot_xy_1, _ = dataset_precompute_K_inv_dot_xy_1(intrinsic_inv, image_h=480, image_w=640)

            else:
                dataset_K_inv_dot_xy_1, _ = dataset_precompute_K_inv_dot_xy_1(intrinsic_inv, image_h=192, image_w=256)

            dataset_dict["K_inv_dot_xy_1"] = torch.from_numpy(dataset_K_inv_dot_xy_1.astype(np.float32)).reshape(3, -1)

        if "segmentation" in data.keys():
            pan_seg_gt = data["segmentation"] # 0~(num_planes-1) plane, 20 non-plane
            segments_info = dataset_dict["segments_info"]
            plane_depth_gt = data["raw_depth"].astype(np.float32)
            params = data["plane"].astype(np.float32)

            dataset_dict['gt_global_pixel_depth'] = torch.as_tensor(plane_depth_gt)

            if self.large_resolution_eval:
                pan_seg_gt = cv2.resize(pan_seg_gt, (640, 480), interpolation=cv2.INTER_NEAREST)
                plane_depth_gt = cv2.resize(plane_depth_gt, (640, 480), interpolation=cv2.INTER_NEAREST)

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            plane_depths = []

            for segment_info in segments_info:
                label_id = 1 # 1 for plane, 0,2 for non-plane/non-label regions
                if not segment_info["iscrowd"]: # polygons for 0, RLE for 1
                    classes.append(label_id)
                    mask = pan_seg_gt == segment_info["id"]
                    masks.append(mask)
                    plane_depths.append(mask*plane_depth_gt)

            assert len(params) == len(masks)

            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
                instances.gt_params = torch.zeros((0, 3))
                instances.gt_plane_depths = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))

                pixel_normal_map = np.zeros((3, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])).astype(np.float32)
                pixel_offset_map = np.zeros((pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])).astype(np.float32)

            else:
                pixel_normal_map, pixel_offset_map, _ = make_pixel_normal_map(masks, params)

                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()
                plane_depths = torch.stack([torch.from_numpy(x.copy()) for x in plane_depths], dim = 0)
                instances.gt_plane_depths = plane_depths
                instances.gt_params = torch.from_numpy(params)

            dataset_dict["instances"] = instances

            dataset_dict['gt_global_pixel_normal'] = torch.as_tensor(pixel_normal_map)
            dataset_dict['gt_global_pixel_offset'] = torch.as_tensor(pixel_offset_map)

        return dataset_dict


if __name__ == "__main__":
    pass
