import copy
import numpy as np
import os
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

__all__ = ['SingleWildDataPlaneDatasetMapper']


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


class SingleWildDataPlaneDatasetMapper():
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
        predict_center,
        mix_anchor,
        normal_class_num,
        offset_class_num,
        classify_inverse_offset,
        backbone
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
            "[WildDataSinglePlaneDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.predict_center = predict_center

        self.backbone = backbone

        self.mix_anchor = mix_anchor

        self.classify_inverse_offset = classify_inverse_offset

        self.indoor_anchor_normals = np.load('./cluster_anchor/new_indoor_mixed_normal_anchors_{}.npy'.format(normal_class_num))
        self.outdoor_anchor_normals = np.load('./cluster_anchor/new_outdoor_mixed_normal_anchors_{}.npy'.format(normal_class_num))

        if self.classify_inverse_offset:
            raise NotImplementedError

        else:
            self.indoor_anchor_offsets = np.load('./cluster_anchor/new_indoor_mixed_offset_anchors_{}.npy'.format(offset_class_num))
            self.outdoor_anchor_offsets = np.load('./cluster_anchor/new_outdoor_mixed_offset_anchors_{}.npy'.format(offset_class_num))

        # else:
        if self.mix_anchor:
            self.anchor_normals = np.load('./cluster_anchor/new_mixed_normal_anchors_{}.npy'.format(normal_class_num))
            self.anchor_offsets = np.load('./cluster_anchor/new_mixed_offset_anchors_{}.npy'.format(offset_class_num))

        else:
            raise NotImplementedError

        self.canonical_focal = 250.0

    @classmethod
    def from_config(cls, cfg, is_train=True):

        tfm_gens = []

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT, # RGB
            "predict_center": cfg.MODEL.MASK_FORMER.PREDICT_CENTER,
            'mix_anchor': cfg.MODEL.MASK_FORMER.MIX_ANCHOR,
            'normal_class_num': cfg.MODEL.MASK_FORMER.NORMAL_CLS_NUM,
            'offset_class_num': cfg.MODEL.MASK_FORMER.OFFSET_CLS_NUM,
            'classify_inverse_offset': cfg.MODEL.MASK_FORMER.CLASSIFY_INVERSE_OFFSET,
            'backbone': cfg.MODEL.BACKBONE.NAME
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
        image = data["image"]
        utils.check_image_size(dataset_dict, image)

        intrinsic = data['intrinsic']

        focal = intrinsic[0][0]
        focal_factor = self.canonical_focal / focal
        dataset_dict['focal_factor'] = torch.as_tensor(focal_factor)

        dataset_dict['dataset_class'] = torch.as_tensor(1)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        dataset_dict['anchor_normals_indoor'] = torch.as_tensor(self.indoor_anchor_normals)
        dataset_dict['anchor_offsets_indoor'] = torch.as_tensor(self.indoor_anchor_offsets)

        dataset_dict['anchor_normals_outdoor'] = torch.as_tensor(self.outdoor_anchor_normals)
        dataset_dict['anchor_offsets_outdoor'] = torch.as_tensor(self.outdoor_anchor_offsets)

        # else:
        dataset_dict['anchor_normals'] = torch.as_tensor(self.anchor_normals)
        dataset_dict['anchor_offsets'] = torch.as_tensor(self.anchor_offsets)

        if not self.is_train:
            if self.backbone == 'DPT_DINOv2':
                image = cv2.resize(image, (280, 210))
                dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            intrinsic_inv = np.linalg.inv(intrinsic)
            dataset_K_inv_dot_xy_1, _ = dataset_precompute_K_inv_dot_xy_1(intrinsic_inv)
            dataset_dict["K_inv_dot_xy_1"] = torch.from_numpy(dataset_K_inv_dot_xy_1.astype(np.float32)).reshape(3, -1)

            # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None)
            # return dataset_dict

        return dataset_dict


if __name__ == "__main__":
    pass
