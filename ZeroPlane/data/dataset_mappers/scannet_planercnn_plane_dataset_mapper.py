import copy
import numpy as np
import os
import torch
import cv2
import logging

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
from PIL import Image

from detectron2.data import transforms as T
from detectron2.data.transforms.transform import ResizeTransform, CropTransform
from detectron2.data.transforms.augmentation import Augmentation
from fvcore.transforms.transform import PadTransform
from fvcore.transforms.transform import Transform, TransformList
from typing import Any, Callable, List, Optional, TypeVar, Tuple
from detectron2.data.transforms import TransformGen

from ...utils.disp import visualizationBatch
from ...utils.plane_utils import make_pixel_normal_map

from .scannetv1_plane_dataset_mapper import NewFixedSizeCrop


__all__ = ['SingleScannetPlaneRCNNPlaneDatasetMapper']

from PIL import ImageEnhance


def random_brightness(image, min_factor = 0.7, max_factor = 1.2):
    factor = np.random.uniform(min_factor, max_factor)
    image_enhancer_brightness = ImageEnhance.Brightness(Image.fromarray(image))
    return np.array(image_enhancer_brightness.enhance(factor))


def random_color(image, min_factor = 0.5, max_factor = 1.2):
    factor = np.random.uniform(min_factor, max_factor)
    image_enhancer_color = ImageEnhance.Color(Image.fromarray(image))
    return np.array(image_enhancer_color.enhance(factor))


def random_contrast(image, min_factor = 0.7, max_factor = 1.2):
    factor = np.random.uniform(min_factor, max_factor)
    image_enhancer_contrast = ImageEnhance.Contrast(Image.fromarray(image))
    return np.array(image_enhancer_contrast.enhance(factor))


def transforms_apply_intrinsic(transforms, intrinsic, h, w):
    """_summary_

    Args:
        transforms (Transform/TransformList): _description_
        intrinsic (numpy.array): [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    Returns:
        numpy.array: [[fx', 0, cx'], [0, fy', cy'], [0, 0, 1]]
    """

    tfm_intrinsic = np.zeros_like(intrinsic)
    tfm_intrinsic[2,2] = 1

    assert isinstance(transforms, (Transform, TransformList)), (
            f"must input an instance of Transform! Got {type(transforms)} instead."
        )
    # e.g.[ResizeTransform(h=192, w=256, new_h=269, new_w=358, interp=2), CropTransform(x0=73, y0=55, w=256, h=192, orig_w=358, orig_h=269),
    # PadTransform(x0=0, y0=0, x1=0, y1=0, orig_w=256, orig_h=192, pad_value=128.0)]
    tfm_list = transforms.transforms

    random_scale = -1
    x0 = 0
    y0 = 0

    new_w = w
    new_h = h

    for tfm in tfm_list:

        if issubclass(tfm.__class__, ResizeTransform):
            h, w, new_h, new_w = tfm.h, tfm.w, tfm.new_h, tfm.new_w
            random_scale = (new_h/h + new_w/w)/2
            continue

        if issubclass(tfm.__class__, CropTransform):
            x0, y0 = tfm.x0, tfm.y0
            continue

    tfm_intrinsic[0,0] = intrinsic[0,0] * random_scale if random_scale > 0 else intrinsic[0,0]
    tfm_intrinsic[1,1] = intrinsic[1,1] * random_scale if random_scale > 0 else intrinsic[1,1]
    tfm_intrinsic[0,2] = intrinsic[0,2] * random_scale if random_scale > 0 else intrinsic[0,2]
    tfm_intrinsic[1,2] = intrinsic[1,2] * random_scale if random_scale > 0 else intrinsic[1,2]

    tfm_intrinsic[0,2] = tfm_intrinsic[0,2] - x0 # If "resize" is getting larger, the x0,y0 of CropTransform becomes 1,1
    tfm_intrinsic[1,2] = tfm_intrinsic[1,2] - y0

    return tfm_intrinsic, random_scale, new_h, new_w


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    if type(image_size)==int or len(image_size) == 1:
        image_size = [image_size, image_size] if type(image_size)==int else [image_size[0], image_size[0]]
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    # seg and depth cannot be transformed!
    if cfg.INPUT.BRIGHT_COLOR_CONTRAST:
        augmentation.extend([
            T.ColorTransform(
                op = random_brightness
            ),
            T.ColorTransform(
                op = random_color
            ),
            T.ColorTransform(
                op = random_contrast
            ),
        ])

    if cfg.INPUT.RESIZE:

        if cfg.MODEL.BACKBONE.NAME == 'DPT_DINOv2':
            # ensure can be divided by 14
            target_height, target_width = 210, 280

            augmentation.extend([
                T.ResizeScale(
                    min_scale=min_scale, max_scale=max_scale, target_height=target_height, target_width=target_width, interp=Image.NEAREST,
                ), # ! interp = 'nearst' for dataset_K_inv_dot_xy
                NewFixedSizeCrop(crop_size=(target_height, target_width), pad = True, pad_value = 0,
                                 seg_pad_value=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES + 1) # 21
            ])

        else:
            augmentation.extend([
                T.ResizeScale(
                    min_scale=min_scale, max_scale=max_scale, target_height=image_size[0], target_width=image_size[1], interp=Image.NEAREST,
                ), # ! interp = 'nearst' for dataset_K_inv_dot_xy
                # T.FixedSizeCrop(crop_size=(image_size[0], image_size[1]), pad = True, pad_value = 0),
                NewFixedSizeCrop(crop_size=(image_size[0], image_size[1]), pad = True, pad_value = 0,
                                 seg_pad_value=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES + 1) # 21
            ])

    return augmentation


def get_plane_parameters(plane, segmentation):
    plane_nums = plane.shape[0]
    # valid_region = segmentation != 20
    valid_region = segmentation < 20 #! add 21 for pading

    h, w = segmentation.shape

    plane_parameters2 = np.ones((3, h, w))
    for i in range(plane_nums):
        plane_mask = segmentation == i
        plane_mask = plane_mask.astype(np.float32)
        cur_plane_param_map = np.ones((3, h, w)) * plane[i, :].reshape(3, 1, 1)
        plane_parameters2 = plane_parameters2 * (1-plane_mask) + cur_plane_param_map * plane_mask

    return plane_parameters2, valid_region


def dataset_precompute_K_inv_dot_xy_1(K_inv, image_h=192, image_w=256):

    x = torch.arange(image_w, dtype=torch.float32).view(1, image_w)
    y = torch.arange(image_h, dtype=torch.float32).view(image_h, 1)

    xx = x.repeat(image_h, 1)
    yy = y.repeat(1, image_w)
    xy1 = torch.stack((xx, yy, torch.ones((image_h, image_w), dtype=torch.float32)))  # (3, image_h, image_w)

    xy1 = xy1.numpy()

    K_inv_dot_xy_1 = np.einsum('ij,jkl->ikl', K_inv, xy1) # (3, 3) *(3, image_h, image_w) -> (3, image_h, image_w)

    return K_inv_dot_xy_1, xy1


def plane2depth(K_inv_dot_xy_1,  tfm_K_inv_dot_xy_1, S_K_inv_dot_xy_1, dataset_K_inv_dot_xy_1,
                plane_parameters, segmentation, gt_depth, h=192, w=256):

    depth_map = 1. / np.sum(K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
    depth_map = depth_map.reshape(h, w)
    # replace non planer region depth using sensor depth map
    depth_map[segmentation >= 20] = gt_depth[segmentation >= 20]

    tfm_depth_map = 1. / np.sum(tfm_K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
    tfm_depth_map = tfm_depth_map.reshape(h, w)
    # replace non planer region depth using sensor depth map
    tfm_depth_map[segmentation >= 20] = gt_depth[segmentation >= 20]

    S_depth_map = 1. / np.sum(S_K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
    S_depth_map = S_depth_map.reshape(h, w)
    # replace non planer region depth using sensor depth map
    S_depth_map[segmentation >= 20] = gt_depth[segmentation >= 20]

    dataset_depth_map = 1. / np.sum(dataset_K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
    dataset_depth_map = dataset_depth_map.reshape(h, w)
    # replace non planer region depth using sensor depth map
    dataset_depth_map[segmentation >= 20] = gt_depth[segmentation >= 20]

    return depth_map, tfm_depth_map, S_depth_map, dataset_depth_map


def after_transform_apply_K_inv_dot_xy_1(tfm_gt_K_inv_dot_xy_1, tfm_gt_segmentation, tfm_gt_depth, plane,
                                         num_queries, new_h, new_w):

    plane = plane.copy()
    tfm_gt_segmentation = tfm_gt_segmentation.copy()

    plane_parameters, valid_region = \
        get_plane_parameters(plane, tfm_gt_segmentation)

    depth_map = 1. / (np.sum(tfm_gt_K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0) + 1e-8)
    depth_map = depth_map.reshape(new_h, new_w)
    # replace non planer region depth using sensor depth map
    depth_map[tfm_gt_segmentation >= num_queries] = tfm_gt_depth[tfm_gt_segmentation >= num_queries]

    # mean_diff = np.abs(depth_map - tfm_gt_depth).mean()
    # print('mean diff after transformation:', mean_diff)

    # if np.sum(np.abs(depth_map - tfm_gt_depth) > 0.00001) / (new_h * new_w) > 0.1:
    #     print("after_transform_apply_K_inv_dot_xy_1, the error ratio > 0.1")

    tfm_labels = np.unique(tfm_gt_segmentation)
    tfm_labels = tfm_labels[tfm_labels < num_queries] #del non-plane label

    return depth_map, tfm_labels


class SingleScannetPlaneRCNNPlaneDatasetMapper():
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
        num_queries,
        common_stride,
        param_domain_separate,
        mix_outdoor,
        mix_anchor,
        classify_inverse_offset,
        backbone
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[SingleSanpoSyntheticPlaneDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.predict_center = predict_center
        self.num_queries = num_queries

        self.backbone = backbone

        self.common_stride = common_stride

        self.param_domain_separate = param_domain_separate
        self.mix_outdoor = mix_outdoor
        self.mix_anchor = mix_anchor

        self.indoor_anchor_normals = np.load('./cluster_anchor/indoor_normal_anchors.npy')
        self.outdoor_anchor_normals = np.load('./cluster_anchor/outdoor_normal_anchors.npy')

        self.classify_inverse_offset = classify_inverse_offset

        if self.classify_inverse_offset:
            self.indoor_anchor_offsets = np.load('./cluster_anchor/indoor_inverse_offset_anchors.npy')
            self.outdoor_anchor_offsets = np.load('./cluster_anchor/outdoor_inverse_offset_anchors.npy')

        else:
            self.indoor_anchor_offsets = np.load('./cluster_anchor/indoor_offset_anchors.npy')
            self.outdoor_anchor_offsets = np.load('./cluster_anchor/outdoor_offset_anchors.npy')

        if self.mix_anchor:
            self.anchor_normals = np.load('./cluster_anchor/mixed_normal_anchors.npy')
            self.anchor_offsets = np.load('./cluster_anchor/mixed_offset_anchors.npy')

        elif self.mix_outdoor:
            self.anchor_normals = np.load('./cluster_anchor/outdoor_normal_anchors.npy')
            self.anchor_offsets = np.load('./cluster_anchor/outdoor_offset_anchors.npy')

        else:
            self.anchor_normals = np.load('./cluster_anchor/syn_normal_anchors.npy')
            self.anchor_offsets = np.load('./cluster_anchor/syn_offset_anchors.npy')

        self.canonical_focal = 250.0

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        # !
        tfm_gens = build_transform_gen(cfg, is_train) if is_train else []

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT, # RGB
            "predict_center": cfg.MODEL.MASK_FORMER.PREDICT_CENTER,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            'param_domain_separate': cfg.MODEL.MASK_FORMER.PARAM_DOMAIN_SEPARATE,
            'mix_outdoor': cfg.MODEL.MASK_FORMER.MIX_OUTDOOR,
            'mix_anchor': cfg.MODEL.MASK_FORMER.MIX_ANCHOR,
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
        intrinsic = data['intrinsic']

        focal = intrinsic[0][0]
        focal_factor = self.canonical_focal / focal
        dataset_dict['focal_factor'] = torch.as_tensor(focal_factor)

        dataset_dict['dataset_class'] = torch.as_tensor(1)

        utils.check_image_size(dataset_dict, image)

        origin_h, origin_w = image.shape[:2]
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        new_h, new_w = image.shape[:2]
        image_shape = image.shape[:2]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        dataset_dict['anchor_normals_indoor'] = torch.as_tensor(self.indoor_anchor_normals)
        dataset_dict['anchor_offsets_indoor'] = torch.as_tensor(self.indoor_anchor_offsets)

        dataset_dict['anchor_normals_outdoor'] = torch.as_tensor(self.outdoor_anchor_normals)
        dataset_dict['anchor_offsets_outdoor'] = torch.as_tensor(self.outdoor_anchor_offsets)

        dataset_dict['anchor_normals_mixed'] = torch.as_tensor(self.indoor_anchor_normals)
        dataset_dict['anchor_offsets_mixed'] = torch.as_tensor(self.indoor_anchor_offsets)

        dataset_dict['anchor_normals'] = torch.as_tensor(self.indoor_anchor_normals)
        dataset_dict['anchor_offsets'] = torch.as_tensor(self.indoor_anchor_offsets)

        if not self.is_train:
            if self.backbone == 'DPT_DINOv2':
                image = cv2.resize(image, (280, 210))
                dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            intrinsic_inv = np.linalg.inv(intrinsic)
            dataset_K_inv_dot_xy_1, _ = dataset_precompute_K_inv_dot_xy_1(intrinsic_inv)
            dataset_dict['K_inv_dot_xy_1'] = torch.from_numpy(dataset_K_inv_dot_xy_1.astype(np.float32)).reshape(3, -1)

            # # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None)
            # return dataset_dict

        if "segmentation" in data.keys():
            # tfm_intrinsic not used here because the transformed K_inv_dot_xy1 is used directly.
            _, random_scale, _, _ = transforms_apply_intrinsic(transforms, intrinsic, origin_h, origin_w)
            intrinsic_inv = np.linalg.inv(intrinsic)
            dataset_K_inv_dot_xy_1, dataset_xy1 = dataset_precompute_K_inv_dot_xy_1(intrinsic_inv, image_h=origin_h, image_w=origin_w)
            dataset_K_inv_dot_xy_1 = dataset_K_inv_dot_xy_1.transpose(1, 2, 0) # (image_h, image_w, 3)

            dataset_dict["random_scale"] = torch.from_numpy(np.array([random_scale], dtype=np.float32))

            pan_seg_gt = data["segmentation"]  # 0~(num_planes-1) plane, 20 non-plane
            pan_seg_gt = pan_seg_gt.astype(np.uint8)

            segments_info = dataset_dict["segments_info"]
            plane_depth_gt = data["depth"].astype(np.float32).squeeze()  # [h, w]
            params = data["plane"].astype(np.float32)

            # del ColorTransform for seg and depth
            new_transforms = []
            for t in transforms.transforms:
                if t.__class__ != T.ColorTransform:
                    new_transforms.append(t)
            new_transforms = TransformList(new_transforms)

            # apply the same transformation to segmentation
            pan_seg_gt = new_transforms.apply_segmentation(pan_seg_gt) # seg_pad_value = 20+1

            # apply the same transformation to depth.
            # The value of depth remains unchanged
            plane_depth_gt = np.expand_dims(new_transforms.apply_image(plane_depth_gt), axis=0) # [1, h, w]  # interp="nearest"

            # apply the same transformation to dataset_K_inv_dot_xy_1.
            tfm_dataset_K_inv_dot_xy_1 = new_transforms.apply_image(dataset_K_inv_dot_xy_1) # interp = 'bilinear'
            tfm_dataset_K_inv_dot_xy_1 = tfm_dataset_K_inv_dot_xy_1.transpose(2, 0, 1)

            instances = Instances(image_shape)

            raw_depth_gt = data['raw_depth'].astype(np.float32).squeeze()
            tfm_raw_depth_gt = new_transforms.apply_image(raw_depth_gt)
            dataset_dict['gt_global_pixel_depth'] = torch.as_tensor(tfm_raw_depth_gt)

            if self.backbone == 'DPT_DINOv2':
                dsize = (80, 60)

            else:
                dsize = (new_w // 4, new_h // 4)

            dataset_dict['gt_resize14_global_pixel_depth'] = torch.as_tensor(cv2.resize(tfm_raw_depth_gt, dsize, interpolation=cv2.INTER_NEAREST))

            tfm_plane_depth_gt, tfm_labels = after_transform_apply_K_inv_dot_xy_1(tfm_dataset_K_inv_dot_xy_1,
                                                                                  tfm_gt_segmentation = pan_seg_gt,
                                                                                  tfm_gt_depth = plane_depth_gt[0],
                                                                                  plane = params,
                                                                                  num_queries = self.num_queries, new_h = new_h, new_w = new_w)
            tfm_plane_depth_gt = np.expand_dims(tfm_plane_depth_gt, axis=0)

            if self.is_train:
                dataset_dict["K_inv_dot_xy_1"] = torch.from_numpy(tfm_dataset_K_inv_dot_xy_1.astype(np.float32))

            classes = []
            masks = []
            tfm_plane_depths = []
            centers = []

            for segment_info in segments_info:
                # Image enlargement may lead to a reduction in the number of planes
                if segment_info["id"] not in tfm_labels:
                    continue

                label_id = 1 # 1 for plane, 0,2 for non-plane/non-label regions

                if not segment_info["iscrowd"]: # polygons for 0, RLE for 1
                    classes.append(label_id)
                    mask = pan_seg_gt == segment_info["id"]
                    masks.append(mask)
                    tfm_plane_depths.append(mask * tfm_plane_depth_gt)

                    if "center" in segment_info and self.predict_center:
                        centers.append(segment_info["center"])

            params = params[tfm_labels]
            assert len(params) == len(masks)

            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
                instances.gt_params = torch.zeros((0, 3))
                instances.gt_plane_depths = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_pixel_normals = torch.zeros((0, 3, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))

                instances.gt_resize14_plane_depths = torch.zeros((0, dsize[1], dsize[0]))
                instances.gt_resize14_pixel_normals = torch.zeros((0, 3, dsize[1], dsize[0]))

                if "center" in segment_info and self.predict_center:
                    instances.gt_centers = torch.zeros((0, 2))

                pixel_normal_map = np.zeros((3, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])).astype(np.float32)
                pixel_offset_map = np.zeros((pan_seg_gt.shape[-2], pan_seg_gt.shape[-1])).astype(np.float32)

            else:
                pixel_normal_map, pixel_offset_map, tfm_pixel_normals = make_pixel_normal_map(masks, params)

                gt_masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = gt_masks.tensor
                instances.gt_boxes = gt_masks.get_bounding_boxes()
                gt_plane_depths = torch.cat([torch.from_numpy(x.copy().astype(np.float32)) for x in tfm_plane_depths], dim = 0) #! tfm_plane_depths
                instances.gt_plane_depths = gt_plane_depths

                gt_pixel_normals = torch.from_numpy(np.asarray(tfm_pixel_normals))
                instances.gt_pixel_normals = gt_pixel_normals

                gt_resize14_plane_depths = []

                for d, m in zip(tfm_plane_depths, masks):
                    d_ = cv2.resize(d[0].copy(), dsize, interpolation=cv2.INTER_AREA)
                    m_ = cv2.resize(m.copy().astype(np.float32), dsize, interpolation=cv2.INTER_NEAREST)
                    gt_resize14_plane_depths.append(torch.from_numpy(d_*m_))

                gt_resize14_plane_depths = torch.stack(gt_resize14_plane_depths, dim = 0)
                instances.gt_resize14_plane_depths = gt_resize14_plane_depths

                gt_resize14_pixel_normals = []
                for n, m in zip(tfm_pixel_normals, masks):
                    n_ = cv2.resize(n.copy().transpose(1, 2, 0), dsize, interpolation=cv2.INTER_AREA)
                    m_ = cv2.resize(m.copy().astype(np.float32), dsize, interpolation=cv2.INTER_NEAREST)
                    gt_resize14_pixel_normals.append(torch.from_numpy(n_*m_[..., None]))

                gt_resize14_pixel_normals = torch.stack(gt_resize14_pixel_normals, dim=0).permute(0, 3, 1, 2)
                instances.gt_resize14_pixel_normals = gt_resize14_pixel_normals

                instances.gt_params = torch.from_numpy(params)
                if "center" in segment_info and self.predict_center:
                    instances.gt_centers = torch.from_numpy(np.array(centers).astype(np.float32))

            dataset_dict["instances"] = instances

            dataset_dict['gt_global_pixel_normal'] = torch.as_tensor(pixel_normal_map)
            dataset_dict['gt_resize14_global_pixel_normal'] = torch.as_tensor(cv2.resize(pixel_normal_map.transpose(1, 2, 0), dsize, interpolation=cv2.INTER_NEAREST)).permute(2, 0, 1)

            dataset_dict['gt_global_pixel_offset'] = torch.as_tensor(pixel_offset_map)
            dataset_dict['gt_resize14_global_pixel_offset'] = torch.as_tensor(cv2.resize(pixel_offset_map, dsize, interpolation=cv2.INTER_NEAREST))

        return dataset_dict

if __name__ == "__main__":
    pass
