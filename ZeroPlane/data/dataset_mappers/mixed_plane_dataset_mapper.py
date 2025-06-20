import copy
import numpy as np
import os
import os.path as osp
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


__all__ = ['SingleMixedPlaneDatasetMapper']

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

    new_h = h
    new_w = w

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


def build_transform_gen(cfg, is_train, large_resolution_input=False, dino_input_h=196, dino_input_w=252, unchanged_aspect_ratio_then_crop=False):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"

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
        # ensure can divide by 14
        if cfg.MODEL.BACKBONE.NAME == 'DPT_DINOv2':
            if unchanged_aspect_ratio_then_crop:
                assert dino_input_h >= 518
                assert dino_input_w >= 518

                target_height, target_width = dino_input_h, dino_input_w
                crop_height, crop_width = 518, 518

            else:
                target_height, target_width = dino_input_h, dino_input_w
                crop_height, crop_width = target_height, target_width

            # if large_resolution_input:
            #     target_height, target_width = 518, 518

            # else:
            #     target_height, target_width = 196, 252

        else:
            if large_resolution_input:
                target_height, target_width = 480, 640

            else:
                target_height, target_width = 192, 256

            crop_height, crop_width = target_height, target_width

        # resizescale: the real scale will be target_h * (min_scale, max_scale), then crop to a fixed size
        augmentation.extend([
            T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=target_height, target_width=target_width, interp=Image.NEAREST,
            ),  # ! interp = 'nearst' for dataset_K_inv_dot_xy
            NewFixedSizeCrop(crop_size=(crop_height, crop_width), pad=True, pad_value=0,
                             seg_pad_value=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES + 1)  # 21
        ])

    return augmentation


def get_plane_parameters(plane, segmentation):
    plane_nums = plane.shape[0]
    # valid_region = segmentation != 20
    valid_region = segmentation < 20  #! add 21 for pading

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


def after_transform_apply_K_inv_dot_xy_1(tfm_gt_K_inv_dot_xy_1, tfm_gt_segmentation, tfm_gt_depth, plane,
                                         num_queries, new_h, new_w, file_name):

    plane = plane.copy()
    tfm_gt_segmentation = tfm_gt_segmentation.copy()

    plane_parameters, valid_region = \
        get_plane_parameters(plane, tfm_gt_segmentation)

    depth_map = 1. / (np.sum(tfm_gt_K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0) + 1e-8)
    depth_map = depth_map.reshape(new_h, new_w)
    # replace non planer region depth using sensor depth map
    depth_map[tfm_gt_segmentation >= num_queries] = tfm_gt_depth[tfm_gt_segmentation >= num_queries]

    tfm_labels = np.unique(tfm_gt_segmentation)
    tfm_labels = tfm_labels[tfm_labels < num_queries]

    return depth_map, tfm_labels


class SingleMixedPlaneDatasetMapper():
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
        use_partial_cluster,
        use_indoor_anchor,
        use_outdoor_anchor,
        mix_anchor,
        normal_class_num,
        offset_class_num,
        use_coupled_anchor,
        classify_inverse_offset,
        backbone,
        with_nonplanar_query,
        large_resolution_input=False,
        large_resolution_eval=False,
        dino_input_h=196,
        dino_input_w=252,
        unchanged_aspect_ratio_then_crop=False
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
            "[SingleMixedPlaneDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.predict_center = predict_center
        self.num_queries = num_queries

        self.backbone = backbone
        self.with_nonplanar_query = with_nonplanar_query

        self.common_stride = common_stride
        self.classify_inverse_offset = classify_inverse_offset

        if not osp.exists('./cluster_anchor/new_indoor_mixed_normal_anchors_{}.npy'.format(normal_class_num)):
            self.indoor_anchor_normals = np.load('./cluster_anchor/new_indoor_mixed_normal_anchors_7.npy')
            self.outdoor_anchor_normals = np.load('./cluster_anchor/new_outdoor_mixed_normal_anchors_7.npy')

        else:
            self.indoor_anchor_normals = np.load('./cluster_anchor/new_indoor_mixed_normal_anchors_{}.npy'.format(normal_class_num))
            self.outdoor_anchor_normals = np.load('./cluster_anchor/new_outdoor_mixed_normal_anchors_{}.npy'.format(normal_class_num))

        if self.classify_inverse_offset:
            raise NotImplementedError

        else:
            if not osp.exists('./cluster_anchor/new_indoor_mixed_offset_anchors_{}.npy'.format(offset_class_num)):
                self.indoor_anchor_offsets = np.load('./cluster_anchor/new_indoor_mixed_offset_anchors_20.npy')
                self.outdoor_anchor_offsets = np.load('./cluster_anchor/new_outdoor_mixed_offset_anchors_20.npy')

            else:
                self.indoor_anchor_offsets = np.load('./cluster_anchor/new_indoor_mixed_offset_anchors_{}.npy'.format(offset_class_num))
                self.outdoor_anchor_offsets = np.load('./cluster_anchor/new_outdoor_mixed_offset_anchors_{}.npy'.format(offset_class_num))

        self.use_indoor_anchor = use_indoor_anchor
        self.use_outdoor_anchor = use_outdoor_anchor

        assert not (self.use_indoor_anchor and self.use_outdoor_anchor)

        self.mix_anchor = mix_anchor

        assert not (self.mix_anchor and self.use_indoor_anchor)
        assert not (self.mix_anchor and self.use_outdoor_anchor)

        self.use_coupled_anchor = use_coupled_anchor

        if use_partial_cluster:
            self.anchor_normals = np.load('./cluster_anchor/partial_new_mixed_normal_anchors_{}.npy'.format(normal_class_num))
            self.anchor_offsets = np.load('./cluster_anchor/partial_new_mixed_offset_anchors_{}.npy'.format(offset_class_num))

        elif use_indoor_anchor:
            self.anchor_normals = np.load('./cluster_anchor/new_indoor_mixed_normal_anchors_{}.npy'.format(normal_class_num))
            self.anchor_offsets = np.load('./cluster_anchor/new_indoor_mixed_offset_anchors_{}.npy'.format(offset_class_num))

        elif use_outdoor_anchor:
            self.anchor_normals = np.load('./cluster_anchor/new_outdoor_mixed_normal_anchors_{}.npy'.format(normal_class_num))
            self.anchor_offsets = np.load('./cluster_anchor/new_outdoor_mixed_offset_anchors_{}.npy'.format(offset_class_num))

        else:
            # assert self.mix_anchor

            if self.use_coupled_anchor:
                self.anchor_normal_divide_offset = np.load('./cluster_anchor/new_mixed_normal_divide_offset_anchors_7.npy')

            else:
                self.anchor_normals = np.load('./cluster_anchor/new_mixed_normal_anchors_{}.npy'.format(normal_class_num))
                self.anchor_offsets = np.load('./cluster_anchor/new_mixed_offset_anchors_{}.npy'.format(offset_class_num))

        self.canonical_focal = 250.0

        self.large_resolution_input = large_resolution_input
        self.large_resolution_eval = large_resolution_eval

        self.dino_input_h = dino_input_h
        self.dino_input_w = dino_input_w

        self.unchanged_aspect_ratio_then_crop = unchanged_aspect_ratio_then_crop

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        # !
        dino_input_h = cfg.INPUT.DINO_INPUT_HEIGHT
        dino_input_w = cfg.INPUT.DINO_INPUT_WIDTH
        unchanged_aspect_ratio_then_crop = cfg.INPUT.DINO_UNCHANGED_ASPECT_RATIO

        tfm_gens = build_transform_gen(cfg, is_train, large_resolution_input=cfg.INPUT.LARGE_RESOLUTION_INPUT, dino_input_h=dino_input_h, dino_input_w=dino_input_w, unchanged_aspect_ratio_then_crop=unchanged_aspect_ratio_then_crop) if is_train else []

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT, # RGB
            "predict_center": cfg.MODEL.MASK_FORMER.PREDICT_CENTER,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "common_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            'use_partial_cluster': cfg.MODEL.MASK_FORMER.USE_PARTIAL_CLUSTER,
            'use_indoor_anchor': cfg.MODEL.MASK_FORMER.USE_INDOOR_ANCHOR,
            'use_outdoor_anchor': cfg.MODEL.MASK_FORMER.USE_OUTDOOR_ANCHOR,
            'mix_anchor': cfg.MODEL.MASK_FORMER.MIX_ANCHOR,
            'normal_class_num': cfg.MODEL.MASK_FORMER.NORMAL_CLS_NUM,
            'offset_class_num': cfg.MODEL.MASK_FORMER.OFFSET_CLS_NUM,
            'use_coupled_anchor': cfg.MODEL.MASK_FORMER.USE_COUPLED_ANCHOR,
            'classify_inverse_offset': cfg.MODEL.MASK_FORMER.CLASSIFY_INVERSE_OFFSET,
            'backbone': cfg.MODEL.BACKBONE.NAME,
            'with_nonplanar_query': cfg.MODEL.MASK_FORMER.WITH_NONPLANAR_QUERY,
            'large_resolution_input': cfg.INPUT.LARGE_RESOLUTION_INPUT,
            'large_resolution_eval': cfg.INPUT.LARGE_RESOLUTION_EVAL,
            'dino_input_h': cfg.INPUT.DINO_INPUT_HEIGHT,
            'dino_input_w': cfg.INPUT.DINO_INPUT_WIDTH,
            'unchanged_aspect_ratio_then_crop': unchanged_aspect_ratio_then_crop
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
            if len(image.shape) > 3:
                image = image.squeeze(axis=0)

            if image.shape != (480, 640, 3):
                print(dataset_dict['npz_file_name'])
                image = cv2.resize(image, (640, 480))

        else:
            image = data["image"]

        intrinsic = data['intrinsic']

        if self.backbone == 'DPT_DINOv2' and self.large_resolution_input:
            intrinsic[0] = intrinsic[0] * self.dino_input_w / 256
            intrinsic[1] = intrinsic[1] * self.dino_input_h / 192

        dataset_name = dataset_dict['npz_file_name'].split('/')[1]

        if dataset_name in ['scannetv1_plane', 'mp3d_plane', 'origin_nyuv2_plane', 'scannet_planercnn_plane', 'new_nyuv2_plane', 'diode_plane', 'taskonomy_plane', 'sevenscenes_plane', 'replica_hm3d_plane']:
            dataset_cls = 1

        else:
            dataset_cls = 0

        dataset_dict['dataset_class'] = torch.as_tensor(dataset_cls)
        origin_h, origin_w = image.shape[:2]

        utils.check_image_size(dataset_dict, image)

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

        if self.mix_anchor:
            if self.use_coupled_anchor:
                dataset_dict['anchor_normal_divide_offset'] = torch.as_tensor(self.anchor_normal_divide_offset)

            else:
                dataset_dict['anchor_normals'] = torch.as_tensor(self.anchor_normals)
                dataset_dict['anchor_offsets'] = torch.as_tensor(self.anchor_offsets)

        else:
            if self.use_indoor_anchor:
                dataset_dict['anchor_normals'] = torch.as_tensor(self.indoor_anchor_normals)
                dataset_dict['anchor_offsets'] = torch.as_tensor(self.indoor_anchor_offsets)

            elif self.use_outdoor_anchor:
                dataset_dict['anchor_normals'] = torch.as_tensor(self.outdoor_anchor_normals)
                dataset_dict['anchor_offsets'] = torch.as_tensor(self.outdoor_anchor_offsets)

            # assume we known each data is indoor or outdoor during testing
            else:
                dataset_dict['anchor_normals'] = torch.as_tensor(self.indoor_anchor_normals) if dataset_cls == 1 \
                    else torch.as_tensor(self.outdoor_anchor_normals)
                dataset_dict['anchor_offsets'] = torch.as_tensor(self.indoor_anchor_offsets) if dataset_cls == 1 \
                    else torch.as_tensor(self.outdoor_anchor_offsets)

        if not self.is_train:
            if self.backbone == 'DPT_DINOv2':
                image = cv2.resize(image, (self.dino_input_w, self.dino_input_h))

                # no transforms during testing, so we need to manually resize the input image to match the shape
                # if self.large_resolution_model:
                #     image = cv2.resize(image, (518, 518))

                # else:
                #     image = cv2.resize(image, (252, 196))

                dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            intrinsic_inv = np.linalg.inv(intrinsic)
            dataset_K_inv_dot_xy_1, _ = dataset_precompute_K_inv_dot_xy_1(intrinsic_inv, image_h=origin_h, image_w=origin_w)
            dataset_dict['K_inv_dot_xy_1'] = torch.from_numpy(dataset_K_inv_dot_xy_1.astype(np.float32)).reshape(3, -1)

            # USER: Modify this if you want to keep them for some reason.
            # dataset_dict.pop("annotations", None)
            # return dataset_dict

        if "segmentation" in data.keys():
            # tfm_intrinsic not used here because the transformed K_inv_dot_xy1 is used directly.
            _, random_scale, _, _ = transforms_apply_intrinsic(transforms, intrinsic, origin_h, origin_w)
            intrinsic_inv = np.linalg.inv(intrinsic)

            # origin intrinsic before resize
            dataset_K_inv_dot_xy_1, dataset_xy1 = dataset_precompute_K_inv_dot_xy_1(intrinsic_inv, image_h=origin_h, image_w=origin_w)
            dataset_K_inv_dot_xy_1 = dataset_K_inv_dot_xy_1.transpose(1, 2, 0)  # (image_h, image_w, 3)

            dataset_dict["random_scale"] = torch.from_numpy(np.array([random_scale], dtype=np.float32))

            pan_seg_gt = data["segmentation"]  # 0~(num_planes-1) plane, 20 non-plane
            if self.large_resolution_input:
                pan_seg_gt = cv2.resize(pan_seg_gt, (640, 480), interpolation=cv2.INTER_NEAREST)

            pan_seg_gt = pan_seg_gt.astype(np.uint8)

            plane_depth_gt = data["depth"].astype(np.float32).squeeze() # [h, w]
            if self.large_resolution_input:
                plane_depth_gt = cv2.resize(plane_depth_gt, (640, 480), interpolation=cv2.INTER_NEAREST)

            # del ColorTransform for seg and depth
            new_transforms = []
            for t in transforms.transforms:
                if t.__class__ != T.ColorTransform:
                    new_transforms.append(t)
            new_transforms = TransformList(new_transforms)

            # print(pan_seg_gt.shape, new_transforms)
            # apply the same transformation to segmentation
            try:
                pan_seg_gt = new_transforms.apply_segmentation(pan_seg_gt) # seg_pad_value = 20+1

            except:
                print(pan_seg_gt.shape, new_transforms, dataset_dict['npz_file_name'])
                exit(1)

            # apply the same transformation to depth.
            # The value of depth remains unchanged
            plane_depth_gt = np.expand_dims(new_transforms.apply_image(plane_depth_gt), axis=0) # [1, h, w]  # interp="nearest"

            # apply the same transformation to dataset_K_inv_dot_xy_1.
            tfm_dataset_K_inv_dot_xy_1 = new_transforms.apply_image(dataset_K_inv_dot_xy_1) # interp = 'bilinear'
            tfm_dataset_K_inv_dot_xy_1 = tfm_dataset_K_inv_dot_xy_1.transpose(2, 0, 1)

            instances = Instances(image_shape)

            raw_depth_gt = data['raw_depth'].astype(np.float32).squeeze()
            if self.large_resolution_input:
                raw_depth_gt = cv2.resize(raw_depth_gt, (640, 480), interpolation=cv2.INTER_NEAREST)

            tfm_raw_depth_gt = new_transforms.apply_image(raw_depth_gt)
            dataset_dict['gt_global_pixel_depth'] = torch.as_tensor(tfm_raw_depth_gt)

            if self.backbone == 'DPT_DINOv2':
                # if self.large_resolution_input:
                #     dsize = (148, 148)

                # else:
                #     dsize = (72, 56)

                if self.unchanged_aspect_ratio_then_crop:
                    dsize = (148, 148)

                else:
                    dsize = (int(self.dino_input_w / 3.5), int(self.dino_input_h / 3.5))

            else:
                dsize = (new_w // 4, new_h // 4)

            dataset_dict['gt_resize14_global_pixel_depth'] = torch.as_tensor(cv2.resize(tfm_raw_depth_gt, dsize, interpolation=cv2.INTER_NEAREST))

            params = data["plane"].astype(np.float32)
            tfm_plane_depth_gt, tfm_labels = after_transform_apply_K_inv_dot_xy_1(tfm_dataset_K_inv_dot_xy_1,
                                                                                  tfm_gt_segmentation = pan_seg_gt,
                                                                                  tfm_gt_depth = plane_depth_gt[0],
                                                                                  plane = params,
                                                                                  num_queries = self.num_queries,
                                                                                  new_h = new_h, new_w = new_w,
                                                                                  file_name = dataset_dict['npz_file_name'])
            tfm_plane_depth_gt = np.expand_dims(tfm_plane_depth_gt, axis=0)

            if self.is_train:
                dataset_dict["K_inv_dot_xy_1"] = torch.from_numpy(tfm_dataset_K_inv_dot_xy_1.astype(np.float32))

            classes = []
            masks = []
            tfm_plane_depths = []
            centers = []

            segments_info = dataset_dict["segments_info"]
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

            if self.with_nonplanar_query:
                classes.append(0)

                if len(masks) == 0:
                    mask = np.ones_like(tfm_plane_depth_gt.squeeze(0)) > 0

                else:
                    mask = (1 - (np.asarray(masks).sum(axis=0) > 0)) > 0

                masks.append(mask)

                tfm_plane_depths.append(mask * tfm_plane_depth_gt)

                if len(params) == 0:
                    params = np.zeros((1, 3)).astype(params.dtype)

                else:
                    params = np.concatenate([params, np.zeros_like(params[0])[None]])

                if self.predict_center:
                    centers.append(np.zeros_like(centers[0]))

            if not len(params) == len(masks):
                print(len(params), len(masks), dataset_dict['npz_file_name'])
                exit(1)

            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
                instances.gt_params = torch.zeros((0, 3))
                instances.gt_plane_depths = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))

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
