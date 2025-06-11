import copy
import numpy as np
import os
import torch
import cv2

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

import logging
from detectron2.data import transforms as T
from detectron2.data.transforms.transform import ResizeTransform, CropTransform
from detectron2.data.transforms.augmentation import Augmentation
from fvcore.transforms.transform import PadTransform
from fvcore.transforms.transform import Transform, TransformList
from typing import Any, Callable, List, Optional, TypeVar, Tuple
from detectron2.data.transforms import TransformGen

from ...utils.disp import visualizationBatch
from ...utils.plane_utils import make_pixel_normal_map


__all__ = ['SingleDIODEPlaneDatasetMapper']

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


class NewFixedSizeCrop(Augmentation):

    def __init__(self, crop_size: Tuple[int], pad: bool = True, pad_value: float = 128.0, seg_pad_value: float = 0.0):
        """
        Args:
            crop_size: target image (height, width).
            pad: if True, will pad images smaller than `crop_size` up to `crop_size`
            pad_value: the padding value.
            seg_pad_value #!add seg_pad_value
        """
        super().__init__()
        self._init(locals())

    def _get_crop(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        max_offset = np.subtract(input_size, output_size)
        max_offset = np.maximum(max_offset, 0)
        # offset = np.multiply(max_offset, np.random.uniform(0.0, 1.0)) #! del random crop
        offset = max_offset/2
        offset = np.round(offset).astype(int)
        return CropTransform(
            offset[1], offset[0], output_size[1], output_size[0], input_size[1], input_size[0]
        )

    def _get_pad(self, image: np.ndarray) -> Transform:
        # Compute the image scale and scaled size.
        input_size = image.shape[:2]
        output_size = self.crop_size

        # Add padding if the image is scaled down.
        pad_size = np.subtract(output_size, input_size)
        pad_size = np.maximum(pad_size, 0)
        offset0 = np.round(pad_size/2).astype(int)
        offset1 = pad_size - offset0
        original_size = np.minimum(input_size, output_size)
        return PadTransform(
            offset0[1], offset0[0], offset1[1], offset1[0], original_size[1], original_size[0], self.pad_value, self.seg_pad_value
        ) #! add seg_pad_value

    def get_transform(self, image: np.ndarray) -> TransformList:
        transforms = [self._get_crop(image)]
        if self.pad:
            transforms.append(self._get_pad(image))
        return TransformList(transforms)


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

        augmentation.extend([
            T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=image_size[0], target_width=image_size[1], interp=Image.NEAREST,
            ), # ! interp = 'nearst' for dataset_K_inv_dot_xy
            # T.FixedSizeCrop(crop_size=(image_size[0], image_size[1]), pad = True, pad_value = 0),
            NewFixedSizeCrop(crop_size=(image_size[0], image_size[1]), pad = True, pad_value = 0,
                                seg_pad_value=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES + 1) # 21
        ])

    return augmentation


############################### test transforms_apply_intrinsic #########################################################################
def get_plane_parameters(plane, plane_nums, segmentation):
    # valid_region = segmentation != 20
    valid_region = segmentation < 20 #! add 21 for pading

    plane = plane[:plane_nums]

    h, w = segmentation.shape

    plane_parameters2 = np.ones((3, h, w))
    for i in range(plane_nums):
        plane_mask = segmentation == i
        plane_mask = plane_mask.astype(np.float32)
        cur_plane_param_map = np.ones((3, h, w)) * plane[i, :].reshape(3, 1, 1)
        plane_parameters2 = plane_parameters2 * (1-plane_mask) + cur_plane_param_map * plane_mask

    return plane_parameters2, valid_region


def precompute_K_inv_dot_xy_1(tfm_K_inv, K_inv, new_h, new_w, image_h=192, image_w=256):
    # if tfm_K_inv != None and K_inv == None:

    x = torch.arange(image_w, dtype=torch.float32).view(1, image_w) * new_w / image_w
    y = torch.arange(image_h, dtype=torch.float32).view(image_h, 1) * new_h / image_h

    xx = x.repeat(image_h, 1)
    yy = y.repeat(1, image_w)
    xy1 = torch.stack((xx, yy, torch.ones((image_h, image_w), dtype=torch.float32)))  # (3, image_h, image_w)

    xy1 = xy1.view(3, -1)  # (3, image_h*image_w)
    xy1 = xy1.numpy()

    tfm_K_inv_dot_xy_1 = np.dot(tfm_K_inv, xy1)

    # elif tfm_K_inv == None and K_inv != None:

    x = torch.arange(image_w, dtype=torch.float32).view(1, image_w)
    y = torch.arange(image_h, dtype=torch.float32).view(image_h, 1)

    xx = x.repeat(image_h, 1)
    yy = y.repeat(1, image_w)
    xy1 = torch.stack((xx, yy, torch.ones((image_h, image_w), dtype=torch.float32)))  # (3, image_h, image_w)

    xy1 = xy1.view(3, -1)  # (3, image_h*image_w)
    xy1 = xy1.numpy()

    K_inv_dot_xy_1 = np.dot(K_inv, xy1)

    x = torch.arange(new_w, dtype=torch.float32).view(1, new_w)
    y = torch.arange(new_h, dtype=torch.float32).view(new_h, 1)

    xx = x.repeat(new_h, 1)
    yy = y.repeat(1, new_w)
    new_xy1 = torch.stack((xx, yy, torch.ones((new_h, new_w), dtype=torch.float32)))  # (3, new_h, new_w)

    if new_h < image_h and new_w < image_w:
        xy1 = torch.zeros((3, image_h, image_w))
        # Add random crop if the image is scaled up.
        max_offset = np.array([image_h - new_h, image_w - new_w])
        crop_offset = max_offset / 2
        crop_offset = np.round(crop_offset).astype(int)
        xy1[:, crop_offset[0]:(crop_offset[0]+new_h), crop_offset[1]:(crop_offset[1]+new_w)] = new_xy1
        xy1[2] = 1.0
    elif new_h > image_h and new_w > image_w:
        xy1 = torch.zeros((3, image_h, image_w))
        xy1 = new_xy1[:, :image_h, :image_w]
    elif new_h == image_h and new_w == image_w:
        xy1 = new_xy1
    else:
        print("Error2!!!!!!!!!!!!!!!!!!!")

    # xy1 = xy1.view(3, -1)  # (3, new_h*image_w)
    xy1 = xy1.reshape(3, -1)
    xy1 = xy1.numpy()

    S_K_inv_dot_xy_1 = np.dot(K_inv, xy1) #! K_inv

    return K_inv_dot_xy_1, tfm_K_inv_dot_xy_1, S_K_inv_dot_xy_1


def dataset_precompute_K_inv_dot_xy_1(K_inv, image_h=192, image_w=256):

    # elif tfm_K_inv == None and K_inv != None:

    x = torch.arange(image_w, dtype=torch.float32).view(1, image_w)
    y = torch.arange(image_h, dtype=torch.float32).view(image_h, 1)

    xx = x.repeat(image_h, 1)
    yy = y.repeat(1, image_w)
    xy1 = torch.stack((xx, yy, torch.ones((image_h, image_w), dtype=torch.float32)))  # (3, image_h, image_w)

    # xy1 = xy1.view(3, -1)  # (3, image_h*image_w)
    xy1 = xy1.numpy()

    K_inv_dot_xy_1 = np.einsum('ij,jkl->ikl', K_inv, xy1) # (3, 3) *(3, image_h, image_w) -> (3, image_h, image_w)

    return K_inv_dot_xy_1, xy1


def plane2depth(K_inv_dot_xy_1,  tfm_K_inv_dot_xy_1, S_K_inv_dot_xy_1, dataset_K_inv_dot_xy_1,
                plane_parameters, num_planes, segmentation, gt_depth, h=192, w=256):

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


def test_tfm_K(dataset_K_inv_dot_xy_1,
                idx, tfm_image, image, plane, tfm_K_inv, K_inv, tfm_depth, gt_depth,
               tfm_segmentation, gt_segmentation, num_planes,
               new_h, new_w, image_h=192, image_w=256):
    error_ths = np.linspace(0.0, 0.01, 20).reshape(1, -1)

    plane = plane.copy()
    plane_parameters, valid_region = \
        get_plane_parameters(plane, num_planes, tfm_segmentation)

    # K_inv_dot_xy_1 = np.dot(tfm_K_inv, xy1) # (3,3) * (3, image_h*image_w) = (3, image_h*image_w)
    K_inv_dot_xy_1, tfm_K_inv_dot_xy_1, S_K_inv_dot_xy_1 = precompute_K_inv_dot_xy_1(tfm_K_inv, K_inv, new_h, new_w, image_h, image_w)

    plane_depth, tfm_plane_depth, S_plane_depth, dataset_plane_depth = plane2depth(K_inv_dot_xy_1, tfm_K_inv_dot_xy_1, S_K_inv_dot_xy_1, dataset_K_inv_dot_xy_1,
                                                              plane_parameters, num_planes, tfm_segmentation, tfm_depth)
    plane_depth, tfm_plane_depth, S_plane_depth, dataset_plane_depth = plane_depth.reshape(192, 256), tfm_plane_depth.reshape(192, 256), S_plane_depth.reshape(192, 256), dataset_plane_depth.reshape(192, 256)
    delta = np.abs(plane_depth - tfm_depth).reshape(-1, 1)
    delta_tfm = np.abs(tfm_plane_depth - tfm_depth).reshape(-1, 1)
    delta_S = np.abs(S_plane_depth - tfm_depth).reshape(-1, 1)
    delta_dataset = np.abs(dataset_plane_depth - tfm_depth).reshape(-1, 1)
    error_nums = np.sum(delta > error_ths, axis = 0)
    error_nums_tfm = np.sum(delta_tfm > error_ths, axis = 0)
    error_nums_S = np.sum(delta_S > error_ths, axis = 0)
    error_nums_dataset = np.sum(delta_dataset > error_ths, axis = 0)
    if error_nums[-1] > 0:
        print("!!!!! th =" + str(error_ths[0][-1]) + " , error_nums = " + str(error_nums[-1]) )
        print(error_ths)
        print(error_nums)

    plane_vis_path = "./apollo_tmp_vis/"

    gt_vis_dict = {
                    'image': image.copy(),
                    'segmentation': gt_segmentation,
                    'K_inv_dot_xy_1': K_inv_dot_xy_1,
                    'depth': gt_depth,
                    }

    visualizationBatch(root_path = plane_vis_path, idx = idx, info = "gt",
                        data_dict = gt_vis_dict, num_queries = 20, save_image = True, save_segmentation = True,
                        save_depth = True, save_ply = True, save_cloud = False)

    tfm_vis_dict = {
                    'image': tfm_image.copy(),
                    'segmentation': tfm_segmentation,
                    'K_inv_dot_xy_1': K_inv_dot_xy_1,
                    'depth': dataset_plane_depth,
                    }

    visualizationBatch(root_path = plane_vis_path, idx = idx, info = "tfm",
                        data_dict = tfm_vis_dict, num_queries = 20, save_image = True, save_segmentation = True,
                        save_depth = True, save_ply = True, save_cloud = False)

    return dataset_plane_depth

###################################### test transforms_apply_intrinsic ###############################################

def after_transform_apply_K_inv_dot_xy_1(tfm_gt_K_inv_dot_xy_1, tfm_gt_segmentation, tfm_gt_depth, plane,
                                         num_planes, num_queries, new_h, new_w, image_h=192, image_w=256):

    plane = plane.copy()
    tfm_gt_segmentation = tfm_gt_segmentation.copy()

    plane_parameters, valid_region = \
        get_plane_parameters(plane, num_planes, tfm_gt_segmentation)

    depth_map = 1. / np.sum(tfm_gt_K_inv_dot_xy_1.reshape(3, -1) * plane_parameters.reshape(3, -1), axis=0)
    depth_map = depth_map.reshape(image_h, image_w)
    # replace non planer region depth using sensor depth map
    depth_map[tfm_gt_segmentation >= num_queries] = tfm_gt_depth[tfm_gt_segmentation >= num_queries]

    if np.sum(np.abs(depth_map - tfm_gt_depth) > 0.00001) / min(new_h*new_w, image_h*image_w) > 0.1:
        print("after_transform_apply_K_inv_dot_xy_1, the error ratio > 0.1")

    tfm_labels = np.unique(tfm_gt_segmentation)
    tfm_labels = tfm_labels[tfm_labels < num_queries] #del non-plane label

    return depth_map, tfm_labels


class SingleDIODEPlaneDatasetMapper():
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
        mix_indoor,
        mix_anchor,
        classify_inverse_offset,
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

        # by default no augmentation is used here
        # fixing the h and w to be 192x256.
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[SingleDIODEPlaneDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train
        self.predict_center = predict_center
        self.num_queries = num_queries

        self.common_stride = common_stride

        self.param_domain_separate = param_domain_separate
        self.mix_indoor = mix_indoor
        self.mix_anchor = mix_anchor

        # if self.domain_separate:
        self.indoor_anchor_normals = np.load('./cluster_anchor/indoor_normal_anchors.npy')
        self.outdoor_anchor_normals = np.load('./cluster_anchor/outdoor_normal_anchors.npy')

        self.classify_inverse_offset = classify_inverse_offset

        if self.classify_inverse_offset:
            self.indoor_anchor_offsets = np.load('./cluster_anchor/indoor_inverse_offset_anchors.npy')
            self.outdoor_anchor_offsets = np.load('./cluster_anchor/outdoor_inverse_offset_anchors.npy')

        else:
            self.indoor_anchor_offsets = np.load('./cluster_anchor/indoor_offset_anchors.npy')
            self.outdoor_anchor_offsets = np.load('./cluster_anchor/outdoor_offset_anchors.npy')

        # else:
        if self.mix_anchor:
            self.anchor_normals = np.load('./cluster_anchor/mixed_normal_anchors.npy')
            self.anchor_offsets = np.load('./cluster_anchor/mixed_offset_anchors.npy')

        elif self.mix_indoor:
            self.anchor_normals = np.load('./cluster_anchor/indoor_normal_anchors.npy')
            self.anchor_offsets = np.load('./cluster_anchor/indoor_offset_anchors.npy')

        else:
            self.anchor_normals = np.load('./cluster_anchor/scannet_normal_anchors.npy')
            self.anchor_offsets = np.load('./cluster_anchor/scannet_offset_anchors.npy')

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
            "param_domain_separate": cfg.MODEL.MASK_FORMER.PARAM_DOMAIN_SEPARATE,
            'mix_indoor': cfg.MODEL.MASK_FORMER.MIX_INDOOR,
            'mix_anchor': cfg.MODEL.MASK_FORMER.MIX_ANCHOR,
            'classify_inverse_offset': cfg.MODEL.MASK_FORMER.CLASSIFY_INVERSE_OFFSET,
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
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image = data["image"]
        utils.check_image_size(dataset_dict, image)

        intrinsic = data['intrinsic']

        focal = intrinsic[0][0]
        focal_factor = self.canonical_focal / focal
        dataset_dict['focal_factor'] = torch.as_tensor(focal_factor)

        dataset_dict['dataset_class'] = torch.as_tensor(1)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # if issubclass(transforms.transforms[-1].__class__, PadTransform):
        #     transforms.transforms[-1].seg_pad_value = self.num_queries + 1 # seg_pad_value: 21, plane0 : 0
        # else:
        #     print("ERROR: issubclass(transforms.transforms[-1].__class__, PadTransform): ",
        #     issubclass(transforms.transforms[-1].__class__, PadTransform))

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # if self.domain_separate:
        dataset_dict['anchor_normals_indoor'] = torch.as_tensor(self.indoor_anchor_normals)
        dataset_dict['anchor_offsets_indoor'] = torch.as_tensor(self.indoor_anchor_offsets)

        dataset_dict['anchor_normals_outdoor'] = torch.as_tensor(self.outdoor_anchor_normals)
        dataset_dict['anchor_offsets_outdoor'] = torch.as_tensor(self.outdoor_anchor_offsets)

        dataset_dict['anchor_normals_mixed'] = torch.as_tensor(self.indoor_anchor_normals)
        dataset_dict['anchor_offsets_mixed'] = torch.as_tensor(self.indoor_anchor_offsets)

        # else:
        dataset_dict['anchor_normals'] = torch.as_tensor(self.anchor_normals)
        dataset_dict['anchor_offsets'] = torch.as_tensor(self.anchor_offsets)

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            intrinsic_inv = np.linalg.inv(intrinsic)
            dataset_K_inv_dot_xy_1, _ = dataset_precompute_K_inv_dot_xy_1(intrinsic_inv)
            dataset_dict["K_inv_dot_xy_1"] = torch.from_numpy(dataset_K_inv_dot_xy_1.astype(np.float32)).reshape(3, -1)

            # dataset_dict.pop("annotations", None)
            # return dataset_dict

        if "segmentation" in data.keys():
            origin_h, origin_w = image.shape[:2]
            # tfm_intrinsic not used here because the transformed K_inv_dot_xy1 is used directly.
            tfm_intrinsic, random_scale, new_h, new_w = transforms_apply_intrinsic(transforms, intrinsic, origin_h, origin_w)
            intrinsic_inv = np.linalg.inv(intrinsic)
            dataset_K_inv_dot_xy_1, _ = dataset_precompute_K_inv_dot_xy_1(intrinsic_inv)
            dataset_K_inv_dot_xy_1 = dataset_K_inv_dot_xy_1.transpose(1,2,0) # (image_h, image_w, 3)

            dataset_dict["random_scale"] = torch.from_numpy(np.array([random_scale], dtype=np.float32))

            # pan_seg_gt = utils.read_image(dataset_dict.pop("plane_seg_file_name"), "RGB")
            pan_seg_gt = data["segmentation"].astype(np.uint8) # 0~(num_planes-1) plane, 20 non-plane
            segments_info = dataset_dict["segments_info"]
            plane_depth_gt = data["depth"].astype(np.float32).squeeze() # [h, w]
            params = data["plane"].astype(np.float32)

            origin_labels = np.unique(pan_seg_gt)

            assert params.shape[0] == len(np.unique(pan_seg_gt)) - 1

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
            plane_depth_gt = np.expand_dims(new_transforms.apply_image(plane_depth_gt), axis = 0) # [1, h, w]  # interp="nearest"

            # apply the same transformation to dataset_K_inv_dot_xy_1.
            tfm_dataset_K_inv_dot_xy_1 = new_transforms.apply_image(dataset_K_inv_dot_xy_1) # interp = 'bilinear'
            tfm_dataset_K_inv_dot_xy_1 = tfm_dataset_K_inv_dot_xy_1.transpose(2, 0, 1)

            instances = Instances(image_shape)

            raw_depth_gt = data['raw_depth'].astype(np.float32).squeeze()
            dataset_dict['gt_global_pixel_depth'] = torch.as_tensor(raw_depth_gt)

            height, width = raw_depth_gt.shape
            dataset_dict['gt_resize14_global_pixel_depth'] = torch.as_tensor(cv2.resize(raw_depth_gt, (width // 4, height // 4), interpolation=cv2.INTER_NEAREST))

            tfm_plane_depth_gt, tfm_labels = after_transform_apply_K_inv_dot_xy_1(tfm_dataset_K_inv_dot_xy_1,
                                                                                  tfm_gt_segmentation = pan_seg_gt,
                                                                                  tfm_gt_depth = plane_depth_gt[0],
                                                                                  plane = params, num_planes = len(params),
                                                                                  num_queries = self.num_queries, new_h = new_h, new_w = new_w)
            tfm_plane_depth_gt = np.expand_dims(tfm_plane_depth_gt, axis=0)

            if self.is_train:
                dataset_dict["K_inv_dot_xy_1"] = torch.from_numpy(tfm_dataset_K_inv_dot_xy_1.astype(np.float32))

            # idx = dataset_dict["image_id"]
            # dataset_plane_depth = test_tfm_K(dataset_K_inv_dot_xy_1 = tfm_dataset_K_inv_dot_xy_1, #(3, 192, 256)
            #            idx = idx, tfm_image = image.copy(), image = data["image"].copy(), plane = params, tfm_K_inv = tfm_intrinsic_inv,
            #         #    idx = idx, image = image.copy(), plane = params, tfm_K_inv = None,
            #            K_inv = intrinsic_inv, tfm_depth = plane_depth_gt[0], gt_depth = data["depth"].astype(np.float32).squeeze().copy(),
            #            tfm_segmentation = pan_seg_gt, gt_segmentation = data["segmentation"].copy(),
            #            num_planes = len(params), new_h = new_h, new_w = new_w)

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
                    # params.append(plane_params[segment_info["id"]])

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

                dsize = (int(image_shape[1] / self.common_stride), int(image_shape[0] / self.common_stride)) # (width/self.common_stride))
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

                dsize = (int(image_shape[1] / self.common_stride), int(image_shape[0] / self.common_stride)) # (width/self.common_stride))
                gt_resize14_plane_depths = []

                for d, m in zip(tfm_plane_depths, masks):
                    d_ = cv2.resize(d[0].copy(), dsize, interpolation=cv2.INTER_AREA)
                    m_ = cv2.resize(m.copy().astype(np.float32), dsize, interpolation=cv2.INTER_NEAREST)
                    gt_resize14_plane_depths.append(torch.from_numpy(d_*m_))

                gt_resize14_plane_depths = torch.stack(gt_resize14_plane_depths, dim=0)
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
            dataset_dict['gt_resize14_global_pixel_normal'] = torch.as_tensor(cv2.resize(pixel_normal_map.transpose(1, 2, 0), (width // 4, height // 4), interpolation=cv2.INTER_NEAREST)).permute(2, 0, 1)

            dataset_dict['gt_global_pixel_offset'] = torch.as_tensor(pixel_offset_map)
            dataset_dict['gt_resize14_global_pixel_offset'] = torch.as_tensor(cv2.resize(pixel_offset_map, (width // 4, height // 4), interpolation=cv2.INTER_NEAREST))

        return dataset_dict

if __name__ == "__main__":
    pass
