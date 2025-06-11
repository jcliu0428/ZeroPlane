# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modified by https://github.com/facebookresearch/Mask2Former
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from fvcore.nn import FlopCountAnalysis

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher


@META_ARCH_REGISTRY.register()
class ZeroPlane(nn.Module):
    """
    Main class for plane segmentation and reconstruction architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        backbone_type: str,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        predict_param: bool,
        predict_depth: bool,
        predict_global_pixel_depth: bool,
        predict_global_pixel_normal: bool,
        with_ins_q_loss: bool,
        plane_mask_threshold: float,
        learn_normal_class: bool,
        learn_offset_class: bool,
        learn_coupled_anchor_class: bool,
        large_resolution_eval: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            k_inv_dot_xy1:
            predict_param: bool,
            predict_depth: bool,
            plane_mask_threshold: float,
        """
        super().__init__()
        self.backbone = backbone
        self.backbone_type = backbone_type

        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.metadata = metadata

        self.num_queries = num_queries

        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.predict_param = predict_param

        # piece-wise depth
        self.predict_depth = predict_depth

        # pixel-wise depth map
        self.predict_global_pixel_depth = predict_global_pixel_depth
        self.predict_global_pixel_normal = predict_global_pixel_normal

        assert not (predict_depth and predict_global_pixel_depth)

        self.with_ins_q_loss = with_ins_q_loss

        self.plane_mask_threshold = plane_mask_threshold

        self.learn_normal_class = learn_normal_class

        self.learn_offset_class = learn_offset_class

        self.learn_coupled_anchor_class = learn_coupled_anchor_class

        assert not (self.learn_normal_class and self.learn_coupled_anchor_class)

        if not self.predict_param:
            assert self.learn_offset_class or self.learn_coupled_anchor_class

        self.large_resolution_eval = large_resolution_eval

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_type = cfg.MODEL.BACKBONE.NAME
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
            # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        param_l1_weight = cfg.MODEL.MASK_FORMER.PARAM_L1_WEIGHT
        param_cos_weight = cfg.MODEL.MASK_FORMER.PARAM_COS_WEIGHT
        q_weight = cfg.MODEL.MASK_FORMER.Q_WEIGHT
        plane_depths_weight = cfg.MODEL.MASK_FORMER.PLANE_DEPTHS_WEIGHT

        pixel_normal_l1_weight = cfg.MODEL.MASK_FORMER.PIXEL_NORMAL_L1_WEIGHT
        pixel_normal_cos_weight = cfg.MODEL.MASK_FORMER.PIXEL_NORMAL_COS_WEIGHT

        global_pixel_depth_weight = cfg.MODEL.MASK_FORMER.GLOBAL_PIXEL_DEPTH_WEIGHT
        global_pixel_normal_l1_weight = cfg.MODEL.MASK_FORMER.GLOBAL_PIXEL_NORMAL_L1_WEIGHT
        global_pixel_normal_cos_weight = cfg.MODEL.MASK_FORMER.GLOBAL_PIXEL_NORMAL_COS_WEIGHT

        # predict bool
        predict_param = cfg.MODEL.MASK_FORMER.PREDICT_PARAM
        predict_depth = cfg.MODEL.MASK_FORMER.PREDICT_DEPTH

        wo_q_loss = cfg.MODEL.MASK_FORMER.WO_Q_LOSS

        predict_global_pixel_depth = cfg.MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_DEPTH
        predict_global_pixel_normal = cfg.MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_NORMAL

        with_ins_q_loss = cfg.MODEL.MASK_FORMER.WITH_INS_Q_LOSS

        ins_offset_weight = cfg.MODEL.MASK_FORMER.INS_OFFSET_WEIGHT
        ins_q_weight = cfg.MODEL.MASK_FORMER.INS_Q_WEIGHT

        learn_normal_class = cfg.MODEL.MASK_FORMER.LEARN_NORMAL_CLS
        normal_class_weight = cfg.MODEL.MASK_FORMER.NORMAL_CLASS_WEIGHT
        normal_residual_weight = cfg.MODEL.MASK_FORMER.NORMAL_RESIDUAL_WEIGHT

        learn_offset_class = cfg.MODEL.MASK_FORMER.LEARN_OFFSET_CLS
        offset_class_weight = cfg.MODEL.MASK_FORMER.OFFSET_CLASS_WEIGHT
        offset_residual_weight = cfg.MODEL.MASK_FORMER.OFFSET_RESIDUAL_WEIGHT

        learn_coupled_anchor_class = cfg.MODEL.MASK_FORMER.USE_COUPLED_ANCHOR
        coupled_anchor_class_weight = cfg.MODEL.MASK_FORMER.NORMAL_CLASS_WEIGHT
        coupled_anchor_residual_weight = cfg.MODEL.MASK_FORMER.NORMAL_RESIDUAL_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class = class_weight,
            cost_mask = mask_weight,
            cost_dice = dice_weight,
            cost_param = param_l1_weight,
            cost_depth = plane_depths_weight,
            cost_pixel_normal = pixel_normal_l1_weight,
            predict_param = predict_param,
            predict_depth = predict_depth,
            num_points = cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            normalize_param = cfg.MODEL.NORMALIZE_PARAM,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
                       "loss_instance_param_l1": param_l1_weight, "loss_instance_param_cos": param_cos_weight,
                       "loss_Q": q_weight, "loss_plane_depths": plane_depths_weight,
                       "loss_pixel_normal_l1": pixel_normal_l1_weight,
                       "loss_pixel_normal_cos": pixel_normal_cos_weight,
                       "loss_global_pixel_depth": global_pixel_depth_weight,
                       "loss_global_pixel_normal_l1": global_pixel_normal_l1_weight,
                       "loss_global_pixel_normal_cos": global_pixel_normal_cos_weight,
                       "loss_instance_offset": ins_offset_weight,
                       "loss_instance_Q": ins_q_weight,
                       'loss_normal_class': normal_class_weight,
                       'loss_normal_residual': normal_residual_weight,
                       'loss_offset_class': offset_class_weight,
                       'loss_offset_residual': offset_residual_weight,
                       'loss_coupled_anchor_class': coupled_anchor_class_weight,
                       'loss_coupled_anchor_residual': coupled_anchor_residual_weight,
                       }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = [
                    "labels",
                    "masks",
                ]

        if predict_param:
            losses.extend(['params'])

        if not wo_q_loss:
            losses.extend(['Q'])

        if predict_depth:
            losses.extend([
                'plane_depths'
            ])

        if predict_global_pixel_depth:
            losses.extend([
                'global_pixel_depth'
            ])

        if predict_global_pixel_normal:
            losses.extend([
                'global_pixel_normal'
            ])

        if learn_normal_class:
            losses.extend([
                'normal_class', 'normal_residual'
            ])

        if learn_offset_class:
            losses.extend([
                'offset_class', 'offset_residual'
            ])

        if learn_coupled_anchor_class:
            losses.extend([
                'coupled_anchor_class', 'coupled_anchor_residual'
            ])

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            normalize_param=cfg.MODEL.NORMALIZE_PARAM,
            pixel_depth_loss_type=cfg.MODEL.PIXEL_DEPTH_LOSS_TYPE,
            upsample_pixel_pred=cfg.MODEL.MASK_FORMER.DEPTH_NORMAL_PRED_UPSAMPLE,
        )

        return {
            "backbone": backbone,
            "backbone_type": backbone_type,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "predict_param": predict_param,
            "predict_depth": predict_depth,
            "predict_global_pixel_depth": predict_global_pixel_depth,
            "predict_global_pixel_normal": predict_global_pixel_normal,
            'with_ins_q_loss': with_ins_q_loss,
            "plane_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.PLANE_MASK_THRESHOLD,
            'learn_normal_class': learn_normal_class,
            'learn_offset_class': learn_offset_class,
            'learn_coupled_anchor_class': learn_coupled_anchor_class,
            'large_resolution_eval': cfg.INPUT.LARGE_RESOLUTION_EVAL,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        features = self.backbone(images.tensor)

        # flops_backbone = FlopCountAnalysis(self.backbone, images.tensor)
        # print(f"Total FLOPs of backbone: {flops_backbone.total()} and the image size:", images.tensor.size())

        anchors = {}

        if self.learn_coupled_anchor_class:
            anchors['anchor_normal_divide_offset'] = torch.stack([x['anchor_normal_divide_offset'] for x in batched_inputs]).to(self.device)

        else:
            anchors['anchor_normals'] = torch.stack([x['anchor_normals'] for x in batched_inputs]).to(self.device)
            anchors['anchor_offsets'] = torch.stack([x['anchor_offsets'] for x in batched_inputs]).squeeze(-1).to(self.device)

        gt_global_pixel_depths = gt_global_pixel_normals = None

        # flops_head = FlopCountAnalysis(self.sem_seg_head, (features, None, anchors))
        # print(f"Total FLOPs of head: {flops_head.total()}")

        # print(f"Total number of FLOPs of the model: {flops_backbone.total() + flops_head.total()}")
        # exit(1)

        outputs = self.sem_seg_head(features, anchors=anchors)

        if self.training:
            K_inv_dot_xy_1s = [x["K_inv_dot_xy_1"].to(self.device) for x in batched_inputs]
            random_scales = [x["random_scale"].to(self.device) for x in batched_inputs]

            gt_resize14_global_pixel_depths = [x['gt_resize14_global_pixel_depth'].to(self.device) for x in batched_inputs]
            gt_resize14_global_pixel_normals = [x['gt_resize14_global_pixel_normal'].to(self.device) for x in batched_inputs]

            gt_global_pixel_depths = [x['gt_global_pixel_depth'].to(self.device) for x in batched_inputs]
            gt_global_pixel_normals = [x['gt_global_pixel_normal'].to(self.device) for x in batched_inputs]

            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images, K_inv_dot_xy_1s, random_scales, gt_global_pixel_depths, gt_global_pixel_normals, gt_resize14_global_pixel_depths, gt_resize14_global_pixel_normals)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, anchors=anchors)

            for k_idx, k in enumerate(list(losses.keys())):
                if k in self.criterion.weight_dict: # {'loss_ce': 2.0, 'loss_mask': 5.0, 'loss_dice': 5.0, 'loss_ce_0': 2.0, 'loss_mask_0': 5.0, 'loss_dice_0': 5.0, 'loss_ce_1': 2.0, 'loss_mask_1': 5.0, 'loss_dice_1': 5.0, 'loss_ce_2': 2.0, 'loss_mask_2': 5.0, 'loss_dice_2': 5.0, 'loss_ce_3': 2.0, 'loss_mask_3': 5.0, ...}
                    losses[k] *= self.criterion.weight_dict[k]

                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses

        else:
            if self.large_resolution_eval:
                target_h, target_w = 480, 640

            else:
                target_h, target_w = 192, 256

            mask_cls_results = outputs["pred_logits"] # torch.Size([b, num_queries, num_classes + 1])
            mask_pred_results = outputs["pred_masks"] # torch.Size([b, num_queries, h/4, w/4])

            # upsample masks
            if not mask_pred_results.shape[-1] == images.tensor.shape[-1]:
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size = (target_h, target_w),
                    mode = "bilinear",
                    align_corners = False,
                )  # torch.Size([b,num_queries,h,w])

            if self.predict_depth:
                depth_pred_results = outputs["pred_depths"] # torch.Size([b, num_queries, h, w]

                if not depth_pred_results.shape[-1] == images.tensor.shape[-1]:
                    depth_pred_results = F.interpolate(
                        depth_pred_results,
                        size = (target_h, target_w),
                        mode = "bilinear",
                        align_corners=False,
                    )  # torch.Size([b,num_queries,h,w])

            elif self.predict_global_pixel_depth:
                depth_pred_results = outputs['pred_depths']

                if not depth_pred_results.shape[-1] == images.tensor.shape[-1]:
                    depth_pred_results = F.interpolate(
                        depth_pred_results,
                        size = (target_h, target_w),
                        mode = "bilinear",
                        align_corners=False,
                    )  # torch.Size([b,num_queries,h,w])

            else:
                depth_pred_results = [None for _ in range(mask_cls_results.size(0))]

            if self.predict_global_pixel_normal:
                pixel_normal_pred_results = outputs['pred_pixel_normals']

                if not pixel_normal_pred_results.shape[-1] == images.tensor.shape[-1]:
                    pixel_normal_pred_results = F.interpolate(
                        pixel_normal_pred_results,
                        size = (target_h, target_w),
                        mode = "bilinear",
                        align_corners=False,
                    )  # torch.Size([b,num_queries,h,w])

            else:
                pixel_normal_pred_results = [None for _ in range(mask_cls_results.size(0))]

            pixel_offset_pred_results = [None for _ in range(mask_cls_results.size(0))]

            if self.predict_param:
                param_pred_results = outputs["pred_params"]  # torch.Size([b, num_queries, 3])

            elif self.learn_normal_class:
                pred_normal_class_logits = outputs["pred_ins_normal_class"]
                pred_normal_class = pred_normal_class_logits.argmax(-1)

                pred_ins_normals_all = outputs['pred_ins_normals']
                bs, num_queries = pred_ins_normals_all.size()[:2]
                pred_ins_normals = pred_ins_normals_all[torch.arange(bs), torch.arange(num_queries), pred_normal_class]

                pred_ins_offsets_all = outputs['pred_ins_offsets']

                if self.learn_offset_class:
                    pred_offset_class_logits = outputs["pred_ins_offset_class"]
                    pred_offset_class = pred_offset_class_logits.argmax(-1)
                    pred_ins_offsets = pred_ins_offsets_all[torch.arange(bs), torch.arange(num_queries), pred_offset_class]

                else:
                    pred_ins_offsets = pred_ins_offsets_all.squeeze(-1)

                param_pred_results = pred_ins_normals / (pred_ins_offsets.unsqueeze(-1) + 1e-4)

            elif self.learn_coupled_anchor_class:
                pred_coupled_anchor_class_logits = outputs["pred_coupled_anchor_class"]
                pred_coupled_anchor_class = pred_coupled_anchor_class_logits.argmax(-1)

                pred_coupled_anchor_all = outputs['pred_coupled_anchor']
                bs, num_queries = pred_coupled_anchor_all.size()[:2]
                param_pred_results = pred_coupled_anchor_all[torch.arange(bs), torch.arange(num_queries), pred_coupled_anchor_class]

            else:
                param_pred_results = [None for _ in range(mask_cls_results.size(0))]

            del outputs

            processed_results = []

            K_inv_dot_xy_1s = [x["K_inv_dot_xy_1"].to(self.device) for x in batched_inputs]

            debug_mode = False

            if debug_mode and "instances" in batched_inputs[0]:
                gt_resize14_global_pixel_depths = [x['gt_resize14_global_pixel_depth'].to(self.device) for x in batched_inputs]
                gt_resize14_global_pixel_normals = [x['gt_resize14_global_pixel_normal'].to(self.device) for x in batched_inputs]

                gt_global_pixel_depths = [x['gt_global_pixel_depth'].to(self.device) for x in batched_inputs]
                gt_global_pixel_normals = [x['gt_global_pixel_normal'].to(self.device) for x in batched_inputs]

                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                random_scales = [x["random_scale"].to(self.device) for x in batched_inputs]

                targets = self.prepare_targets(gt_instances, images, K_inv_dot_xy_1s, random_scales, gt_global_pixel_depths, gt_global_pixel_normals, gt_resize14_global_pixel_depths, gt_resize14_global_pixel_normals)

            else:
                targets = None

            for mask_cls_result, mask_pred_result, param_pred_result, depth_pred_result, pixel_normal_pred_result, pixel_offset_pred_result, input_per_image, image_size, k_inv_dot_xy_1 in zip(
                mask_cls_results, mask_pred_results, param_pred_results, depth_pred_results, pixel_normal_pred_results, pixel_offset_pred_results, batched_inputs, images.image_sizes, K_inv_dot_xy_1s
            ):
                # height = input_per_image.get("height", image_size[0]) # ep 349
                # width = input_per_image.get("width", image_size[1]) # ep 640

                height, width = target_h, target_w

                processed_results.append({})

                mask_cls_result = mask_cls_result.to(mask_pred_result) # torch.Size([num_queries, num_classes, num_classes + 1])

                # plane inference
                if self.semantic_on:
                    plane_seg, inferred_planes_depth, inferred_seg_depth, inferred_planes_depth_from_seg_depth, valid_param = retry_if_cuda_oom(self.plane_inference)(mask_cls_result, mask_pred_result, param_pred_result, depth_pred_result, pixel_normal_pred_result, k_inv_dot_xy_1)

                    processed_results[-1]["sem_seg"] = plane_seg
                    processed_results[-1]["planes_depth"] = inferred_planes_depth
                    processed_results[-1]["seg_depth"] = inferred_seg_depth
                    processed_results[-1]["planes_depth_from_pixel_depth"] = inferred_planes_depth_from_seg_depth
                    processed_results[-1]["valid_params"] = valid_param
                    processed_results[-1]['K_inv_dot_xy_1'] = k_inv_dot_xy_1

                    processed_results[-1]['global_pixel_normal'] = pixel_normal_pred_result

            return processed_results

    def prepare_targets(self, targets, images, K_inv_dot_xy_1s, random_scales, gt_global_pixel_depths, gt_global_pixel_normals, \
                        gt_resize14_global_pixel_depths, gt_resize14_global_pixel_normals):

        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image, K_inv_dot_xy_1, scale, gt_global_pixel_depth, gt_global_pixel_normal, gt_resize14_global_pixel_depth, gt_resize14_global_pixel_normal in zip(targets, K_inv_dot_xy_1s, random_scales, gt_global_pixel_depths, gt_global_pixel_normals, gt_resize14_global_pixel_depths, gt_resize14_global_pixel_normals):
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "params": targets_per_image.gt_params,
                    "plane_depths": targets_per_image.gt_plane_depths,
                    "resize14_plane_depths": targets_per_image.gt_resize14_plane_depths,
                    "K_inv_dot_xy_1": K_inv_dot_xy_1,
                    "random_scale": scale,
                    "resize14_pixel_normals": targets_per_image.gt_resize14_pixel_normals,
                    'global_pixel_depth': gt_global_pixel_depth,
                    'global_pixel_normal': gt_global_pixel_normal,
                    'resize14_global_pixel_depth': gt_resize14_global_pixel_depth,
                    'resize14_global_pixel_normal': gt_resize14_global_pixel_normal,
                }
            )
        return new_targets

    def plane_inference(self, mask_cls, mask_pred, param_pred, depth_pred, pixel_normal_pred, k_inv_dot_xy1):
        mask_cls = F.softmax(mask_cls, dim=-1)  # torch.Size([num_queries, num_classes + 1 = 3])
        score, labels = mask_cls.max(dim=-1)

        labels[labels != 1] = 0  # [num_queries]
        label_mask = labels > 0  # [num_queries]
        if sum(label_mask) == 0:
            _, max_pro_idx = mask_cls[:, 1].max(dim=0)
            label_mask[max_pro_idx] = 1

        mask_pred = mask_pred.sigmoid()  # torch.Size([num_queries, h, w])
        valid_mask_pred = mask_pred[label_mask]  # [valid_plane_num,h,w]

        tmp = torch.zeros((self.num_queries + 1 - valid_mask_pred.shape[0], valid_mask_pred.shape[1], valid_mask_pred.shape[2]),
                          dtype = valid_mask_pred.dtype, device = valid_mask_pred.device)

        non_plane_mask = (valid_mask_pred > self.plane_mask_threshold).sum(0) == 0

        tmp[-1][non_plane_mask] = 1
        plane_seg = torch.cat((valid_mask_pred, tmp), dim = 0)  # [num_queries, h, w]
        plane_seg = plane_seg.sigmoid()  # [num_queries, h, w]

        valid_num = valid_mask_pred.shape[0]
        inferred_planes_depth = None

        if self.predict_depth:
            valid_depth_pred = depth_pred[label_mask]  # [valid_plane_num, h, w]
            segmentation = (plane_seg[:valid_num].argmax(dim=0)[:,:,None] == torch.arange(valid_num).to(plane_seg)).permute(2, 0, 1) # [h, w, 1] == []  -> [valid_plane_num, h, w]
            inferred_seg_depth = (segmentation * valid_depth_pred).sum(0) # [h, w]
            inferred_seg_depth[non_plane_mask] = 0.0  # del non-plane regions

        elif self.predict_global_pixel_depth:
            inferred_seg_depth = depth_pred.clone().squeeze(0)

        else:
            inferred_seg_depth = torch.zeros_like(mask_pred[0])

        if self.predict_param or self.learn_normal_class or self.learn_coupled_anchor_class:
            valid_param = param_pred[label_mask, :].float()  # valid_plane_num, 3

            # get depth map
            h, w = plane_seg.shape[-2:]
            plane_seg_map = plane_seg[:valid_num].argmax(dim=0)

            depth_maps_inv = torch.matmul(valid_param, k_inv_dot_xy1.to(self.pixel_mean.device))
            depth_maps_inv = torch.clamp(depth_maps_inv, min=1e-2, max=1e4)
            depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)

            inferred_planes_depth = depth_maps.t()[range(h*w), plane_seg_map.view(-1)] # plane depth [h,w]
            inferred_planes_depth = inferred_planes_depth.view(h, w)
            inferred_planes_depth[non_plane_mask] = 0.0 # del non-plane regions

            normal_offset_from_depth = []

            for plane_idx, param in enumerate(valid_param):
                mask = plane_seg_map == plane_idx
                normal = param / (torch.linalg.norm(param) + 1e-8)

                offset_map = normal @ k_inv_dot_xy1.to(normal.device) * inferred_seg_depth.view(-1)
                offset = torch.mean(offset_map[mask.view(-1)])

                normal_offset_from_depth.append(normal / (offset + 1e-8))

            normal_offset_from_depth = torch.stack(normal_offset_from_depth)

            depth_maps_inv = torch.matmul(normal_offset_from_depth, k_inv_dot_xy1.to(self.pixel_mean.device))
            depth_maps_inv = torch.clamp(depth_maps_inv, min=1e-2, max=1e4)
            depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)

            normal_offset_inferred_planes_depth = depth_maps.t()[range(h*w), plane_seg_map.view(-1)] # plane depth [h,w]
            normal_offset_inferred_planes_depth = normal_offset_inferred_planes_depth.view(h, w)
            normal_offset_inferred_planes_depth[non_plane_mask] = 0.0 # del non-plane regions

        elif self.predict_global_pixel_normal:
            # get depth map
            h, w = plane_seg.shape[-2:]

            plane_seg_map = plane_seg[:valid_num].argmax(dim=0)

            normal_offset_from_depth = []
            valid_param = []

            for plane_idx in range(valid_num):
                mask = plane_seg_map == plane_idx

                if self.predict_global_pixel_normal:
                    normal = torch.mean(pixel_normal_pred[:, mask], dim=-1)

                normal = normal / (torch.linalg.norm(normal) + 1e-8)

                offset_map = normal @ k_inv_dot_xy1.to(normal.device) * inferred_seg_depth.view(-1)
                offset = torch.mean(offset_map[mask.view(-1)])

                normal_offset_from_depth.append(normal / (offset + 1e-8))
                valid_param.append(normal / (offset + 1e-8))

            normal_offset_from_depth = torch.stack(normal_offset_from_depth)
            valid_param = torch.stack(valid_param)

            depth_maps_inv = torch.matmul(valid_param, k_inv_dot_xy1.to(self.pixel_mean.device))
            depth_maps_inv = torch.clamp(depth_maps_inv, min=1e-2, max=1e4)
            depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)

            inferred_planes_depth = depth_maps.t()[range(h*w), plane_seg_map.view(-1)] # plane depth [h,w]
            inferred_planes_depth = inferred_planes_depth.view(h, w)
            inferred_planes_depth[non_plane_mask] = 0.0 # del non-plane regions

            depth_maps_inv = torch.matmul(normal_offset_from_depth, k_inv_dot_xy1.to(self.pixel_mean.device))
            depth_maps_inv = torch.clamp(depth_maps_inv, min=1e-2, max=1e4)
            depth_maps = 1. / depth_maps_inv  # (valid_plane_num, h*w)

            normal_offset_inferred_planes_depth = depth_maps.t()[range(h*w), plane_seg_map.view(-1)] # plane depth [h,w]
            normal_offset_inferred_planes_depth = normal_offset_inferred_planes_depth.view(h, w)
            normal_offset_inferred_planes_depth[non_plane_mask] = 0.0 # del non-plane regions

        return plane_seg, inferred_planes_depth, inferred_seg_depth, normal_offset_inferred_planes_depth, valid_param
