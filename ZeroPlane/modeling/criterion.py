# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
# and by https://github.com/facebookresearch/Mask2Former
import logging
import copy

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


# from detectron2.utils.memory import retry_if_cuda_oom
# from detectron2.modeling.postprocessing import sem_seg_postprocess


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def l1_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):

    return torch.mean(torch.sum(torch.abs(targets - inputs), dim=1))


l1_loss_jit = torch.jit.script(
    l1_loss
)


def cos_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    ):
    similarity = torch.nn.functional.cosine_similarity(inputs, targets, dim=1)  # N
    return torch.mean(1-similarity)

cos_loss_jit = torch.jit.script(
    cos_loss
)


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


# Main loss function used in AdaBins paper
def calculate_silog_loss(input, target, mask=None):
    if mask is None:
        mask = (input > 1e-4) * (target > 1e-4)

    input = input[mask]
    target = target[mask]

    g = torch.log(input) - torch.log(target)

    Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
    return 1.0 * torch.sqrt(Dg)


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,
                 normalize_param, pixel_depth_loss_type, upsample_pixel_pred):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.normalize_param = normalize_param
        self.pixel_depth_loss_type = pixel_depth_loss_type

        self.upsample_pixel_pred = upsample_pixel_pred

    def loss_labels(self, outputs, targets, indices, num_masks, anchors=None, focal_factors=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()  # torch.Size([b, num_queries, 3])

        idx = self._get_src_permutation_idx(indices)  # (batch_idx, indices[0] for src)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  # target["labels"][indice[1](for tgt)]
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks, anchors=None, focal_factors=None):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices) # (batch_idx[b*num_tgt_planes], src_idx)
        tgt_idx = self._get_tgt_permutation_idx(indices) # (batch_idx, tgt_idx)
        src_masks = outputs["pred_masks"] # [b, num_queries, h/4, w/4]
        src_masks = src_masks[src_idx] # [b*num_tgt_planes_i, h/4, w/4]
        masks = [t["masks"] for t in targets] # [num_tgt_planes_i, h, w] * b
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose() # [b, max([num_tgt_planes_i]), h, w]
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords# (h, w, b*3) -> (b, h, w, 3)
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )  # torch.Size([b*num_tgt_planes, num_points, 2])
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)  # torch.Size([b*num_tgt_planes, num_points])

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)  #

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_instance_params(self, outputs, targets, indices, num_planes, log=True, anchors=None, focal_factors=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
            The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_params' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_param = outputs['pred_params'][idx]  # N, 3
        target_param = torch.cat([t["params"][J] for t, (_, J) in zip(targets, indices)])

        if self.normalize_param:
            src_param = src_param / (torch.linalg.norm(src_param, dim=-1) + 1e-8).unsqueeze(-1)
            target_param = target_param / (torch.linalg.norm(target_param, dim=-1) + 1e-8).unsqueeze(-1)

        valid = torch.norm(target_param, dim=-1) > 1e-4
        src_param = src_param[valid]
        target_param = target_param[valid]

        # l1 loss
        loss_param_l1 = torch.mean(torch.sum(torch.abs(target_param - src_param), dim=1))

        # cos loss
        similarity = torch.nn.functional.cosine_similarity(src_param, target_param, dim=1)  # N
        loss_param_cos = torch.mean(1 - similarity)
        angle = torch.mean(torch.acos(torch.clamp(similarity, -1, 1)))

        losses = {}
        losses['loss_instance_param_l1'] = loss_param_l1
        losses['loss_instance_param_cos'] = loss_param_cos
        if log:
            losses['mean_angle'] = angle * 180.0 / np.pi

        return losses

    def loss_instance_normals(self, outputs, targets, indices, num_planes, anchors=None, focal_factors=None):
        idx = self._get_src_permutation_idx(indices)

        src_normal = outputs['pred_ins_normals'][idx]  # N, 3
        target_param = torch.cat([t["params"][J] for t, (_, J) in zip(targets, indices)])

        target_normal = target_param / (torch.linalg.norm(target_param, dim=1).unsqueeze(1) + 1e-4)

        valid = torch.norm(target_normal, dim=-1) > 1e-4
        src_normal = src_normal[valid]
        target_normal = target_normal[valid]

        # l1 loss
        loss_normal_l1 = torch.mean(torch.sum(torch.abs(target_normal - src_normal), dim=1))

        # cos loss
        similarity = torch.nn.functional.cosine_similarity(src_normal, target_normal, dim=1)  # N
        loss_normal_cos = torch.mean(1 - similarity)

        losses = {}
        losses['loss_instance_normal_l1'] = loss_normal_l1
        losses['loss_instance_normal_cos'] = loss_normal_cos

        return losses

    def loss_instance_offsets(self, outputs, targets, indices, num_planes, anchors=None, focal_factors=None):
        idx = self._get_src_permutation_idx(indices)

        src_offset = outputs['pred_ins_offsets'][idx]  # N, 3
        target_param = torch.cat([t["params"][J] for t, (_, J) in zip(targets, indices)])

        valid = torch.norm(target_param, dim=-1) > 1e-4
        src_offset = src_offset[valid]
        target_param = target_param[valid]

        target_offset = 1. / (torch.linalg.norm(target_param, dim=1).unsqueeze(1) + 1e-4)

        # l1 loss
        loss_offset_l1 = torch.mean(torch.abs(target_offset - src_offset))

        losses = {}
        losses['loss_instance_offset'] = loss_offset_l1

        return losses

    def loss_instance_Q(self, outputs, targets, indices, num_planes_sum, log=True, anchors=None, focal_factors=None):

        gt_depths = torch.stack([t["plane_depths"].sum(dim = 0) for t in targets]) # b , h, w

        b, h, w = gt_depths.shape
        assert b == len(targets)

        losses = 0.

        for bi in range(b):

            segmentation = targets[bi]['masks']  # num_tgt_planes, h, w
            num_planes = segmentation.shape[0]
            device = segmentation.device

            depth = gt_depths[bi]  # 1, h, w
            k_inv_dot_xy1_map = targets[bi]["K_inv_dot_xy_1"].view(3, h, w) # 3, h, w
            gt_pts_map = k_inv_dot_xy1_map * depth  # 3, h, w

            indices_bi = indices[bi]
            idx_out = indices_bi[0]
            idx_tgt = indices_bi[1]

            # num_planes = idx_tgt.max() + 1
            assert idx_tgt.max() + 1 == num_planes

            # select pixel with segmentation
            loss_bi = 0.
            valid_num_planes = 0.

            for i in range(num_planes):
                gt_plane_idx = int(idx_tgt[i])
                mask = segmentation[gt_plane_idx, :, :].view(1, h, w)
                mask = mask > 0

                pts = torch.masked_select(gt_pts_map, mask).view(3, -1)  # 3, plane_pt_num

                pred_plane_idx = int(idx_out[i])

                ins_normal = outputs['pred_ins_normals'][bi][pred_plane_idx].view(1, 3)
                ins_offset = outputs['pred_ins_offsets'][bi][pred_plane_idx]

                param = ins_normal / (ins_offset + 1e-4)

                loss = torch.abs(torch.matmul(param, pts) - 1)  # 1, plane_pt_num
                loss = loss.mean()

                if not torch.isinf(loss) and not torch.isnan(loss):
                    loss_bi += loss
                    valid_num_planes += 1.0

            loss_bi = loss_bi / (float(valid_num_planes) + 1e-8)
            losses += loss_bi

        losses_dict = {}
        losses_dict['loss_instance_Q'] = losses / float(b)

        return losses_dict

    def loss_refine_instance_params(self, outputs, targets, indices, num_planes, anchors=None, focal_factors=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
            The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)

        src_normals = outputs['pred_ins_normals'][idx]  # N, 3
        src_offsets = outputs['pred_ins_offsets'][idx]

        src_param = src_normals / (src_offsets + 1e-4)
        target_param = torch.cat([t["params"][J] for t, (_, J) in zip(targets, indices)])

        valid = torch.norm(target_param, dim=-1) > 1e-4
        src_param = src_param[valid]
        target_param = target_param[valid]

        # l1 loss
        loss_param_l1 = torch.mean(torch.sum(torch.abs(target_param - src_param), dim=1))

        # cos loss
        similarity = torch.nn.functional.cosine_similarity(src_param, target_param, dim=1)  # N
        loss_param_cos = torch.mean(1 - similarity)

        losses = {}
        losses['loss_refine_instance_param_l1'] = loss_param_l1
        losses['loss_refine_instance_param_cos'] = loss_param_cos

        return losses

    def loss_Q(self, outputs, targets, indices, num_planes_sum, log=True, anchors=None, focal_factors=None):

        gt_depths = torch.stack([t["plane_depths"].sum(dim = 0) for t in targets]) # b , h, w

        b, h, w = gt_depths.shape
        assert b == len(targets)

        losses = 0.

        for bi in range(b):

            segmentation = targets[bi]['masks']  # num_tgt_planes, h, w
            num_planes = segmentation.shape[0]
            device = segmentation.device

            depth = gt_depths[bi]  # 1, h, w
            k_inv_dot_xy1_map = targets[bi]["K_inv_dot_xy_1"].view(3, h, w) # 3, h, w
            gt_pts_map = k_inv_dot_xy1_map * depth  # 3, h, w

            indices_bi = indices[bi]
            idx_out = indices_bi[0]
            idx_tgt = indices_bi[1]

            # num_planes = idx_tgt.max() + 1
            assert idx_tgt.max() + 1 == num_planes

            # select pixel with segmentation
            loss_bi = 0.
            valid_num_planes = 0.

            for i in range(num_planes):
                gt_plane_idx = int(idx_tgt[i])
                mask = segmentation[gt_plane_idx, :, :].view(1, h, w)
                mask = mask > 0

                pts = torch.masked_select(gt_pts_map, mask).view(3, -1)  # 3, plane_pt_num

                pred_plane_idx = int(idx_out[i])
                param = outputs['pred_params'][bi][pred_plane_idx].view(1, 3)

                if self.normalize_param:
                    gt_params = targets[bi]['params']
                    gt_param = gt_params[gt_plane_idx].view(-1, 3)
                    gt_offset = 1. / (torch.linalg.norm(gt_param, dim=-1) + 1e-8)

                    param = param / (torch.linalg.norm(param, dim=-1) + 1e-8).unsqueeze(-1)
                    param = param / (gt_offset + 1e-8)

                    loss = torch.abs(torch.matmul(param, pts) - 1)

                else:
                    loss = torch.abs(torch.matmul(param, pts.type_as(param)) - 1)  # 1, plane_pt_num

                loss = loss.mean()

                if not torch.isinf(loss) and not torch.isnan(loss):
                    loss_bi += loss
                    valid_num_planes += 1.0

            loss_bi = loss_bi / (float(valid_num_planes) + 1e-8)
            losses += loss_bi

        losses_dict = {}
        losses_dict['loss_Q'] = losses / float(b)

        return losses_dict

    def loss_global_pixel_depth(self, outputs, targets, indices=None, num_planes_sum=None, anchors=None, focal_factors=None):
        depth_pred = outputs['pred_depths'].squeeze(1)

        if self.upsample_pixel_pred:
            depth_target = torch.stack([t['global_pixel_depth'] for t in targets])

        else:
            depth_target = torch.stack([t['resize14_global_pixel_depth'] for t in targets])

        if not depth_target.size() == depth_pred.size():
            pred_h, pred_w = depth_pred.size()[-2:]
            depth_target = F.interpolate(depth_target.unsqueeze(1), (pred_h, pred_w)).squeeze(1)

        mask = depth_target > 1e-4

        if self.pixel_depth_loss_type == 'l1':
            loss = {
                    "loss_global_pixel_depth":  torch.sum(torch.abs((depth_pred - depth_target) * mask)) / (mask.sum() + 1e-4)}

        elif self.pixel_depth_loss_type == 'silog':
            silog_loss = calculate_silog_loss(depth_pred, depth_target)

            loss = {
                    "loss_global_pixel_depth": silog_loss}

        else:
            raise NotImplementedError

        return loss

    def loss_global_pixel_normal(self, outputs, targets, indices=None, num_planes_sum=None, anchors=None, focal_factors=None):
        normal_pred = outputs['pred_pixel_normals']

        if self.upsample_pixel_pred:
            normal_target = torch.stack([t['global_pixel_normal'] for t in targets])

        else:
            normal_target = torch.stack([t['resize14_global_pixel_normal'] for t in targets])

        valid_mask = torch.linalg.norm(normal_target, dim=1) > 1e-4

        normal_pred = normal_pred.permute(0, 2, 3, 1)[valid_mask]
        normal_target = normal_target.permute(0, 2, 3, 1)[valid_mask]

        # l1 loss
        loss_pixel_normal_l1 = torch.mean(torch.sum(torch.abs(normal_pred - normal_target), dim=1))

        # cos loss
        similarity = torch.nn.functional.cosine_similarity(normal_pred, normal_target, dim=1)  # N
        loss_pixel_normal_cos = torch.mean(1 - similarity)

        losses = {}
        losses['loss_global_pixel_normal_l1'] = loss_pixel_normal_l1
        losses['loss_global_pixel_normal_cos'] = loss_pixel_normal_cos

        return losses

    def loss_plane_depths(self, outputs, targets, indices, num_planes_sum, log=True, anchors=None, focal_factors=None):

        src_idx = self._get_src_permutation_idx(indices) # [batch_idx, src_idx]

        src_depths = outputs["pred_depths"] # [b, num_queries, h, w]
        src_depths = src_depths[src_idx] # [b * num_tgt_planes_i, h, w]
        target_plane_depths = torch.cat([t["plane_depths"][J] for t, (_, J) in zip(targets, indices)]) # [b * num_tgt_planes_i, height, width]
        if src_depths.shape[-1] != target_plane_depths.shape[-1]:
            target_plane_depths = torch.cat([t["resize14_plane_depths"][J] for t,(_, J) in zip(targets, indices)]) #

        mask = target_plane_depths > 1e-4
        src_plane_depths = mask * src_depths # [b * num_tgt_planes_i, h, w]

        if self.pixel_depth_loss_type == 'l1':
            loss = {
                    "loss_plane_depths":  torch.sum(torch.abs((src_plane_depths - target_plane_depths) * mask)) / (mask.sum() + 1e-4)}

        elif self.pixel_depth_loss_type == 'silog':
            silog_loss = calculate_silog_loss(src_plane_depths, target_plane_depths)

            loss = {
                    "loss_plane_depths": silog_loss}

        else:
            raise NotImplementedError

        return loss

    def loss_normal_class(self, outputs, targets, indices, num_planes_sum, anchors=None, focal_factors=None):
        idx = self._get_src_permutation_idx(indices)
        pred_normal_class_logits = outputs['pred_ins_normal_class'][idx]

        target_param = torch.cat([t["params"][J] for t, (_, J) in zip(targets, indices)])

        valid = torch.norm(target_param, dim=-1) > 1e-4
        pred_normal_class_logits = pred_normal_class_logits[valid]
        target_param = target_param[valid]

        target_normals = target_param / (torch.linalg.norm(target_param, dim=-1).unsqueeze(-1) + 1e-4)

        batch_idx, _ = idx
        anchor_normals = torch.stack([anchors['anchor_normals'][i] for i in batch_idx], dim=0)

        # plane_num, normal_anchor_num
        normal_dists = torch.linalg.norm(target_normals.unsqueeze(1) - anchor_normals, dim=-1)
        gt_normal_class = normal_dists.argmin(-1)

        loss_normal_class = F.cross_entropy(pred_normal_class_logits, gt_normal_class)

        losses = {}
        losses['loss_normal_class'] = loss_normal_class

        return losses

    def loss_offset_class(self, outputs, targets, indices, num_planes_sum, anchors=None, focal_factors=None):
        idx = self._get_src_permutation_idx(indices)

        target_param = torch.cat([t["params"][J] for t, (_, J) in zip(targets, indices)])

        valid = torch.norm(target_param, dim=-1) > 1e-4
        target_param = target_param[valid]
        target_offsets = 1. / (torch.linalg.norm(target_param, dim=-1) + 1e-4)

        batch_idx, _ = idx

        anchor_offsets = torch.stack([anchors['anchor_offsets'][i] for i in batch_idx], dim=0)
        # plane_num, offset_anchor_num
        offset_dists = torch.abs(target_offsets.unsqueeze(1) - anchor_offsets)
        gt_offset_class = offset_dists.argmin(-1)

        pred_offset_class_logits = outputs['pred_ins_offset_class'][idx]
        pred_offset_class_logits = pred_offset_class_logits[valid]

        loss_offset_class = F.cross_entropy(pred_offset_class_logits, gt_offset_class)

        losses = {}
        losses['loss_offset_class'] = loss_offset_class

        return losses

    def loss_coupled_anchor_class(self, outputs, targets, indices, num_planes_sum, anchors=None, focal_factors=None):
        idx = self._get_src_permutation_idx(indices)
        pred_coupled_anchor_class_logits = outputs['pred_coupled_anchor_class'][idx]

        target_param = torch.cat([t["params"][J] for t, (_, J) in zip(targets, indices)])

        valid = torch.norm(target_param, dim=-1) > 1e-4
        pred_coupled_anchor_class_logits = pred_coupled_anchor_class_logits[valid]
        target_param = target_param[valid]

        batch_idx, _ = idx
        anchors = torch.stack([anchors['anchor_normal_divide_offset'][i] for i in batch_idx], dim=0)

        dists = torch.linalg.norm(target_param.unsqueeze(1) - anchors, dim=-1)
        gt_class = dists.argmin(-1)

        loss_coupled_anchor_class = F.cross_entropy(pred_coupled_anchor_class_logits, gt_class)

        losses = {}
        losses['loss_coupled_anchor_class'] = loss_coupled_anchor_class

        return losses

    def loss_normal_residual(self, outputs, targets, indices, num_planes_sum, anchors=None, focal_factors=None):
        idx = self._get_src_permutation_idx(indices)

        pred_ins_normals = outputs['pred_ins_normals'][idx]
        target_param = torch.cat([t["params"][J] for t, (_, J) in zip(targets, indices)])

        valid = torch.norm(target_param, dim=-1) > 1e-4
        pred_ins_normals = pred_ins_normals[valid]
        target_param = target_param[valid]

        target_normals = target_param / torch.linalg.norm(target_param, dim=-1).unsqueeze(-1)

        batch_idx, _ = idx
        anchor_normals = torch.stack([anchors['anchor_normals'][i] for i in batch_idx], dim=0)

        # plane_num, normal_anchor_num
        normal_dists = torch.linalg.norm(target_normals.unsqueeze(1) - anchor_normals, dim=-1)
        gt_normal_class = normal_dists.argmin(-1)

        # only train param residual over the correct plane class
        selected_pred_ins_normals = pred_ins_normals[torch.arange(pred_ins_normals.shape[0]), gt_normal_class]

        # l1 loss
        loss_normal_l1 = torch.mean(torch.sum(torch.abs(target_normals - selected_pred_ins_normals), dim=1))

        # cos loss
        similarity = torch.nn.functional.cosine_similarity(selected_pred_ins_normals, target_normals, dim=1)  # N
        loss_normal_cos = torch.mean(1 - similarity)

        losses = {}
        losses['loss_normal_residual'] = loss_normal_l1 + loss_normal_cos

        return losses

    def loss_offset_residual(self, outputs, targets, indices, num_planes_sum, anchors=None, focal_factors=None):
        idx = self._get_src_permutation_idx(indices)
        pred_ins_offsets = outputs['pred_ins_offsets'][idx]
        batch_idx, _ = idx

        target_param = torch.cat([t["params"][J] for t, (_, J) in zip(targets, indices)])

        valid = torch.norm(target_param, dim=-1) > 1e-4
        pred_ins_offsets = pred_ins_offsets[valid]
        target_param = target_param[valid]

        target_offsets = 1. / torch.linalg.norm(target_param, dim=-1)

        anchor_offsets = torch.stack([anchors['anchor_offsets'][i] for i in batch_idx], dim=0)

        offset_dists = torch.abs(target_offsets.unsqueeze(1) - anchor_offsets)

        gt_offset_class = offset_dists.argmin(-1)

        # select the best offset class and only supervise this prediction
        selected_pred_ins_offsets = pred_ins_offsets[torch.arange(pred_ins_offsets.shape[0]), gt_offset_class]
        # l1 loss
        loss_offset_l1 = torch.mean(torch.abs(target_offsets - selected_pred_ins_offsets))

        losses = {}
        losses['loss_offset_residual'] = loss_offset_l1

        return losses

    def loss_coupled_anchor_residual(self, outputs, targets, indices, num_planes_sum, anchors=None, focal_factors=None):
        idx = self._get_src_permutation_idx(indices)
        pred_coupled_anchor = outputs['pred_coupled_anchor'][idx]
        batch_idx, _ = idx

        target_param = torch.cat([t["params"][J] for t, (_, J) in zip(targets, indices)])

        valid = torch.norm(target_param, dim=-1) > 1e-4
        pred_coupled_anchor = pred_coupled_anchor[valid]
        target_param = target_param[valid]

        anchors = torch.stack([anchors['anchor_normal_divide_offset'][i] for i in batch_idx], dim=0)

        dists = torch.linalg.norm(target_param.unsqueeze(1) - anchors, dim=-1)
        gt_class = dists.argmin(-1)

        # select the best offset class and only supervise this prediction
        selected_pred_coupled_anchor = pred_coupled_anchor[torch.arange(pred_coupled_anchor.shape[0]), gt_class]

        loss_coupled_anchor_l1 = torch.mean(torch.abs(target_param - selected_pred_coupled_anchor))

        losses = {}
        losses['loss_coupled_anchor_residual'] = loss_coupled_anchor_l1

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, anchors=None, focal_factors=None):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'params': self.loss_instance_params,
            'Q': self.loss_Q,
            'plane_depths': self.loss_plane_depths,
            'global_pixel_depth': self.loss_global_pixel_depth,
            'global_pixel_normal': self.loss_global_pixel_normal,
            'instance_normals': self.loss_instance_normals,
            'instance_offsets': self.loss_instance_offsets,
            'instance_q': self.loss_instance_Q,
            'refine_instance_params': self.loss_refine_instance_params,
            'normal_class': self.loss_normal_class,
            'normal_residual': self.loss_normal_residual,
            'offset_class': self.loss_offset_class,
            'offset_residual': self.loss_offset_residual,
            'coupled_anchor_class': self.loss_coupled_anchor_class,
            'coupled_anchor_residual': self.loss_coupled_anchor_residual,
        }

        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, anchors=anchors, focal_factors=focal_factors)

    def forward(self, outputs, targets, anchors=None, focal_factors=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        valid_targets = []
        valid_t_idxs = []

        for t_idx, t in enumerate(targets):
            if t['masks'].size(0) > 0:
                valid_targets.append(t)
                valid_t_idxs.append(t_idx)

        valid_t_idxs = torch.tensor(valid_t_idxs).long().cuda()

        targets = valid_targets
        valid_outputs = dict()

        for k, v in outputs.items():
            if k != 'aux_outputs' and outputs[k] is not None:
                valid_outputs[k] = outputs[k][valid_t_idxs]

            else:
                valid_outputs[k] = []

                if outputs[k] is not None:
                    for o in outputs[k]:
                        o_dict = dict()

                        for kk, vv in o.items():
                            if o[kk] is not None:
                                o_dict[kk] = o[kk][valid_t_idxs]

                            else:
                                o_dict[kk] = None

                        valid_outputs[k].append(o_dict)

        outputs = valid_outputs

        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        # {'pred_logits': torch.Size([1, 100, 61]), 'pred_masks': torch.Size([1, 100, 120, 160])}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets) # targets: {"label":tensor([0, 3, 1, 4, 6, 5, 2, 7, 8], device='cuda:0'), "masks": torch.Size([9, 480, 640]) }

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets) # num_tgt_planes
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, anchors=anchors, focal_factors=focal_factors))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, anchors=anchors, focal_factors=focal_factors)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
