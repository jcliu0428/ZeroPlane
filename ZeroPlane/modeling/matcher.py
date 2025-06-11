# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
# and by https://github.com/facebookresearch/Mask2Former
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
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
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1,
                 cost_param: float = 1, cost_depth: float = 1, cost_pixel_normal: float = 1,
                 predict_param: bool = True, predict_depth: bool = True, predict_pixel_normal: bool = False,
                 num_points: int = 0, normalize_param: bool = False):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_param = cost_param
        self.cost_depth = cost_depth
        self.cost_pixel_normal = cost_pixel_normal

        self.predict_depth = predict_depth
        self.predict_param = predict_param

        self.predict_pixel_normal = predict_pixel_normal

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

        self.normalize_param = normalize_param

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2] # 1, 100

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes] ep [60, 3]
            tgt_ids = targets[b]["labels"] # [num_tgt_planes]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids] # [num_queries, num_tgt_planes]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred] ep torch.Size([100, 120, 160])
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask) #  [num_tgt_planes, H_tgt, W_tgt] ep torch.Size([8, 480, 640])

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1) # torch.Size([num_tgt_planes, num_points])

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1) # torch.Size([num_queries, num_points])

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
                # + self.cost_param * cost_param
            ) # torch.Size([100, 9])

            if self.predict_param:
                # Compute the L1 cost between params
                out_param = outputs["pred_params"][b] # [num_queries, 3]
                tgt_param = targets[b]["params"] # [num_tgt_planes, 3]

                if self.normalize_param:
                    out_param = out_param / (torch.linalg.norm(out_param, dim=-1) + 1e-8).unsqueeze(-1)
                    tgt_param = tgt_param / (torch.linalg.norm(tgt_param, dim=-1) + 1e-8).unsqueeze(-1)

                cost_param = torch.cdist(out_param, tgt_param, p=1)  # [num_queries , num_target_planes]

                C += self.cost_param * cost_param

            if self.predict_depth:
                out_depth = outputs["pred_depths"][b].flatten(start_dim = 1)  # [num_queries, H_pred*W_pred]
                tgt_depth = targets[b]["plane_depths"].flatten(start_dim = 1)  # [num_tgt_planes, h*w]
                depth_mask = targets[b]["masks"].to(out_mask)[None,:,:,:].flatten(start_dim = 2)  # [1, num_tgt_planes, h*w]

                if out_depth.shape[-1] != tgt_depth.shape[-1]:
                    tgt_depth = targets[b]["resize14_plane_depths"].flatten(start_dim = 1)  # [num_tgt_planes, (h/4)*(w/4)]
                    depth_mask = (tgt_depth > 0)[None,:,:]

                cost_depth = (torch.abs(
                    (out_depth[:,None,:] - tgt_depth[None,:,:]) * depth_mask  # [num_queries, num_tgt_planes, h*w] * [1, num_tgt_planes, h*w]
                    ).sum(dim = -1) + 1) / torch.clip(depth_mask.sum(dim = -1).float(), min=1e-6)  # [num_queries, num_tgt_planes]/ [1, num_tgt_planes]
                C += self.cost_depth * cost_depth

            if self.predict_pixel_normal:
                out_pixel_normal = outputs['pred_pixel_normals'][b].flatten(start_dim=2)
                tgt_pixel_normal = targets[b]['resize14_pixel_normals'].flatten(start_dim=2)

                pixel_normal_mask = torch.norm(tgt_pixel_normal, dim=1) > 1e-4
                pixel_normal_mask = torch.stack([pixel_normal_mask, pixel_normal_mask, pixel_normal_mask], dim=1)

                cost_pixel_normal = (torch.abs(
                    (out_pixel_normal[:, None] - tgt_pixel_normal[None, ...]) * pixel_normal_mask
                ).sum(dim=-1) + 1) / torch.clip(pixel_normal_mask.sum(-1).float(), min=1e-6)
                cost_pixel_normal = cost_pixel_normal.sum(dim=-1)

                C += self.cost_pixel_normal * cost_pixel_normal

            # if torch.sum(torch.isnan(C))>0:

            C = C.reshape(num_queries, -1).cpu() # ep torch.Size([num_queries, num_tgt_planes])

            indices.append(linear_sum_assignment(C))

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks
                 "params": Tensor of dim [num_target_boxes, 3] containing the target plane parameters

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
            "cost_param: {}".format(self.cost_param),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
