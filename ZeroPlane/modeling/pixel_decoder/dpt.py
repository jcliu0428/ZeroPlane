import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch.cuda.amp import autocast

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.config import configurable
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape

    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if size is None:
            modifier = {"scale_factor": 1}

        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True
    )


class DPTHead(nn.Module):
    def __init__(self, nclass, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024]):
        super(DPTHead, self).__init__()

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False
        )

        self.out_channels = features

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # head_features_1 = features
        # head_features_2 = 32

        # if nclass > 1:
        #     self.scratch.output_conv = nn.Sequential(
        #         nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(True),
        #         nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
        #     )

        # else:
        #     self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

        #     self.scratch.output_conv2 = nn.Sequential(
        #         nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
        #         nn.ReLU(True),
        #         nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
        #         nn.ReLU(True),
        #         nn.Identity()
        #     )

    def forward(self, out):
        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        return path_4, path_3, path_2, path_1


@SEM_SEG_HEADS_REGISTRY.register()
class DPTDecoder(nn.Module):
    @configurable
    def __init__(self,
                 nclass,
                 features,
                 out_channels,
                 use_bn):
        super().__init__()

        self.decoder = DPTHead(nclass, features, use_bn, out_channels)
        self.maskformer_num_feature_levels = 3

        # use 1x1 conv instead
        self.mask_features = Conv2d(
            features,
            features,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        weight_init.c2_xavier_fill(self.mask_features)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}

        ret['nclass'] = cfg.MODEL.DINOv2.NCLASS
        ret['features'] = cfg.MODEL.DINOv2.FEATURES

        if cfg.MODEL.BACKBONE.NAME == 'D2SwinTransformer':
            ret['out_channels'] = [128, 256, 512, 1024]

        elif cfg.MODEL.BACKBONE.NAME == 'build_resnet_backbone':
            ret['out_channels'] = [256, 512, 1024, 2048]

        elif cfg.MODEL.BACKBONE.NAME == 'ConvNeXt':
            ret['out_channels'] = cfg.MODEL.convnext.OUT_CHANNELS

        elif cfg.MODEL.BACKBONE.NAME == 'DPT_DINOv2':
            ret['out_channels'] = cfg.MODEL.DINOv2.OUT_CHANNELS

        ret['use_bn'] = cfg.MODEL.DINOv2.USE_BN

        return ret

    # @autocast(enabled=False)
    def forward_features(self, features):
        features_list = [features['res2'], features['res3'], features['res4'], features['res5']]

        out = self.decoder(features_list)

        transformer_encoder_feature = out[0]

        num_cur_levels = 0
        multi_scale_features = []

        # for input of transformer decoder
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        # mask feat: (60, 80)
        # multi-scale: (15, 20), (30, 40), (60, 80)
        return self.mask_features(out[-1]), transformer_encoder_feature, multi_scale_features
