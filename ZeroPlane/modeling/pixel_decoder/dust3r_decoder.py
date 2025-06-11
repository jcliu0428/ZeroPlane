import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch.cuda.amp import autocast

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.config import configurable
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


@SEM_SEG_HEADS_REGISTRY.register()
class Dust3RDecoder(nn.Module):
    @configurable
    def __init__(self,
                 features):
        super().__init__()

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
        ret['features'] = cfg.MODEL.DUST3R.FEATURES

        return ret

    # @autocast(enabled=False)
    def forward_features(self, features):
        # res from low to high
        out = [features['res2'], features['res3'], features['res4'], features['res5']]

        transformer_encoder_feature = out[0]

        num_cur_levels = 0
        multi_scale_features = []

        # for input of transformer decoder
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        # mask feat: (96, 128)
        # multi-scale: (12, 16), (24, 32), (48, 64), (96, 128)
        return self.mask_features(out[-2]), transformer_encoder_feature, multi_scale_features
