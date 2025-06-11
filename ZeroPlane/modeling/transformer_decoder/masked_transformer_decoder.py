# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# and by https://github.com/facebookresearch/Mask2Former
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .dpt_utils import Interpolate

# from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from detectron2.utils.registry import Registry

TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
TRANSFORMER_DECODER_REGISTRY.__doc__ = """
Registry for transformer module in PlaneRecTR.
"""


def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        param_prediction=True,
        depth_prediction=True,
        global_pixel_depth_prediction=False,
        global_pixel_normal_prediction=False,
        depth_normal_pred_upsample=False,
        learn_normal_class=False,
        learn_offset_class=False,
        learn_coupled_anchor_class=False,
        normal_class_num=7,
        offset_class_num=20,
        normal_pred_as_src=False,
        depth_pred_as_src=False,
        with_pixel_normal_attention=False,
        with_pixel_depth_attention=False,
        separate_pixel_attention=False,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        depth_dim: int,
        pixel_normal_dim: int,
        offset_dim: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.param_prediction = param_prediction
        self.depth_prediction = depth_prediction

        self.global_pixel_depth_prediction = global_pixel_depth_prediction
        self.global_pixel_normal_prediction = global_pixel_normal_prediction

        self.depth_normal_pred_upsample = depth_normal_pred_upsample

        self.with_pixel_normal_attention = with_pixel_normal_attention
        self.with_pixel_depth_attention = with_pixel_depth_attention

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        if self.with_pixel_normal_attention or self.with_pixel_depth_attention:
            self.separate_pixel_attention = separate_pixel_attention

            if self.separate_pixel_attention:
                if self.with_pixel_normal_attention:
                    self.pixel_normal_transformer_self_attention_layers = nn.ModuleList()
                    self.pixel_normal_transformer_cross_attention_layers = nn.ModuleList()
                    self.pixel_normal_transformer_ffn_layers = nn.ModuleList()

                if self.with_pixel_depth_attention:
                    self.pixel_depth_transformer_self_attention_layers = nn.ModuleList()
                    self.pixel_depth_transformer_cross_attention_layers = nn.ModuleList()
                    self.pixel_depth_transformer_ffn_layers = nn.ModuleList()

            else:
                self.pixel_geometry_transformer_self_attention_layers = nn.ModuleList()
                self.pixel_geometry_transformer_cross_attention_layers = nn.ModuleList()
                self.pixel_geometry_transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            if self.with_pixel_normal_attention or self.with_pixel_depth_attention:
                if self.separate_pixel_attention:
                    if self.with_pixel_normal_attention:
                        self.pixel_normal_transformer_self_attention_layers.append(
                            SelfAttentionLayer(
                                d_model=hidden_dim,
                                nhead=nheads,
                                dropout=0.0,
                                normalize_before=pre_norm,
                            )
                        )

                        self.pixel_normal_transformer_cross_attention_layers.append(
                            CrossAttentionLayer(
                                d_model=hidden_dim,
                                nhead=nheads,
                                dropout=0.0,
                                normalize_before=pre_norm,
                            )
                        )

                        self.pixel_normal_transformer_ffn_layers.append(
                            FFNLayer(
                                d_model=hidden_dim,
                                dim_feedforward=dim_feedforward,
                                dropout=0.0,
                                normalize_before=pre_norm,
                            )
                        )

                    if self.with_pixel_depth_attention:
                        self.pixel_depth_transformer_self_attention_layers.append(
                            SelfAttentionLayer(
                                d_model=hidden_dim,
                                nhead=nheads,
                                dropout=0.0,
                                normalize_before=pre_norm,
                            )
                        )

                        self.pixel_depth_transformer_cross_attention_layers.append(
                            CrossAttentionLayer(
                                d_model=hidden_dim,
                                nhead=nheads,
                                dropout=0.0,
                                normalize_before=pre_norm,
                            )
                        )

                        self.pixel_depth_transformer_ffn_layers.append(
                            FFNLayer(
                                d_model=hidden_dim,
                                dim_feedforward=dim_feedforward,
                                dropout=0.0,
                                normalize_before=pre_norm,
                            )
                        )

                else:
                    self.pixel_geometry_transformer_self_attention_layers.append(
                        SelfAttentionLayer(
                            d_model=hidden_dim,
                            nhead=nheads,
                            dropout=0.0,
                            normalize_before=pre_norm,
                        )
                    )

                    self.pixel_geometry_transformer_cross_attention_layers.append(
                        CrossAttentionLayer(
                            d_model=hidden_dim,
                            nhead=nheads,
                            dropout=0.0,
                            normalize_before=pre_norm,
                        )
                    )

                    self.pixel_geometry_transformer_ffn_layers.append(
                        FFNLayer(
                            d_model=hidden_dim,
                            dim_feedforward=dim_feedforward,
                            dropout=0.0,
                            normalize_before=pre_norm,
                        )
                    )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries

        # learnable query features
        self.query_feat = nn.Embedding(self.num_queries, hidden_dim)

        if self.num_layers > 0:
            # learnable query p.e.
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3

        if self.num_layers > 0:
            self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

        self.depth_pred_as_src = depth_pred_as_src
        if self.depth_pred_as_src:
            assert self.global_pixel_depth_prediction

        self.normal_pred_as_src = normal_pred_as_src
        if self.normal_pred_as_src:
            assert self.global_pixel_normal_prediction

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])

            else:
                self.input_proj.append(nn.Sequential())

        if self.normal_pred_as_src or self.with_pixel_normal_attention:
            self.normal_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                self.normal_proj.append(Conv2d(3, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.normal_proj[-1])

        if self.depth_pred_as_src or self.with_pixel_depth_attention:
            self.depth_proj = nn.ModuleList()
            for _ in range(self.num_feature_levels):
                self.depth_proj.append(Conv2d(1, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.depth_proj[-1])

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)

        if self.param_prediction:
            self.param_embed = nn.Linear(hidden_dim, 3)

        self.learn_normal_class = learn_normal_class

        self.learn_offset_class = learn_offset_class

        self.learn_coupled_anchor_class = learn_coupled_anchor_class

        self.normal_class_num = normal_class_num
        self.offset_class_num = offset_class_num

        if self.depth_prediction:
            self.depth_embed = MLP(hidden_dim, hidden_dim, depth_dim, 3)

        elif self.global_pixel_depth_prediction:
            # from DPT https://github.com/isl-org/DPT/blob/main/dpt/models.py#L99
            non_negative = True

            if self.depth_normal_pred_upsample:
                self.global_pixel_depth_head = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
                    Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels // 2, 32, kernel_size=3, stride=1, padding=1),
                    Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.ReLU(True),
                    nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(True) if non_negative else nn.Identity(),
                    nn.Identity(),
                )

            else:
                self.global_pixel_depth_head = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels // 2, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(True) if non_negative else nn.Identity(),
                    nn.Identity(),
                )

        if self.global_pixel_normal_prediction:
            if self.depth_normal_pred_upsample:
                self.global_pixel_normal_head = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
                    Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels // 2, 32, kernel_size=3, stride=1, padding=1),
                    Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.ReLU(True),
                    nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                    nn.Identity(),
                )

            else:
                self.global_pixel_normal_head = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(in_channels // 2, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                    nn.Identity(),
                )

        if self.learn_normal_class:
            self.normal_class_embed = nn.Linear(hidden_dim, self.normal_class_num)
            self.normal_residual_embed = nn.Linear(hidden_dim, self.normal_class_num*3)

            if self.learn_offset_class:
                self.offset_class_embed = nn.Linear(hidden_dim, self.offset_class_num)
                self.offset_residual_embed = nn.Linear(hidden_dim, self.offset_class_num)

        elif self.learn_coupled_anchor_class:
            self.coupled_anchor_class_embed = nn.Linear(hidden_dim, self.normal_class_num)
            self.coupled_anchor_residual_embed = nn.Linear(hidden_dim, self.normal_class_num*3)

        else:
            assert self.param_prediction

        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        ret["param_prediction"] = cfg.MODEL.MASK_FORMER.PREDICT_PARAM
        ret["depth_prediction"] = cfg.MODEL.MASK_FORMER.PREDICT_DEPTH

        ret["global_pixel_depth_prediction"] = cfg.MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_DEPTH
        ret["global_pixel_normal_prediction"] = cfg.MODEL.MASK_FORMER.PREDICT_GLOBAL_PIXEL_NORMAL

        ret['depth_normal_pred_upsample'] = cfg.MODEL.MASK_FORMER.DEPTH_NORMAL_PRED_UPSAMPLE

        ret['learn_normal_class'] = cfg.MODEL.MASK_FORMER.LEARN_NORMAL_CLS

        ret['learn_offset_class'] = cfg.MODEL.MASK_FORMER.LEARN_OFFSET_CLS

        ret['learn_coupled_anchor_class'] = cfg.MODEL.MASK_FORMER.USE_COUPLED_ANCHOR

        ret['normal_class_num'] = cfg.MODEL.MASK_FORMER.NORMAL_CLS_NUM
        ret['offset_class_num'] = cfg.MODEL.MASK_FORMER.OFFSET_CLS_NUM

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["depth_dim"] = cfg.MODEL.SEM_SEG_HEAD.DEPTH_DIM
        ret["pixel_normal_dim"] = cfg.MODEL.SEM_SEG_HEAD.PIXEL_NORMAL_DIM
        ret['offset_dim'] = cfg.MODEL.SEM_SEG_HEAD.OFFSET_DIM

        ret['normal_pred_as_src'] = cfg.MODEL.MASK_FORMER.NORMAL_PRED_AS_SRC
        ret['depth_pred_as_src'] = cfg.MODEL.MASK_FORMER.DEPTH_PRED_AS_SRC

        ret['with_pixel_normal_attention'] = cfg.MODEL.MASK_FORMER.WITH_PIXEL_NORMAL_ATTENTION
        ret['with_pixel_depth_attention'] = cfg.MODEL.MASK_FORMER.WITH_PIXEL_DEPTH_ATTENTION

        ret['separate_pixel_attention'] = cfg.MODEL.MASK_FORMER.SEPARATE_PIXEL_ATTENTION

        return ret

    def forward(self, x, mask_features, mask=None, anchors=None):

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        if self.with_pixel_normal_attention or self.with_pixel_depth_attention:
            if self.separate_pixel_attention:
                if self.with_pixel_depth_attention:
                    pixel_depth_src = []

                if self.with_pixel_normal_attention:
                    pixel_normal_src = []

            else:
                pixel_geo_src = []

        if self.num_layers > 0:
            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))

                src_ = self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None]

                if self.normal_pred_as_src or self.with_pixel_normal_attention:
                    normal_i = self.global_pixel_normal_head(x[i])
                    normal_i = normal_i / (torch.norm(normal_i, dim=1).unsqueeze(1) + 1e-8)

                    normal_src = self.normal_proj[i](normal_i).flatten(2)

                    if self.normal_pred_as_src:
                        src_ = src_ + normal_src

                    if self.with_pixel_normal_attention:
                        if self.separate_pixel_attention:
                            pixel_normal_src.append(normal_src)

                        else:
                            pixel_geo_src.append(normal_src)

                if self.depth_pred_as_src or self.with_pixel_depth_attention:
                    depth_i = self.global_pixel_depth_head(x[i])
                    depth_src = self.depth_proj[i](depth_i).flatten(2)

                    if self.depth_pred_as_src:
                        src_ = src_ + depth_src

                    if self.with_pixel_depth_attention:
                        if self.separate_pixel_attention:
                            pixel_depth_src.append(depth_src)

                        else:
                            if self.with_pixel_normal_attention:
                                pixel_geo_src[-1] = pixel_geo_src[-1] + depth_src

                            else:
                                pixel_geo_src.append(depth_src)

                src.append(src_)

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)  # [950/3800/15200, 1, 256]
                src[-1] = src[-1].permute(2, 0, 1)  # [950/3800/15200, 1, 256]

                if self.with_pixel_normal_attention or self.with_pixel_depth_attention:
                    if self.separate_pixel_attention:
                        pixel_depth_src[-1] = pixel_depth_src[-1].permute(2, 0, 1)
                        pixel_normal_src[-1] = pixel_normal_src[-1].permute(2, 0, 1)

                    else:
                        pixel_geo_src[-1] = pixel_geo_src[-1].permute(2, 0, 1)

        else:
            # no use just to make the code run
            size_list.append(x[self.num_feature_levels-1].shape[-2:])

        bs = mask_features.size(0)

        if self.num_layers > 0:
            # QxNxC
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # [100, 1, 256]

        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)  # [100, 1, 256]

        predictions_class = []
        predictions_mask = []
        predictions_param = []
        predictions_depth = []

        predictions_pixel_normal = []

        predictions_ins_normal = []
        predictions_ins_offset = []

        predictions_ins_normal_class = []
        predictions_ins_offset_class = []

        predictions_coupled_anchor = []
        predictions_coupled_anchor_class = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask, outputs_param, outputs_depth, outputs_pixel_normal, outputs_ins_normal, outputs_ins_offset, outputs_coupled_anchor, outputs_ins_normal_class, outputs_ins_offset_class, outputs_coupled_anchor_class = \
            self.forward_prediction_heads(output, x, mask_features, attn_mask_target_size=size_list[0], anchors=anchors)

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_param.append(outputs_param)

        predictions_depth.append(outputs_depth)
        predictions_pixel_normal.append(outputs_pixel_normal)

        predictions_ins_normal.append(outputs_ins_normal)
        predictions_ins_offset.append(outputs_ins_offset)

        predictions_ins_normal_class.append(outputs_ins_normal_class)
        predictions_ins_offset_class.append(outputs_ins_offset_class)

        predictions_coupled_anchor.append(outputs_coupled_anchor)
        predictions_coupled_anchor_class.append(outputs_coupled_anchor_class)

        for i in range(self.num_layers): # num_layers = 9
            level_index = i % self.num_feature_levels # 3
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  #[]

            if self.with_pixel_normal_attention or self.with_pixel_depth_attention:
                if self.separate_pixel_attention:
                    if self.with_pixel_depth_attention:
                        # attention: cross-attention first
                        pixel_depth_enhanced_output = self.pixel_depth_transformer_cross_attention_layers[i](
                            output, pixel_depth_src[level_index],
                            memory_mask=attn_mask,
                            memory_key_padding_mask=None,  # here we do not apply masking on padded region
                            pos=pos[level_index], query_pos=query_embed
                        )

                        pixel_depth_enhanced_output = self.pixel_depth_transformer_self_attention_layers[i](
                            pixel_depth_enhanced_output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=query_embed
                        )  # (100, 1, 256)

                        # FFN
                        pixel_depth_enhanced_output = self.pixel_depth_transformer_ffn_layers[i](
                            pixel_depth_enhanced_output
                        )  # (100, 1, 256)

                    if self.with_pixel_normal_attention:
                        # attention: cross-attention first
                        pixel_normal_enhanced_output = self.pixel_normal_transformer_cross_attention_layers[i](
                            output, pixel_normal_src[level_index],
                            memory_mask=attn_mask,
                            memory_key_padding_mask=None,  # here we do not apply masking on padded region
                            pos=pos[level_index], query_pos=query_embed
                        )

                        pixel_normal_enhanced_output = self.pixel_normal_transformer_self_attention_layers[i](
                            pixel_normal_enhanced_output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=query_embed
                        )  # (100, 1, 256)

                        # FFN
                        pixel_normal_enhanced_output = self.pixel_normal_transformer_ffn_layers[i](
                            pixel_normal_enhanced_output
                        )  # (100, 1, 256)

                else:
                    # attention: cross-attention first
                    geo_enhanced_output = self.pixel_geometry_transformer_cross_attention_layers[i](
                        output, pixel_geo_src[level_index],
                        memory_mask=attn_mask,
                        memory_key_padding_mask=None,  # here we do not apply masking on padded region
                        pos=pos[level_index], query_pos=query_embed
                    )

                    geo_enhanced_output = self.pixel_geometry_transformer_self_attention_layers[i](
                        geo_enhanced_output, tgt_mask=None,
                        tgt_key_padding_mask=None,
                        query_pos=query_embed
                    )  # (100, 1, 256)

                    # FFN
                    geo_enhanced_output = self.pixel_geometry_transformer_ffn_layers[i](
                        geo_enhanced_output
                    )  # (100, 1, 256)

            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            ) # (100, 1, 256)

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            ) # (100, 1, 256)

            if self.with_pixel_normal_attention or self.with_pixel_depth_attention:
                if self.separate_pixel_attention:
                    if self.with_pixel_normal_attention:
                        output = output + pixel_normal_enhanced_output

                    if self.with_pixel_depth_attention:
                        output = output + pixel_depth_enhanced_output

                else:
                    output = output + geo_enhanced_output

            outputs_class, outputs_mask, attn_mask, outputs_param, outputs_depth, outputs_pixel_normal, outputs_ins_normal, outputs_ins_offset, outputs_coupled_anchor, outputs_ins_normal_class, outputs_ins_offset_class, outputs_coupled_anchor_class = \
                self.forward_prediction_heads(output, x, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], anchors=anchors)

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_param.append(outputs_param)

            predictions_depth.append(outputs_depth)
            predictions_pixel_normal.append(outputs_pixel_normal)

            predictions_ins_normal.append(outputs_ins_normal)
            predictions_ins_offset.append(outputs_ins_offset)

            predictions_coupled_anchor.append(outputs_coupled_anchor)

            predictions_ins_normal_class.append(outputs_ins_normal_class)
            predictions_ins_offset_class.append(outputs_ins_offset_class)

            predictions_coupled_anchor_class.append(outputs_coupled_anchor_class)

        assert len(predictions_class) == self.num_layers + 1 #10

        out = {
            'pred_logits': predictions_class[-1], # [b, num_queries, num_classes + 1]
            'pred_masks': predictions_mask[-1], # [b, num_queries, h/4, w/4]
            'pred_params': predictions_param[-1], # [b, num_queries,3] or None
            'pred_depths': predictions_depth[-1], # [b, num_queries, h/4, w/4]
            'pred_pixel_normals': predictions_pixel_normal[-1], # [b, num_queries, 3, h/4, w/4]
            'pred_ins_normals': predictions_ins_normal[-1],
            'pred_ins_offsets': predictions_ins_offset[-1],
            'pred_coupled_anchor': predictions_coupled_anchor[-1],
            'pred_ins_normal_class': predictions_ins_normal_class[-1],
            'pred_ins_offset_class': predictions_ins_offset_class[-1],
            'pred_coupled_anchor_class': predictions_coupled_anchor_class[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_param, predictions_depth, predictions_pixel_normal, \
                predictions_ins_normal, predictions_ins_offset, predictions_coupled_anchor, predictions_ins_normal_class, predictions_ins_offset_class, predictions_coupled_anchor_class
            ),  # [ {"pred_logits": xx, "pred_masks": xx, , "pred_params": xx, "pred_depths": xx}*9],
        }

        return out

    def forward_prediction_heads(self, output, multiscale_features, mask_features, attn_mask_target_size, anchors=None): #mask_features [b, 256, h/4, w/4]]
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1) # (b, num_queries, 256)

        instance_decoder_output = decoder_output
        outputs_class = self.class_embed(instance_decoder_output) # (b, num_queries, 134) float16

        mask_embed = self.mask_embed(instance_decoder_output) # (b, num_queries, 256)
        outputs_mask = torch.einsum("bqc, bchw->bqhw", mask_embed, mask_features) # [b, num_queries, h/4, w/4]

        if self.depth_prediction:
            depth_embed = self.depth_embed(instance_decoder_output) # (b, num_queries, 256)
            outputs_depth = torch.einsum("bqc, bchw->bqhw", depth_embed, mask_features) # [b, num_queries, h/4, w/4]

        elif self.global_pixel_depth_prediction:
            outputs_depth = self.global_pixel_depth_head(mask_features)

        else:
            outputs_depth = None

        if self.global_pixel_normal_prediction:
            outputs_pixel_normal = self.global_pixel_normal_head(mask_features)
            outputs_pixel_normal = outputs_pixel_normal / (torch.linalg.norm(outputs_pixel_normal, dim=1).unsqueeze(1) + 1e-8)

        else:
            outputs_pixel_normal = None

        outputs_ins_normal = None

        outputs_ins_normal_class = None
        outputs_ins_offset_class = None

        if self.learn_normal_class:
            bs, num_queries = instance_decoder_output.shape[:2]

            outputs_ins_normal_class = self.normal_class_embed(instance_decoder_output)

            outputs_normal_residual = self.normal_residual_embed(instance_decoder_output).view(bs, num_queries, -1, 3)
            outputs_ins_normal = anchors['anchor_normals'].unsqueeze(1) + outputs_normal_residual

            outputs_ins_normal = outputs_ins_normal / (torch.linalg.norm(outputs_ins_normal, dim=-1).unsqueeze(-1) + 1e-4)

            if self.learn_offset_class:
                outputs_ins_offset_class = self.offset_class_embed(instance_decoder_output)
                outputs_offset_residual = self.offset_residual_embed(instance_decoder_output)
                outputs_ins_offset = anchors['anchor_offsets'].unsqueeze(1) + outputs_offset_residual

        outputs_coupled_anchor = None
        outputs_coupled_anchor_class = None

        if self.learn_coupled_anchor_class:
            bs, num_queries = instance_decoder_output.shape[:2]

            outputs_coupled_anchor_class = self.coupled_anchor_class_embed(instance_decoder_output)
            outputs_coupled_anchor_residual = self.coupled_anchor_residual_embed(instance_decoder_output).view(bs, num_queries, -1, 3)

            outputs_coupled_anchor = anchors['anchor_normal_divide_offset'].unsqueeze(1) + outputs_coupled_anchor_residual

        if self.param_prediction:
            outputs_param = self.param_embed(instance_decoder_output) # (b, num_queries, 3)

        elif self.learn_normal_class:
            assert self.learn_offset_class

            bs, num_queries, num_normal_cls, _ = outputs_ins_normal.size()

            pred_normal_class = outputs_ins_normal_class.argmax(-1)
            pred_ins_normals = torch.stack([ins_normal[torch.arange(num_queries), normal_class] for ins_normal, normal_class in zip(outputs_ins_normal, pred_normal_class)])

            bs, num_queries, num_offset_cls = outputs_ins_offset_class.size()
            pred_offset_class = outputs_ins_offset_class.argmax(-1)
            pred_ins_offsets = torch.stack([ins_offset[torch.arange(num_queries), offset_class] for ins_offset, offset_class in zip(outputs_ins_offset, pred_offset_class)])

            outputs_param = pred_ins_normals / (pred_ins_offsets.unsqueeze(-1) + 1e-4)

        else:
            outputs_param = None

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask, outputs_param, outputs_depth, outputs_pixel_normal, outputs_ins_normal, outputs_ins_offset, outputs_coupled_anchor, outputs_ins_normal_class, outputs_ins_offset_class, outputs_coupled_anchor_class

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_param, outputs_depth, outputs_pixel_normal, outputs_ins_normal, outputs_ins_offset, outputs_coupled_anchor, outputs_ins_normal_class, outputs_ins_offset_class, outputs_coupled_anchor_class):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_params":c, "pred_depths": d, "pred_pixel_normals": e, \
                 'pred_ins_normals': g, 'pred_ins_offsets': h, 'pred_ins_normal_class': i, 'pred_ins_offset_class': j, \
                 'pred_coupled_anchor': m, 'pred_coupled_anchor_class': n}
                for a, b, c, d, e, g, h, i, j, m, n in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_param[:-1], outputs_depth[:-1], outputs_pixel_normal[:-1], outputs_ins_normal[:-1], outputs_ins_offset[:-1], outputs_ins_normal_class[:-1], outputs_ins_offset_class[:-1], outputs_coupled_anchor[:-1], outputs_coupled_anchor_class[:-1])
            ]

        else:
            return [{"pred_masks": b, "pred_params": c, "pred_depths": d, "pred_pixel_normals": e, \
                     'pred_ins_normals': g, 'pred_ins_offsets': h, 'pred_ins_normal_class': i, 'pred_ins_offset_class': j, \
                     'pred_coupled_anchor': m, 'pred_coupled_anchor_class': n}
                    for b, c, d, e, g, h, i, j, l, m, n in zip(outputs_seg_masks[:-1], outputs_param[:-1], outputs_depth[:-1], outputs_pixel_normal[:-1], outputs_ins_normal[:-1], outputs_ins_offset[:-1], outputs_ins_normal_class[:-1], outputs_ins_offset_class[:-1], outputs_coupled_anchor[:-1], outputs_coupled_anchor_class[:-1])
                    ]
