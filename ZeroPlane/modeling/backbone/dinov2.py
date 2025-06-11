import torch
import torch.nn as nn

from detectron2.modeling import BACKBONE_REGISTRY, Backbone


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

        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn == True:
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
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

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

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class DPTEncoder(nn.Module):
    def __init__(self, in_channels, use_bn=False,
                 out_channels=[256, 512, 1024, 1024], use_clstoken=False, features=256):
        super(DPTEncoder, self).__init__()

        self.use_clstoken = use_clstoken

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        # self.scratch = _make_scratch(
        #     out_channels,
        #     features,
        #     groups=1,
        #     expand=False,
        # )

        # self.scratch.stem_transpose = None

        # self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        # self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # head_features_1 = features
        # head_features_2 = 32

        # self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)

        # self.scratch.output_conv2 = nn.Sequential(
        #     nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True),
        #     nn.Identity(),
        # )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            # tokenized feature, (bs, sq_len, feat)
            out.append(x)

        # for out_path in out:
        #     print(out_path.size())

        # layer_1, layer_2, layer_3, layer_4 = out

        # layer_1_rn = self.scratch.layer1_rn(layer_1)
        # layer_2_rn = self.scratch.layer2_rn(layer_2)
        # layer_3_rn = self.scratch.layer3_rn(layer_3)
        # layer_4_rn = self.scratch.layer4_rn(layer_4)

        # path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        # path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        # path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        # path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # for path in [path_1, path_2, path_3, path_4]:
        #     print(path.size())

        # exit(1)

        # out = self.scratch.output_conv1(path_1)
        # out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        # out = self.scratch.output_conv2(out)

        return out


@BACKBONE_REGISTRY.register()
class DPT_DINOv2(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()

        encoder = cfg.MODEL.DINOv2.ENCODER
        out_channels = cfg.MODEL.DINOv2.OUT_CHANNELS
        use_bn = cfg.MODEL.DINOv2.USE_BN
        use_clstoken = cfg.MODEL.DINOv2.USE_CLSTOKEN
        localhub = cfg.MODEL.DINOv2.LOCALHUB

        load_dav1_backbone = cfg.MODEL.DINOv2.LOAD_DAv1
        load_dav2_backbone = cfg.MODEL.DINOv2.LOAD_DAv2

        self._out_features = cfg.MODEL.DINOv2.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": out_channels[0],
            "res3": out_channels[1],
            "res4": out_channels[2],
            "res5": out_channels[3],
        }

        assert encoder in ['vits', 'vitb', 'vitl']

        if load_dav1_backbone:
            # load arch
            self.pretrained = torch.hub.load('./third_party/Depth-Anything/torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=True)
            dav1_weights = torch.load('./third_party/Depth-Anything/checkpoints/depth_anything_vitb14.pth', map_location='cpu')

            dav1_dino_weights = {}
            for key, val in dav1_weights.items():
                if not key.startswith('pretrained'):
                    continue

                dav1_dino_weights[key.replace('pretrained.', '')] = val

            # update params
            self.pretrained.load_state_dict(dav1_dino_weights, strict=True)

        elif load_dav2_backbone:
            # load arch
            self.pretrained = torch.hub.load('./third_party/Depth-Anything/torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=True)
            dav2_weights = torch.load('./third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth', map_location='cpu')

            dav2_dino_weights = {}
            for key, val in dav2_weights.items():
                if not key.startswith('pretrained'):
                    continue

                dav2_dino_weights[key.replace('pretrained.', '')] = val

            # update params
            self.pretrained.load_state_dict(dav2_dino_weights, strict=True)

        else:
            # in case the Internet connection is not stable, please load the DINOv2 locally
            if localhub:
                print('loading pretrained dino models...')
                self.pretrained = torch.hub.load('./third_party/Depth-Anything/torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=True)

            else:
                self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))

        dim = self.pretrained.blocks[0].attn.qkv.in_features

        self.depth_head = DPTEncoder(dim, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        self.freeze_dino_params = cfg.MODEL.DINOv2.FREEZE_BACKBONE

        if self.freeze_dino_params:
            self.freeze_pretrained_parameters(['pretrained'])

        # for name, param in self.named_parameters():
        #     print(name, param.requires_grad)

        # exit(1)

    def freeze_pretrained_parameters(self, layers_prefix_to_freeze):
        """
        Freeze parameters of specified layers.

        Args:
            layers_prefix_to_freeze (list of str): List of layer names to freeze.
        """
        for name, param in self.named_parameters():
            for layer_prefix in layers_prefix_to_freeze:
                if name.startswith(layer_prefix):
                    param.requires_grad = False
                    print(f"Freezing: {name}")

    def forward(self, x):
        h, w = x.shape[-2:]

        # tokenized feature, (bs, t, feat_dim)
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)

        patch_h, patch_w = h // 14, w // 14

        # multi-res spatial dense feature, (bs, c, h, w)
        features = self.depth_head(features, patch_h, patch_w)

        res = {}
        for idx, feat in enumerate(features):
            res['res' + str(idx + 2)] = feat

        # patch_h, patch_w = h // 14, w // 14

        # depth = self.depth_head(features, patch_h, patch_w)
        # depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        # depth = F.relu(depth)

        return res
