import types

import torch
import torch.nn as nn
from timm.models.edgenext import ConvBlock, SplitTransposeBlock

from .layer_mods import ConvBlockExport, SplitTransposeBlockExport


def normalize_image(img: torch.Tensor) -> torch.Tensor:
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (img / 255.0 - mean) / std


class FeatureProjection(nn.Module):
    def __init__(self, model: nn.Module, image_shape: tuple[int, int, int, int], patch_encoder: bool = True):
        super().__init__()
        if patch_encoder:
            self.feature = patch_edgenext(model.feature,  image_shape)
        else:
            self.feature = model.feature
        self.stem_2 = model.stem_2

        self.cnet = model.cnet
        self.cam = model.cam
        self.sam = model.sam
        self.proj_cmb = model.proj_cmb

    def forward(self, left_image: torch.Tensor, right_image: torch.Tensor):
        left_image = normalize_image(left_image)
        right_image = normalize_image(right_image)

        features = self.feature(torch.cat([left_image, right_image], dim=0))
        split_features = [torch.chunk(item, chunks=2, dim=0) for item in features]
        left_features = [f[0] for f in split_features]
        right_features = [f[1] for f in split_features]

        stem_2x = self.stem_2(left_image)

        cnet_list = self.cnet(left_features[0], left_features[1], left_features[2])
        cnet_list = list(cnet_list)
        net_list = [torch.tanh(x[0]) for x in cnet_list]
        inp_list = [torch.relu(x[1]) for x in cnet_list]
        inp_list = [self.cam(x) * x for x in inp_list]
        att = [self.sam(x) for x in inp_list]
        proj_left_04 = self.proj_cmb(left_features[0])
        proj_right_04 = self.proj_cmb(right_features[0])
        return (
            left_features[0],
            left_features[1],
            left_features[2],
            left_features[3],
            right_features[0],
            proj_left_04,
            proj_right_04,
            stem_2x,
            net_list,
            inp_list,
            att,
        )


def patch_edgenext(model: nn.Module, input_size=(1, 3, 480, 640)) -> nn.Module:
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(*input_size).to(next(model.parameters()).device)

        # Initial run to compute and cache positional embeddings
        model(dummy_input)

    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, SplitTransposeBlock):
            replacements.append((name, SplitTransposeBlockExport.from_original(module)))
        elif isinstance(module, ConvBlock):
            replacements.append((name, ConvBlockExport.from_original(module)))

    # 3. Apply replacements recursively
    for name, new_module in replacements:
        # Split 'stage1.blocks.0.conv' into 'stage1.blocks.0' and 'conv'
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            # Navigate to the parent module
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, child_name, new_module)
        else:
            # It's a top-level module
            setattr(model, name, new_module)

    model.forward = types.MethodType(forward_no_profiler, model)
    return model


def forward_no_profiler(self, x):
    if hasattr(self, 'stem'):
        x = self.stem(x)
        x4 = self.stages[0](x)
        x8 = self.stages[1](x4)
        x16 = self.stages[2](x8)
        x32 = self.stages[3](x16)
    else:
        intermediates = self.model.forward_intermediates(x, intermediates_only=True)
        x4, x8, x16, x32 = intermediates[-4:]

    x16 = self.deconv32_16(x32, x16)
    x8 = self.deconv16_8(x16, x8)
    x4 = self.deconv8_4(x8, x4)
    x4 = self.conv4(x4)
    if hasattr(self, 'conv8'):
        x8 = self.conv8(x8)
        x16 = self.conv16(x16)
        x32 = self.conv32(x32)
    return [x4, x8, x16, x32]
