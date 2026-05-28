import torch
import torch.nn as nn
import torch.nn.functional as F


class CostStemClassifier(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.corr_stem = model.corr_stem
        self.corr_feature_att = model.corr_feature_att
        self.cost_agg = model.cost_agg
        self.classifier = model.classifier

    def forward(
        self,
        combined_volume: torch.Tensor,
        left_04: torch.Tensor,
        left_08: torch.Tensor,
        left_16: torch.Tensor,
        left_32: torch.Tensor,
    ):
        left_features = [left_04, left_08, left_16, left_32]
        volume = self.corr_stem(combined_volume)
        volume = self.corr_feature_att(volume, left_04)
        volume = self.cost_agg(volume, left_features)
        logits = self.classifier(volume).squeeze(1)
        return volume, logits


class CorrStemAttention(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.corr_stem = model.corr_stem
        self.corr_feature_att = model.corr_feature_att

    def forward(self, combined_volume: torch.Tensor, left_04: torch.Tensor):
        volume = self.corr_stem(combined_volume)
        volume = self.corr_feature_att(volume, left_04)
        return volume


class CostAggDown(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.conv3 = model.cost_agg.conv3
        self.feature_att_8 = model.cost_agg.feature_att_8
        self.feature_att_16 = model.cost_agg.feature_att_16
        self.feature_att_32 = model.cost_agg.feature_att_32

    def forward(
        self,
        stem_volume: torch.Tensor,
        left_08: torch.Tensor,
        left_16: torch.Tensor,
        left_32: torch.Tensor,
    ):
        conv1 = self.feature_att_8(stem_volume, left_08)
        conv2 = self.feature_att_16(conv1, left_16)
        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, left_32)
        return conv1, conv2, conv3


class CostAggPost32Out(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.post = model.cost_agg.post32_to_16

    def forward(
        self,
        post32_deconv_raw: torch.Tensor,
        cost_conv2: torch.Tensor,
        left_16: torch.Tensor,
    ):
        upsample_block = self.post.upsample[0]
        conv3_up = upsample_block.bn(post32_deconv_raw)
        if isinstance(upsample_block.relu, bool):
            if upsample_block.relu:
                conv3_up = F.leaky_relu(conv3_up, negative_slope=0.01)
        else:
            conv3_up = upsample_block.relu(conv3_up)
        x = conv3_up + cost_conv2
        for layer in self.post.out:
            if layer.__class__.__name__ == "FeatureAtt":
                x = layer(x, left_16)
            else:
                x = layer(x)
        return x


class CostAggPost16Out(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.post = model.cost_agg.post16_to_8

    def forward(
        self,
        post16_deconv_raw: torch.Tensor,
        cost_conv1: torch.Tensor,
        left_08: torch.Tensor,
    ):
        upsample_block = self.post.upsample[0]
        conv2_up = upsample_block.bn(post16_deconv_raw)
        if isinstance(upsample_block.relu, bool):
            if upsample_block.relu:
                conv2_up = F.leaky_relu(conv2_up, negative_slope=0.01)
        else:
            conv2_up = upsample_block.relu(conv2_up)
        x = conv2_up + cost_conv1
        for layer in self.post.out:
            if layer.__class__.__name__ == "FeatureAtt":
                x = layer(x, left_08)
            else:
                x = layer(x)
        return x


def _apply_basic_conv_bn_relu_from_raw(block, raw: torch.Tensor) -> torch.Tensor:
    x = block.bn(raw)
    if isinstance(block.relu, bool):
        if block.relu:
            x = F.leaky_relu(x, negative_slope=0.01)
    else:
        x = block.relu(x)
    return x


class CostAggPost8Low(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.conv1_up = model.cost_agg.conv1_up
        self.post = model.cost_agg.post8_to_4

    def forward(self, post8_deconv_raw: torch.Tensor):
        conv = _apply_basic_conv_bn_relu_from_raw(self.conv1_up, post8_deconv_raw)
        x = self.post.upsample[0](conv)
        x = self.post.upsample[1](x)
        return x


class CostAggPost8Out(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.post = model.cost_agg.post8_to_4

    def forward(self, post8_upsampled: torch.Tensor, stem_volume: torch.Tensor):
        x = post8_upsampled + stem_volume
        for layer in self.post.out:
            x = layer(x)
        return x


class ClassifierLogits(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.classifier = model.classifier

    def forward(self, regularized_volume: torch.Tensor):
        return self.classifier(regularized_volume).squeeze(1)


class RefinementUpdateStep(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.update_block = model.update_block
        self.spx_2_gru = model.spx_2_gru
        self.spx_gru = model.spx_gru

    def forward(
        self,
        net_list: torch.Tensor,
        inp_list: torch.Tensor,
        geo_feat: torch.Tensor,
        disp: torch.Tensor,
        att: torch.Tensor,
        stem_2x: torch.Tensor,
    ):
        motion_features = self.update_block.encoder(disp, geo_feat)
        motion_features = torch.cat([inp_list, motion_features], dim=1)
        next_net = self.update_block.gru04(att, net_list, motion_features)
        delta_disp = self.update_block.disp_head(next_net)
        mask_feat_4 = 0.25 * self.update_block.mask(next_net)
        xspx = self.spx_2_gru(mask_feat_4, stem_2x)
        up_weights = F.softmax(self.spx_gru(xspx), dim=1)
        return next_net, mask_feat_4, delta_disp, up_weights
