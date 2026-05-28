import torch
import torch.nn as nn
from timm.models.edgenext import PositionalEncodingFourier
from timm.models.layers import DropPath, Mlp, create_conv2d


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        x = x - u
        s = x.pow(2).mean(1, keepdim=True)
        x = x * torch.rsqrt(s + self.eps)
        return x * self.weight + self.bias
        

class ConvBlockExport(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        kernel_size=7,
        stride=1,
        conv_bias=True,
        expand_ratio=4,
        ls_init_value=1e-6,
        drop_path=0.,
    ):
        super().__init__()
        dim_out = dim_out or dim
        self.shortcut_after_dw = stride > 1 or dim != dim_out

        self.conv_dw = create_conv2d(
            dim, dim_out, kernel_size=kernel_size, stride=stride, depthwise=True, bias=conv_bias)
        self.norm = LayerNorm2d(dim_out)
        self.mlp = Mlp(dim_out, int(expand_ratio * dim_out), use_conv=True)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(1, dim_out, 1, 1)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.shortcut_after_dw:
            shortcut = x

        x = self.norm(x)
        x = self.mlp(x)
        
        if self.gamma is not None:
            x = x * self.gamma
            
        x = shortcut + self.drop_path(x)
        return x

    @classmethod
    def from_original(cls, orig: nn.Module) -> nn.Module:
        dim = orig.conv_dw.in_channels
        dim_out = orig.conv_dw.out_channels
        print(f"Dim out: {dim_out}")
        kernel_size = orig.conv_dw.kernel_size
        stride = orig.conv_dw.stride[0]
        conv_bias = orig.conv_dw.bias is not None
        expand_ratio = orig.mlp.fc1.out_features // orig.mlp.fc1.in_features
        ls_init_value = 1e-6 if orig.gamma is not None else 0.0
        drop_path = orig.drop_path.drop_prob if isinstance(orig.drop_path, DropPath) else 0.0

        layer = cls(
            dim=dim,
            dim_out=dim_out,
            kernel_size=kernel_size,
            stride=stride,
            conv_bias=conv_bias,
            expand_ratio=expand_ratio,
            ls_init_value=ls_init_value,
            drop_path=drop_path,
        )
        layer.conv_dw.load_state_dict(orig.conv_dw.state_dict())

        with torch.no_grad():
            layer.norm.weight.copy_(orig.norm.weight[None, :, None, None])
            layer.norm.bias.copy_(orig.norm.bias[None, :, None, None])

            layer.mlp.fc1.weight.copy_(orig.mlp.fc1.weight[:, :, None, None])
            layer.mlp.fc1.bias.copy_(orig.mlp.fc1.bias)

            layer.mlp.fc2.weight.copy_(orig.mlp.fc2.weight[:, :, None, None])
            layer.mlp.fc2.bias.copy_(orig.mlp.fc2.bias)

            if layer.gamma is not None:
                assert orig.gamma is not None
                layer.gamma.copy_(orig.gamma[None, :, None, None])

        return layer


class SplitTransposeBlockExport(nn.Module):
    def __init__(
        self,
        dim,
        width,
        num_scales=1,
        num_heads=8,
        expand_ratio=4,
        use_pos_emb=True,
        conv_bias=True,
        qkv_bias=True,
        ls_init_value=1e-6,
        act_layer=nn.GELU,
        drop_path=0.,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()

        self.width = width
        self.num_scales = num_scales # Not the same as goes to the __init__ of original module

        convs = []
        for i in range(self.num_scales):
            convs.append(create_conv2d(width, width, kernel_size=3, depthwise=True, bias=conv_bias))
        self.convs = nn.ModuleList(convs)

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)
        self.norm_xca = LayerNorm2d(dim, eps=1e-6)
        self.gamma_xca = nn.Parameter(ls_init_value * torch.ones(1, dim, 1, 1)) if ls_init_value > 0 else None
        self.xca = CrossCovarianceAttnExport(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)

        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(expand_ratio * dim), act_layer=act_layer, use_conv=True)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(1, dim, 1, 1)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        spx = x.chunk(len(self.convs) + 1, dim=1)
        spo = []
        sp = spx[0]
        for i, conv in enumerate(self.convs):
            if i > 0:
                sp = sp + spx[i]
            sp = conv(sp)
            spo.append(sp)
        spo.append(spx[-1])
        x = torch.cat(spo, 1)

        if self.pos_embd is not None:
            B, _, H, W = x.shape
            x = x + self.pos_embd((B, H, W))

        xca_shortcut = x
        x = self.xca(self.norm_xca(x))
        if self.gamma_xca is not None:
            x = self.gamma_xca * x
        x = xca_shortcut + self.drop_path(x)

        # Inverted Bottleneck
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = shortcut + self.drop_path(x)
        return x

    @classmethod
    def from_original(cls, orig: nn.Module) -> nn.Module:
        dim = orig.mlp.fc1.in_features
        width = orig.width
        num_scales = orig.num_scales
        num_heads = orig.xca.num_heads
        expand_ratio = orig.mlp.fc1.out_features // orig.mlp.fc1.in_features
        use_pos_emb = orig.pos_embd is not None
        conv_bias = orig.convs[0].bias is not None
        qkv_bias = orig.xca.qkv.bias is not None
        ls_init_value = 1e-6 if orig.gamma is not None else 0.0
        drop_path = orig.drop_path.drop_prob if isinstance(orig.drop_path, DropPath) else 0.0
        attn_drop = orig.xca.attn_drop.p
        proj_drop = orig.xca.proj_drop.p
        layer = cls(
            dim=dim,
            width=width,
            num_scales=num_scales,
            num_heads=num_heads,
            expand_ratio=expand_ratio,
            use_pos_emb=use_pos_emb,
            conv_bias=conv_bias,
            qkv_bias=qkv_bias,
            ls_init_value=ls_init_value,
            drop_path=drop_path,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        layer.convs.load_state_dict(orig.convs.state_dict())
        if layer.pos_embd is not None:
            layer.pos_embd.load_state_dict(orig.pos_embd.state_dict())

        with torch.no_grad():
            layer.norm.weight.copy_(orig.norm.weight[None, :, None, None])
            layer.norm.bias.copy_(orig.norm.bias[None, :, None, None])

            layer.norm_xca.weight.copy_(orig.norm_xca.weight[None, :, None, None])
            layer.norm_xca.bias.copy_(orig.norm_xca.bias[None, :, None, None])

            layer.xca.temperature.copy_(orig.xca.temperature[None, :, :, :])
            layer.xca.qkv.weight.copy_(orig.xca.qkv.weight[:, :, None, None])
            if layer.xca.qkv.bias is not None:
                layer.xca.qkv.bias.copy_(orig.xca.qkv.bias)

            layer.xca.proj.weight.copy_(orig.xca.proj.weight[:, :, None, None])
            layer.xca.proj.bias.copy_(orig.xca.proj.bias)

            if layer.gamma_xca is not None:
                assert orig.gamma_xca is not None
                layer.gamma_xca.copy_(orig.gamma_xca[None, :, None, None])
            if layer.gamma is not None:
                assert orig.gamma is not None
                layer.gamma.copy_(orig.gamma[None, :, None, None])

            layer.mlp.fc1.weight.copy_(orig.mlp.fc1.weight[:, :, None, None])
            layer.mlp.fc1.bias.copy_(orig.mlp.fc1.bias)

            layer.mlp.fc2.weight.copy_(orig.mlp.fc2.weight[:, :, None, None])
            layer.mlp.fc2.bias.copy_(orig.mlp.fc2.bias)
        return layer


def ane_normalize(t, eps=1e-12):
    # t: (BH, d, 1, N)
    square_sum = torch.sum(t * t, dim=-1, keepdim=True)
    return t * torch.rsqrt(square_sum + eps)


class CrossCovarianceAttnExport(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # Temperature is now shaped for batch-folded broadcasting
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        # Use 1x1 Convs instead of Linear
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # Input x is (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W

        # 1. Generate QKV and reshape spatial to 1D
        # (B, 3*C, H, W) -> (B, 3*C, 1, N)
        qkv = self.qkv(x).reshape(B, 3 * C, 1, N)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # 2. Batch-Head Folding: Fold Heads into Batch to stay 4D
        # (B, C, 1, N) -> (B * num_heads, head_dim, 1, N)
        q = q.reshape(B * self.num_heads, self.head_dim, 1, N)
        k = k.reshape(B * self.num_heads, self.head_dim, 1, N)
        v = v.reshape(B * self.num_heads, self.head_dim, 1, N)

        # 3. L2 Normalization (ANE-friendly version)
        # Normalizing across the spatial dimension (N)
        q = ane_normalize(q)
        k = ane_normalize(k)

        # 4. Cross-Covariance Attention: (d x N) @ (N x d) -> (d x d)
        attn = torch.einsum('bcin,bdin->bcid', q, k)

        # Apply temperature
        t_scaled = self.temperature.repeat(B, 1, 1, 1).view(B * self.num_heads, 1, 1, 1)
        attn = attn * t_scaled

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 5. Apply attention to values: (d x d) @ (d x N) -> (d x N)
        # attn: (BH, d, 1, d), v: (BH, d, 1, N)
        x = torch.einsum('bcid,bdin->bcin', attn, v)

        # 6. Unfold and Reshape back to NCHW
        x = x.reshape(B, C, 1, N).reshape(B, C, H, W)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}
