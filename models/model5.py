import numbers

import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer
from timm.models.layers import DropPath
import torch.nn.functional as F
from .utils import Compute_z


def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)

    return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class LKABlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, linear=linear)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                               * self.mlp(self.norm2(x)))

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.PixelUnshuffle(2),
                                  nn.Conv2d(n_feat * 2 * 2, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        _, _, h, w = x.shape
        if h % 2 != 0:
            x = F.pad(x, [0, 0, 1, 0])
        if w % 2 != 0:
            x = F.pad(x, [1, 0, 0, 0])
        return self.body(x)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * 3)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x


# Frequency-Domain Pixel Attention
class FDPA(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1)

        self.conv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x2_fft = torch.fft.fft2(x2)

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1))
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


class HybridDomainAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, bias=False)

        # Frequency-Domain Pixel Attention
        self.fpa = FDPA(dim)

        # Spatial-Domain Channel Attention
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # frequency pixel attention
        out = self.fpa(x)

        # spatial channel attention
        s_attn = self.conv(self.pool(self.norm1(out)))
        out = s_attn * out

        out = x + out
        out = out + self.ffn(self.norm2(out))

        return out


# Composite Shape Convolution
class CSC_Block(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        # Horizontal Strip Convolution
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        # Vertical Strip Convolution
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        # Square Kernel Convolution
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

    def forward(self, x):
        out = self.in_conv(x)

        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out)
        out = self.act(out)
        return self.out_conv(out)


import torch
import torch.nn as nn
from mamba_ssm import Mamba

class PVMLayer_NoSplit(nn.Module):
    """
    单路 Mamba：不再对序列维做 4 等分，d_model = input_dim
    仅做 [B,C,H,W] ↔ [B,N,C] 的 token 化与 LN
    """
    def __init__(self, input_dim, output_dim=None, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,   # 直接等于通道数
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        # 若 in/out 相同，可省 proj；这里保留以增强灵活性
        self.proj = nn.Linear(input_dim, self.output_dim) if self.output_dim != input_dim else None

    def forward(self, x):
        ori_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        B, C = x.shape[:2]
        assert C == self.input_dim, f"PVMLayer_NoSplit: expect C={self.input_dim}, got {C}"
        H, W = x.shape[2], x.shape[3]
        N = H * W

        x_seq = x.reshape(B, C, N).transpose(-1, -2)     # [B,N,C]
        x_seq = self.norm(x_seq)
        y_seq = self.mamba(x_seq)                        # [B,N,C]
        if self.proj is not None:
            y_seq = self.proj(y_seq)
        y = y_seq.transpose(-1, -2).reshape(B, self.output_dim, H, W)
        return y.to(ori_dtype)


class MambaBlock2D_NoSplit(nn.Module):
    """
    2D Mamba 包装（不四等分）：保持 [B,C,H,W] 接口，内置 BN+GELU+1x1 融合与残差
    不再要求 C 是 4 的倍数
    """
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.core = PVMLayer_NoSplit(channels, channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.bn   = nn.BatchNorm2d(channels)
        self.act  = nn.GELU()
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.core(x)
        y = self.fuse(self.act(self.bn(y)))
        return x + y
# ---------- Haar DWT/IWT ----------
def dwt_haar(x):
    # x: [B,C,H,W]
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    LL = x1 + x2 + x3 + x4
    HL = -x1 - x2 + x3 + x4
    LH = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4
    return LL, HL, LH, HH        # 各 [B,C,H/2,W/2]

def iwt_haar(LL, HL, LH, HH):
    # 各 [B,C,H/2,W/2] -> [B,C,H,W]
    B, C, Hh, Wh = LL.size()
    H, W = Hh*2, Wh*2
    x1 = (LL - HL - LH + HH) / 2
    x2 = (LL - HL + LH - HH) / 2
    x3 = (LL + HL - LH - HH) / 2
    x4 = (LL + HL + LH + HH) / 2

    out = torch.zeros(B, C, H, W, device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, 0::2] = x1
    out[:, :, 1::2, 0::2] = x2
    out[:, :, 0::2, 1::2] = x3
    out[:, :, 1::2, 1::2] = x4
    return out


class WaveletSubbandMamba2D(nn.Module):
    """
    全小波 → 四子带各自单路 Mamba → IWT 重建（通道数不变）
    可直接替换原 MambaBlock/LKABlock：输入输出形状一致 [B,C,H,W]
    """
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 use_wavelet_attn: bool = True, reduction: int = 16):
        super().__init__()
        self.m_ll = MambaBlock2D_NoSplit(channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.m_hl = MambaBlock2D_NoSplit(channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.m_lh = MambaBlock2D_NoSplit(channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.m_hh = MambaBlock2D_NoSplit(channels, d_state=d_state, d_conv=d_conv, expand=expand)

        self.use_wavelet_attn = use_wavelet_attn
        if use_wavelet_attn:
            # 简易小波域注意力（可选）
            self.sa = nn.Sequential(
                nn.Conv2d(4*channels, 1, kernel_size=5, padding=2, bias=False),
                nn.Sigmoid()
            )
            self.ca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(4*channels, 4*channels//reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(4*channels//reduction, 4*channels, 1, bias=False),
                nn.Sigmoid()
            )
            self.merge = nn.Conv2d(4*channels*2, 4*channels, 1, bias=False)  # concat(sa,ca) 后压回 4C

        # 图像域融合（可轻量）
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.fuse = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        # 全小波
        LL, HL, LH, HH = dwt_haar(x)
        # 子带级单路 Mamba
        LLp = self.m_ll(LL)
        HLp = self.m_hl(HL)
        LHp = self.m_lh(LH)
        HHp = self.m_hh(HH)

        if self.use_wavelet_attn:
            wave = torch.cat([LLp, HLp, LHp, HHp], dim=1)  # [B,4C,H/2,W/2]
            sa = self.sa(wave) * wave
            ca = self.ca(wave) * wave
            wave = self.merge(torch.cat([sa, ca], dim=1)) + wave
            # 再切回 4 份
            C = LLp.size(1)
            LLp, HLp, LHp, HHp = torch.chunk(wave, 4, dim=1)

        # 逆小波重建 + 残差样式融合
        y = iwt_haar(LLp, HLp, LHp, HHp)          # [B,C,H,W]
        y = self.fuse(self.act(self.bn(y)))
        return x + y


class UIR_PolyKernel(nn.Module):
    """
    在原 UIR_PolyKernel 上引入 CVAE + PAdaIN（自适应实例归一化）的概率不确定性建模：
      • 训练阶段：构建先验分支 p(z|x) 与后验分支 q(z|x,y)，在解码浅层融合处提取特征 -> Compute_z_pr/po 回归两组高斯 (u,s)
                   由后验分布重参数化采样 (u,s)，经 1×1 Conv 升维后对先验特征做 PAdaIN：IN(pr_feat) * |s| + u
      • 测试阶段：仅用先验 Compute_z_pr 输出 (mu,sigma)，取 MP 策略 (mu + sigma * 0) 注入 PAdaIN 得到确定增强结果
    """
    def __init__(self, in_channels=3, out_channels=3, dim=36, bias=False,
                 z_dim=20):  # z_dim：潜变量维度
        super(UIR_PolyKernel, self).__init__()

        # ---------------- 原模型先验主干（prior, 输入为 x） ----------------
        self.input_embed = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.encoder_level1 = HybridDomainAttention(dim)

        self.down1_2 = Downsample(dim)
        #self.encoder_level2 = LKABlock(int(dim * 2 ** 1))
        self.encoder_level2 = WaveletSubbandMamba2D(int(dim * 2 ** 1))
        
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        #self.encoder_level3 = LKABlock(int(dim * 2 ** 2))
        self.encoder_level3 = WaveletSubbandMamba2D(int(dim * 2 ** 2))

        self.bottleneck = CSC_Block(int(dim * 2 ** 2))

        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        #self.decoder_level3 = LKABlock(int(dim * 2 ** 2))
        self.decoder_level3 = WaveletSubbandMamba2D(int(dim * 2 ** 2))
        
        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        #self.decoder_level2 = LKABlock(int(dim * 2 ** 1))
        self.decoder_level2 = WaveletSubbandMamba2D(int(dim * 2 ** 1))
        
        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.reduce_chan_level1 = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = HybridDomainAttention(int(dim))

        self.final_conv = nn.Conv2d(dim, out_channels, kernel_size=1)
        self.norm = nn.Sigmoid()

        # ---------------- 概率建模新增：后验分支（posterior, 输入为 cat(x, y)） ----------------
        # 说明：仅在训练阶段使用，使得 q(z|x,y) 与 p(z|x) 的 KL 可计算；结构尽量与先验浅层对齐，保证提取到同尺度语义
        self.input_embed_po = nn.Conv2d(in_channels * 2, dim, kernel_size=1)
        self.encoder_level1_po = HybridDomainAttention(dim)
        self.down1_2_po = Downsample(dim)
        #self.encoder_level2_po = LKABlock(int(dim * 2 ** 1))
        self.encoder_level2_po = WaveletSubbandMamba2D(int(dim * 2 ** 1))   # ← 替换
        
        self.down2_3_po = Downsample(int(dim * 2 ** 1))
        #self.encoder_level3_po = LKABlock(int(dim * 2 ** 2))
        self.encoder_level3_po = WaveletSubbandMamba2D(int(dim * 2 ** 2))   # ← 替换
        
        self.bottleneck_po = CSC_Block(int(dim * 2 ** 2))

        # 为了与先验解码浅层融合点对齐，后验也需降通道到 dim
        self.reduce_chan_level3_po = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.up3_2_po = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2_po = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.up2_1_po = Upsample(int(dim * 2 ** 1))
        self.reduce_chan_level1_po = nn.Conv2d(int(dim * 2), int(dim), kernel_size=1, bias=bias)

        # ---------------- 概率建模新增：z 映射到通道（PAdaIN 所需的 a、b） ----------------
        self.z_dim = z_dim
        # 这里的 Compute_z_* 请用你工程中的实现（接口见上方说明）
        #self.compute_z_pr = Compute_z(z_dim)
        #self.compute_z_po = Compute_z(z_dim)
        self.compute_z_pr = Compute_z(latent_dim=z_dim, in_channels=dim)
        self.compute_z_po = Compute_z(latent_dim=z_dim, in_channels=dim)
        
        # z->[C]（对每个通道生成 a、b），注意此处 C=dim（在最浅层 reduce 后的通道数）
        self.conv_u = nn.Conv2d(z_dim, dim, kernel_size=1, padding=0)
        self.conv_s = nn.Conv2d(z_dim, dim, kernel_size=1, padding=0)

        # PAdaIN 使用的 InstanceNorm（通道对齐到 dim）
        self.insnorm = nn.InstanceNorm2d(dim, affine=False)

    # ------------------------ 辅助：先验/后验路径的“特征提取到浅层融合点” ------------------------
    def _forward_prior_until_shallow(self, x):
        """
        先验路径：返回
          out_enc_level1, out_enc_level2, out_enc_level3, shallow_pr_feat
        其中 shallow_pr_feat 即 reduce_chan_level1 之后、decoder_level1 之前的先验浅层特征（C=dim）
        """
        inp = self.input_embed(x)
        out_enc_level1 = self.encoder_level1(inp)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        latent = self.bottleneck(out_enc_level3)

        inp_dec_level3 = cat(latent, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = cat(inp_dec_level2, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = cat(inp_dec_level1, out_enc_level1)
        shallow_pr_feat = self.reduce_chan_level1(inp_dec_level1)  # C=dim

        return out_enc_level1, out_enc_level2, out_enc_level3, shallow_pr_feat

    def _forward_posterior_until_shallow(self, x_and_y):
        """
        后验路径（仅训练用）：与先验形状对齐，输出 shallow_po_feat（C=dim）
        """
        inp = self.input_embed_po(x_and_y)
        out_enc_level1 = self.encoder_level1_po(inp)

        inp_enc_level2 = self.down1_2_po(out_enc_level1)
        out_enc_level2 = self.encoder_level2_po(inp_enc_level2)

        inp_enc_level3 = self.down2_3_po(out_enc_level2)
        out_enc_level3 = self.encoder_level3_po(inp_enc_level3)

        latent = self.bottleneck_po(out_enc_level3)

        inp_dec_level3 = cat(latent, out_enc_level3)
        inp_dec_level3 = self.reduce_chan_level3_po(inp_dec_level3)
        out_dec_level3 = self.up3_2_po(inp_dec_level3)  # 轻量：后验不再堆 block，仅对齐分辨率

        inp_dec_level2 = cat(out_dec_level3, out_enc_level2)
        inp_dec_level2 = self.reduce_chan_level2_po(inp_dec_level2)
        out_dec_level2 = self.up2_1_po(inp_dec_level2)

        inp_dec_level1 = cat(out_dec_level2, out_enc_level1)
        shallow_po_feat = self.reduce_chan_level1_po(inp_dec_level1)  # C=dim
        return shallow_po_feat

    # ------------------------ 概率注入（PAdaIN）：IN(x) * |s| + u ------------------------
    def _p_adain(self, feat, u_vec, s_vec):
        """
        feat: 先验浅层特征 [B, C=dim, H, W]
        u_vec/s_vec: 由 z 变换得到的 [B, C=dim, 1, 1]
        """
        feat_norm = self.insnorm(feat)
        return feat_norm * torch.abs(s_vec) + u_vec

    # ------------------------ 前向：支持训练/测试两阶段 ------------------------
    def forward(self, x, target=None, training: bool = True):
        """
        训练阶段：需要提供 target（增强参考），输出 (out, pr_u_dist, pr_s_dist, po_u_dist, po_s_dist)
        测试阶段：不需要 target，输出 out（采用 MP 策略）
        """
        # 先验：取到浅层融合前的特征（C=dim）
        out_enc_level1, out_enc_level2, out_enc_level3, pr_shallow = self._forward_prior_until_shallow(x)

        if training:
            assert target is not None, "training=True 时必须提供 target（用于后验分支 q(z|x,y)）"
            # ---- 后验分支（仅训练） ----
            po_input = torch.cat([x, target], dim=1)  # [B, 6, H, W]
            po_shallow = self._forward_posterior_until_shallow(po_input)  # [B, dim, H, W]

            # ---- 由浅层特征回归两组高斯分布 (u/s)，并从后验采样 ----
            pr_u_dist, pr_s_dist, _, _, _, _ = self.compute_z_pr(pr_shallow)
            po_u_dist, po_s_dist, _, _, _, _ = self.compute_z_po(po_shallow)

            # 重参数化采样（后验）
            po_latent_u = po_u_dist.rsample()  # [B, z_dim]
            po_latent_s = po_s_dist.rsample()  # [B, z_dim]
            # 升维到 [B, z_dim, 1, 1]
            po_latent_u = po_latent_u.unsqueeze(-1).unsqueeze(-1)
            po_latent_s = po_latent_s.unsqueeze(-1).unsqueeze(-1)
            # z -> [B, dim, 1, 1]
            po_u = self.conv_u(po_latent_u)
            po_s = self.conv_s(po_latent_s)

            # ---- 概率注入（PAdaIN）到先验浅层特征 ----
            pr_shallow_mod = self._p_adain(pr_shallow, po_u, po_s)

            # ---- 进入原有最浅层解码块 ----
            out_dec_level1 = self.decoder_level1(pr_shallow_mod)
            out = self.norm(self.final_conv(out_dec_level1) + x)

            # 训练阶段返回分布，便于计算 KL(po||pr) 与重建/感知损失
            return out, pr_u_dist, pr_s_dist, po_u_dist, po_s_dist

        else:
            # ---- 测试阶段（MP）：仅用先验，取均值点（最大概率） ----
            pr_u_dist, pr_s_dist, u_mu, s_mu, u_sigma, s_sigma = self.compute_z_pr(pr_shallow)

            pr_latent_u = u_mu + u_sigma * 0  # MP：取均值点
            pr_latent_s = s_mu + s_sigma * 0
            pr_latent_u = pr_latent_u.unsqueeze(-1).unsqueeze(-1)
            pr_latent_s = pr_latent_s.unsqueeze(-1).unsqueeze(-1)

            pr_u = self.conv_u(pr_latent_u)  # [B, dim, 1, 1]
            pr_s = self.conv_s(pr_latent_s)

            pr_shallow_mod = self._p_adain(pr_shallow, pr_u, pr_s)

            out_dec_level1 = self.decoder_level1(pr_shallow_mod)
            out = self.norm(self.final_conv(out_dec_level1) + x)
            return out
