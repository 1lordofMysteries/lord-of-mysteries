import numbers

import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer
from timm.models.layers import DropPath
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import math


class Compute_z(nn.Module):
    """
    概率参数回归模块（支持任意输入通道数）：
      • 输入: x ∈ [B, C, H, W]
      • 先做全局统计（均值/标准差），得到 [B, C, 1, 1]
      • 再用 1×1 卷积回归 (μ, logσ)，维度为 2*latent_dim
      • 用 softplus(logσ) 得到 σ，数值更稳定
      • 输出两条一维高斯分布（u_dist/s_dist）以及对应 μ/σ，便于 KL 与 MP/MC 使用
    """
    def __init__(self, latent_dim: int, in_channels: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        # 注意：不再写死 128 通道，以 in_channels 自适应
        self.u_head = nn.Conv2d(in_channels, 2 * latent_dim, kernel_size=1, padding=0, bias=True)
        self.s_head = nn.Conv2d(in_channels, 2 * latent_dim, kernel_size=1, padding=0, bias=True)
        self.eps = 1e-6  # 防止数值为 0

    def forward(self, x: torch.Tensor):
        """
        返回：
          u_dist, s_dist: Independent(Normal(...), 1)
          u_mu, s_mu: [B, latent_dim]
          u_sigma, s_sigma: [B, latent_dim]（>=0）
        """
        # -------- 全局统计（一次性在 H,W 两个维度上）--------
        # 全局均值：[B, C, 1, 1]
        u_encoding = x.mean(dim=(2, 3), keepdim=True)
        # 全局标准差（无偏项关掉，数值更稳）：[B, C, 1, 1]
        s_encoding = x.std(dim=(2, 3), keepdim=True, unbiased=False)

        # -------- 回归 (μ, logσ) 并拆分 --------
        u_mu_logsig = self.u_head(u_encoding).squeeze(-1).squeeze(-1)  # [B, 2L]
        s_mu_logsig = self.s_head(s_encoding).squeeze(-1).squeeze(-1)  # [B, 2L]

        u_mu, u_logsig = torch.split(u_mu_logsig, self.latent_dim, dim=1)  # [B, L], [B, L]
        s_mu, s_logsig = torch.split(s_mu_logsig, self.latent_dim, dim=1)  # [B, L], [B, L]

        # -------- 用 softplus 保证 σ>0，且数值更稳定 --------
        u_sigma = F.softplus(u_logsig) + self.eps
        s_sigma = F.softplus(s_logsig) + self.eps

        # -------- 构造一维高斯分布，并用 Independent 包一维事件 --------
        u_dist = Independent(Normal(loc=u_mu, scale=u_sigma), 1)
        s_dist = Independent(Normal(loc=s_mu, scale=s_sigma), 1)

        return u_dist, s_dist, u_mu, s_mu, u_sigma, s_sigma



def cat(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    x = torch.cat([x2, x1], dim=1)

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

# 轻量 ECA 通道注意：几乎零参数
class ECA(nn.Module):
    def __init__(self, channels: int, k_size: int = None):
        super().__init__()
        if k_size is None:
            t = int(abs((math.log2(channels) + 1) / 2))
            k_size = t if t % 2 else t + 1
            k_size = max(3, min(k_size, 9))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act  = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)                            # [B,C,1,1]
        y = y.squeeze(-1).transpose(1, 2)          # [B,1,C]
        y = self.conv(y)                           # [B,1,C]
        y = y.transpose(1, 2).unsqueeze(-1)        # [B,C,1,1]
        y = self.act(y)
        return x * y


# 多扩张-局部注意（共享Q/K/V，按dilation展开邻域）
class MDLA(nn.Module):
    def __init__(self, dim, num_heads=4, kernel_size=3, dilations=(1, 2),
                 qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilations = list(dilations)

        # 共享 Q/K/V（一次性计算，减少算力）
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)

        # 为不同 dilation 构造 Unfold
        self.unfolds = nn.ModuleList([
            nn.Unfold(kernel_size, dilation=d, padding=d * (kernel_size - 1) // 2, stride=1)
            for d in self.dilations
        ])

        # 多扩张输出加权：learnable softmax 权重（标量级）
        self.mix_logits = nn.Parameter(torch.zeros(len(self.dilations)))

        # 输出投影（1x1 Conv 比 Linear 更自然地回到 [B,C,H,W]）
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  # x: [B,C,H,W]
        B, C, H, W = x.shape

        # 共享 Q/K/V
        qkv = self.qkv(x)                                    # [B,3C,H,W]
        q, k, v = qkv.chunk(3, dim=1)                        # [B,C,H,W] * 3

        # 头分组
        h, d = self.num_heads, self.head_dim
        # Q: [B,h,HW,1,d]
        q = q.view(B, h, d, H, W).permute(0, 1, 3, 4, 2).reshape(B, h, H * W, 1, d)

        outs = []
        for uf in self.unfolds:
            # K/V 展开邻域： [B, C*k^2, HW]
            k_nei = uf(k)  # [B, C*K2, N]
            v_nei = uf(v)  # [B, C*K2, N]
            K2 = self.kernel_size * self.kernel_size

            # reshape 到多头：K/V -> [B,h,N,K2,d]
            k_nei = k_nei.view(B, h, d * K2, H * W).permute(0, 1, 3, 2) \
                        .reshape(B, h, H * W, K2, d)
            v_nei = v_nei.view(B, h, d * K2, H * W).permute(0, 1, 3, 2) \
                        .reshape(B, h, H * W, K2, d)

            # 局部注意：attn = softmax(q @ k / sqrt(d))  over K2
            attn = torch.matmul(q, k_nei.transpose(-1, -2)) * self.scale   # [B,h,N,1,K2]
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            # 聚合 V： [B,h,N,1,d] -> squeeze -> [B,h,N,d]
            out = torch.matmul(attn, v_nei).squeeze(-2)  # [B,h,N,d]
            outs.append(out)

        # 多扩张融合（标量 softmax 权重）
        mix_w = F.softmax(self.mix_logits, dim=0)  # [D]
        out = 0
        for i, o in enumerate(outs):
            out = out + mix_w[i] * o                # [B,h,N,d]

        # fold 回 [B,C,H,W]
        out = out.reshape(B, h, H, W, d).permute(0, 1, 4, 2, 3).reshape(B, C, H, W)

        # 输出线性映射
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# --------- 新版 HybridDomainAttention（类名/IO 不变） ---------
class HybridDomainAttention(nn.Module):
    """
    轻量级多扩张局部注意 + ECA 通道注意：
      - 局部注意只看 k×k 邻域，多扩张(dilations)覆盖尺度变化，计算量远小于全局注意
      - 共享 Q/K/V，Unfold 不同 dilation，标量 softmax 融合不同扩张输出
      - Pre-LN + 残差缩放，适合放在增强的头/尾，稳
      - 输出维度与输入一致 [B,C,H,W]
    """
    def __init__(self, dim, num_heads=4, kernel_size=3, dilations=(1, 2),
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)

        # 多扩张-局部注意（轻量）
        self.mdla = MDLA(dim, num_heads=num_heads, kernel_size=kernel_size,
                         dilations=dilations, qkv_bias=qkv_bias,
                         attn_drop=attn_drop, proj_drop=proj_drop)

        # 轻量通道注意
        self.chan_attn = ECA(dim)

        # 前馈（沿用你的 FeedForward）
        self.ffn = FeedForward(dim, bias=False)

        # 两处残差缩放（初始化很小 → 稳定注入）
        self.attn_scale = nn.Parameter(torch.tensor(1e-3))
        self.ffn_scale  = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x):  # x: [B,C,H,W]
        # Pre-LN
        y = self.norm1(x)

        # 多扩张局部注意
        y = self.mdla(y)

        # 轻量通道注意
        y = self.chan_attn(y)

        # 残差一（小尺度）
        x = x + self.attn_scale * y

        # FFN 残差二（Pre-LN）
        y = self.ffn(self.norm2(x))
        x = x + self.ffn_scale * y
        return x


# Composite Shape Convolution
import torch
import torch.nn as nn
import torch.nn.functional as F

# Composite Shape Convolution (Axial Dilated Pyramid + PPM + Gated Fusion)
class CSC_Block(nn.Module):
    """
    全新 CSC 变体（不含任意大核/条带核）：
      • Axial Dilated Pyramid：沿 H/W 轴用 (1x3)/(3x1) 深度卷积，dilations=[1,2,4,8,16]，指数扩张感受野
      • Pyramid Pooling (PPM)：自适应多尺度池化 {1,2,4}，上采样回原尺寸补充全局上下文
      • Gated Fusion：三个分支（H 轴、W 轴、PPM）逐分支通道门控，自适应融合
      • 保持入口/出口 1x1 与残差，I/O 与原版完全一致
    """
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

        # -------- 输入/输出投影（与原版保持一致） --------
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=False),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=False)

        # -------- 轴向空洞金字塔（均为 depth-wise，极轻）--------
        dilations = [1, 2, 4, 8, 16]

        def make_axial_dw_stack(kernel, axis='h'):
            layers = []
            for d in dilations:
                if axis == 'h':  # 水平方向：1x3
                    layers.append(nn.Conv2d(self.dim, self.dim, kernel_size=(1, kernel),
                                            padding=(0, d), dilation=(1, d),
                                            groups=self.dim, bias=False))
                else:            # 垂直方向：3x1
                    layers.append(nn.Conv2d(self.dim, self.dim, kernel_size=(kernel, 1),
                                            padding=(d, 0), dilation=(d, 1),
                                            groups=self.dim, bias=False))
            return nn.ModuleList(layers)

        self.h_stack = make_axial_dw_stack(kernel=3, axis='h')  # (1x3) dilated
        self.v_stack = make_axial_dw_stack(kernel=3, axis='v')  # (3x1) dilated

        # -------- Pyramid Pooling Module (PPM) --------
        # 3 个尺度：1（全局）、2、4
        self.ppm_scales = [1, 2, 4]
        self.ppm_reduces = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            ) for s in self.ppm_scales
        ])

        # -------- 逐分支门控（通道级）--------
        # 3 个分支：H 轴、V 轴、PPM
        self.num_branches = 3
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 对每个分支做独立的通道缩放（分组 1x1 保持互不干扰）
        self.gate = nn.Sequential(
            nn.Conv2d(self.dim * self.num_branches, self.dim * self.num_branches,
                      kernel_size=1, groups=self.num_branches, bias=False),
            nn.Sigmoid()
        )

        # -------- 分支融合 + 轻量精修 --------
        self.fuse = nn.Conv2d(self.dim * self.num_branches, self.dim, kernel_size=1, bias=False)
        self.refine_dw = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim, bias=False)
        self.refine_pw = nn.Conv2d(self.dim, self.dim, kernel_size=1, bias=False)

        self.act = nn.ReLU(inplace=True)

    def _axial_pyramid(self, x, stack):
        """
        x: [B,C,H,W], stack: ModuleList of depth-wise convs with different dilations
        逐层相加形成金字塔输出（残差样式，有利于稳定）
        """
        y = 0
        for conv in stack:
            y = y + conv(x)
        return y

    def _ppm(self, x):
        """
        Pyramid Pooling：多尺度自适应平均池化 → 1x1 降维 → 上采样回原尺寸 → 求和
        """
        B, C, H, W = x.shape
        outs = []
        for block, s in zip(self.ppm_reduces, self.ppm_scales):
            z = block(x)                         # [B,C,s,s]
            z = F.interpolate(z, size=(H, W), mode='bilinear', align_corners=True)
            outs.append(z)
        y = outs[0]
        for t in outs[1:]:
            y = y + t
        return y

    def forward(self, x):
        """
        x: [B, C, H, W] —— 与原版一致
        """
        B, C, H, W = x.shape

        # 入口 1×1
        stem = self.in_conv(x)  # [B,C,H,W]

        # 轴向空洞金字塔（水平/垂直）
        h_feat = self._axial_pyramid(stem, self.h_stack)   # [B,C,H,W]
        v_feat = self._axial_pyramid(stem, self.v_stack)   # [B,C,H,W]

        # 金字塔池化上下文
        p_feat = self._ppm(stem)                           # [B,C,H,W]

        # 拼接 3 个分支
        feat = torch.cat([h_feat, v_feat, p_feat], dim=1)  # [B, 3C, H, W]

        # 分支门控（通道级）
        alpha = self.gate(self.pool(feat))                 # [B, 3C, 1, 1]
        feat = feat * alpha

        # 融合 + 轻量精修
        y = self.fuse(feat)                                # [B, C, H, W]
        y = self.refine_pw(self.refine_dw(y))              # [B, C, H, W]

        # 残差 + 激活 + 出口 1×1
        out = self.act(x + y)
        return self.out_conv(out)


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from mamba_ssm import Mamba


# ===== 改进后的 PVMLayer（接口保持不变）=====
class PVMLayer(nn.Module):
    """
    改进点：
      • 2D Fourier Positional Encoding：在序列化后为每个 token 注入显式二维位置编码
      • Bi-Mamba：对同一序列做正向/反向两次 Mamba 建模并融合，获得更充分的长程依赖
      • Chunk-wise SE Gating：对四个分块的输出做轻量门控，提升跨块全局一致性与鲁棒性
      • 与原版保持：输入/输出形状、半精度兼容、最终 LayerNorm + Linear 投影
    """
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2,
                 posBands: int = 16):  # posBands: Fourier频带数（每轴）
        super().__init__()
        assert input_dim % 4 == 0, f"PVMLayer要求 input_dim % 4 == 0，但得到 {input_dim}"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.chunk_dim = input_dim // 4
        self.posBands = posBands

        # 归一化（对 token 维的特征做 LN）
        self.norm_in = nn.LayerNorm(input_dim)

        # 双向 Mamba（两个方向各一套核心）
        self.mamba_fwd = Mamba(
            d_model=self.chunk_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=self.chunk_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # 位置编码：把 2D Fourier 特征映射到 chunk 维度并相加
        # 2 * posBands（sin+cos，每轴 posBands） * 2轴 = 4 * posBands
        self.pos_proj = nn.Linear(4 * posBands, self.chunk_dim)

        # 跨分块门控（SE风格）：对 4 个分块的输出进行逐通道门控
        # 输入是 [B, 4, chunk_dim] 的全局平均，输出同形状门控权重
        hidden = max(16, self.chunk_dim // 4)
        self.chunk_gate = nn.Sequential(
            nn.Linear(self.chunk_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.chunk_dim),
            nn.Sigmoid()
        )

        # 残差安全网
        self.skip_scale = nn.Parameter(torch.ones(1))

        # 输出层：规范化 + 线性投影
        self.norm_out = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    @torch.no_grad()
    def _build_fourier_pos(self, H: int, W: int, device, dtype):
        """
        构建二维傅里叶位置编码（固定频带，训练中不更新）。
        返回形状：[1, H*W, 4*posBands]，四块分别是：
        sin(2π f_x x), cos(2π f_x x), sin(2π f_y y), cos(2π f_y y)
        """
        # 归一化坐标到 [0,1]
        ys = torch.linspace(0, 1, steps=H, device=device, dtype=dtype)
        xs = torch.linspace(0, 1, steps=W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

        # 准备频带（指数或线性，这里用指数 1,2,4,...）
        freqs = 2.0 ** torch.arange(self.posBands, device=device, dtype=dtype)  # [P]

        # [H, W, P]
        x_feats = (2.0 * torch.pi) * xx[..., None] * freqs  # 2π f x
        y_feats = (2.0 * torch.pi) * yy[..., None] * freqs  # 2π f y

        # sin/cos 展开并拼接 -> [H, W, 4P]
        pos = torch.cat([torch.sin(x_feats), torch.cos(x_feats),
                         torch.sin(y_feats), torch.cos(y_feats)], dim=-1)
        pos = pos.view(1, H * W, 4 * self.posBands)  # [1, N, 4P]
        return pos

    def forward(self, x):
        """
        输入: x ∈ [B, C_in, H, W]
        输出: out ∈ [B, C_out, H, W]
        """
        ori_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C = x.shape[:2]
        assert C == self.input_dim, f"期望 C={self.input_dim}，但得到 {C}"
        H, W = x.shape[2], x.shape[3]
        N = H * W

        # 展平为序列并做输入LN
        x_seq = x.reshape(B, C, N).transpose(-1, -2)  # [B, N, C]
        x_norm = self.norm_in(x_seq)

        # 2D Fourier 位置编码（一次构建，整个 batch 共享）
        pos = self._build_fourier_pos(H, W, device=x.device, dtype=x_norm.dtype)  # [1, N, 4P]
        pos_emb = self.pos_proj(pos)  # [1, N, chunk_dim]

        # 四分块（沿通道维切分）
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)  # 各 [B, N, chunk_dim]

        # 为每个分块加入相同的 2D 位置编码
        x1 = x1 + pos_emb
        x2 = x2 + pos_emb
        x3 = x3 + pos_emb
        x4 = x4 + pos_emb

        # 双向 Mamba：正向 + 反向（通过反转序列维度）
        def bimamba(core_fwd: Mamba, core_bwd: Mamba, z: torch.Tensor):
            z_f = core_fwd(z)                          # 正向
            z_b = torch.flip(z, dims=[1])              # 反向序列
            z_b = core_bwd(z_b)                        # 反向建模
            z_b = torch.flip(z_b, dims=[1])            # 翻回原顺序
            return 0.5 * (z_f + z_b) + self.skip_scale * z  # 残差

        y1 = bimamba(self.mamba_fwd, self.mamba_bwd, x1)
        y2 = bimamba(self.mamba_fwd, self.mamba_bwd, x2)
        y3 = bimamba(self.mamba_fwd, self.mamba_bwd, x3)
        y4 = bimamba(self.mamba_fwd, self.mamba_bwd, x4)

        # 跨分块门控：按 token 全局平均后得到 [B, 4, chunk_dim] 的统计
        g1 = y1.mean(dim=1)  # [B, chunk_dim]
        g2 = y2.mean(dim=1)
        g3 = y3.mean(dim=1)
        g4 = y4.mean(dim=1)
        G = torch.stack([g1, g2, g3, g4], dim=1)   # [B, 4, chunk_dim]
        Wg = self.chunk_gate(G)                    # [B, 4, chunk_dim] in (0,1)

        # 施加门控（逐通道缩放）
        y1 = y1 * Wg[:, 0:1, :]
        y2 = y2 * Wg[:, 1:2, :]
        y3 = y3 * Wg[:, 2:3, :]
        y4 = y4 * Wg[:, 3:4, :]

        # 拼回通道
        y = torch.cat([y1, y2, y3, y4], dim=2)  # [B, N, C]

        # 输出规范化与线性映射
        y = self.norm_out(y)
        y = self.proj(y)                        # [B, N, C_out]

        # 还原空间形状
        out = y.transpose(-1, -2).reshape(B, self.output_dim, H, W)
        return out.to(ori_dtype)


# ===== 原有的自适配 2D Mamba 模块（无需改动）=====
class MambaBlock2D(nn.Module):
    """
    自适配 2D Mamba 模块（用于替换 LKABlock）：
      - 若通道 C 不是 4 的倍数，自动用 1×1 升/降到 nearest_4(C)，Mamba 后再映回 C
      - 内置残差安全网：out = x + drop_path(proj_out)
    """
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.C = channels

        # 计算最近的可被 4 整除的通道数（优先不降维）
        if channels % 4 == 0:
            c_proj = channels
        else:
            c_proj = ((channels + 3) // 4) * 4   # 向上取整到4的倍数

        self.need_proj = (c_proj != channels)
        if self.need_proj:
            self.in_proj  = nn.Conv2d(channels, c_proj, kernel_size=1, bias=False)
            self.out_proj = nn.Conv2d(c_proj, channels, kernel_size=1, bias=False)

        self.core = PVMLayer(input_dim=c_proj, output_dim=c_proj, d_state=d_state, d_conv=d_conv, expand=expand)

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        if self.need_proj:
            x_proj = self.in_proj(x)
        else:
            x_proj = x

        y = self.core(x_proj)          # [B, Cproj, H, W]

        if self.need_proj:
            y = self.out_proj(y)       # 回到 [B, C, H, W]

        y = self.fuse(self.act(self.bn(y)))
        out = identity + self.drop_path(y)
        return out

class AdaLINCore2d(nn.Module):
    def __init__(self, C, rho_init=0.9, eps=1e-5):
        super().__init__()
        self.rho = nn.Parameter(torch.full((1, C, 1, 1), rho_init))
        self.eps = eps
    def forward(self, x):
        # IN
        mu_in  = x.mean(dim=(2,3), keepdim=True)
        var_in = x.var(dim=(2,3), keepdim=True, unbiased=False)
        x_in   = (x - mu_in) / (var_in + self.eps).sqrt()
        # LN
        mu_ln  = x.mean(dim=(1,2,3), keepdim=True)
        var_ln = x.var(dim=(1,2,3), keepdim=True, unbiased=False)
        x_ln   = (x - mu_ln) / (var_ln + self.eps).sqrt()
        rho = torch.clamp(self.rho, 0.0, 1.0)
        return rho * x_in + (1 - rho) * x_ln

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
        #self.encoder_level2 = MambaBlock2D(int(dim * 2 ** 1))
        self.encoder_level2 = nn.Identity()

        self.down2_3 = Downsample(int(dim * 2 ** 1))
        #self.encoder_level3 = LKABlock(int(dim * 2 ** 2))
        #self.encoder_level3 = MambaBlock2D(int(dim * 2 ** 2))
        self.encoder_level3 = nn.Identity()

        self.bottleneck = CSC_Block(int(dim * 2 ** 2))

        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        #self.decoder_level3 = LKABlock(int(dim * 2 ** 2))
        #self.decoder_level3 = MambaBlock2D(int(dim * 2 ** 2))
        self.decoder_level3 = nn.Identity()
        
        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        #self.decoder_level2 = LKABlock(int(dim * 2 ** 1))
        #self.decoder_level2 = MambaBlock2D(int(dim * 2 ** 1))
        self.decoder_level2 = nn.Identity()
        
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
        #self.encoder_level2_po = MambaBlock2D(int(dim * 2 ** 1))   # ← 替换
        self.encoder_level2_po = nn.Identity()
        
        self.down2_3_po = Downsample(int(dim * 2 ** 1))
        #self.encoder_level3_po = LKABlock(int(dim * 2 ** 2))
        #self.encoder_level3_po = MambaBlock2D(int(dim * 2 ** 2))   # ← 替换
        self.encoder_level3_po = nn.Identity()
        
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
        #self.insnorm = nn.InstanceNorm2d(dim, affine=False)
        self.insnorm = AdaLINCore2d(dim, rho_init=0.9)

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

if __name__ == '__main__':
    from thop import profile, clever_format

    t = torch.randn(1, 3, 256, 256).cuda()
    model = UIR_PolyKernel().cuda()
    #macs, params = profile(model, inputs=(t,))
    macs, params = profile(model, inputs=(t, None, False))
    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)