import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
import numpy as np
import torch.nn.functional as F

class SELayer(torch.nn.Module):
    def __init__(self, num_filter):
        super(SELayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_double = torch.nn.Sequential(
            nn.Conv2d(num_filter, num_filter // 16, 1, 1, 0, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(num_filter // 16, num_filter, 1, 1, 0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        mask = self.global_pool(x)
        mask = self.conv_double(mask)
        x = x * mask
        return x


class ResBlock(nn.Module):
    def __init__(self, num_filter):
        super(ResBlock, self).__init__()
        body = []
        for i in range(2):
            body.append(nn.ReflectionPad2d(1))
            body.append(nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=0))
            if i == 0:
                body.append(nn.LeakyReLU())
        body.append(SELayer(num_filter))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        x = res + x
        return x


class Up(nn.Module):
    def __init__(self):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


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
