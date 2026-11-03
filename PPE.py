import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalPriorEstimator(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, use_coral=False):
        super().__init__()
        self.use_coral = use_coral
        # 特征提取：两个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GELU()
        )
        # 输出头
        self.T_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.A_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels, 3, 1),
            nn.Sigmoid()
        )
        self.S_head = nn.Sequential(
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh()
        )
        # 全局池化（用于CORAL）
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 特征提取
        feat = self.conv1(x)
        feat = self.conv2(feat)
        # 物理参数估计
        T = self.T_head(feat)
        A = self.A_head(feat)
        S = self.S_head(feat)
        # 如果启用 CORAL，额外返回特征向量
        if self.use_coral:
            feat_pool = self.global_pool(feat)  # (B, C, 1, 1)
            feat_vec = feat_pool.flatten(1)     # (B, C)
            return T, A, S, feat_vec
        else:
            return T, A, S
