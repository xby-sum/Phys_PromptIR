import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicalPriorEstimator(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(PhysicalPriorEstimator, self).__init__()
        
        # Low-frequency scattering branch (T and A)
        self.scatter_conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.scatter_conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, 2, 1)
        self.scatter_conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, 2, 1)
        self.scatter_conv4 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, 1, 1)
        
        self.t_head = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        
        self.a_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_channels * 4, base_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 1),
            nn.Sigmoid()
        )
        
        # High-frequency occlusion branch (S)
        self.occlusion_conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.occlusion_conv2 = nn.Conv2d(base_channels, base_channels * 2, 3, 1, 1)
        self.occlusion_res1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1)
        )
        self.occlusion_res2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, 1, 1)
        )
        self.s_head = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 1, 1, 0),
            nn.Sigmoid()
        )
        
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        # Scattering branch
        f_scatter = F.relu(self.scatter_conv1(x))
        f_scatter = F.relu(self.scatter_conv2(f_scatter))
        f_scatter = F.relu(self.scatter_conv3(f_scatter))
        f_scatter = F.relu(self.scatter_conv4(f_scatter))
        
        T = self.t_head(f_scatter)
        T = self.upsample(T)
        A = self.a_head(f_scatter)
        
        # Occlusion branch
        f_occ = F.relu(self.occlusion_conv1(x))
        f_occ = F.relu(self.occlusion_conv2(f_occ))
        f_occ = f_occ + self.occlusion_res1(f_occ)
        f_occ = f_occ + self.occlusion_res2(f_occ)
        S = self.s_head(f_occ)
        
        return T, A, S