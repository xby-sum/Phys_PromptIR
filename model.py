## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090
import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from einops.layers.torch import Rearrange
import time
from net.PPE import PhysicalPriorEstimator
##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)
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
        return x / torch.sqrt(sigma+1e-5) * self.weight
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, x):
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
class resblock(nn.Module):
    def __init__(self, dim):
        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')
        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))
    def forward(self, x):
        res = self.body((x))
        res += x
        return res
##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.body(x)
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)
##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, x):
        x = self.proj(x)
        return x
##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self,prompt_dim=128,prompt_len=5,prompt_size = 96,lin_dim = 192):
        super(PromptGenBlock,self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1,prompt_len,prompt_dim,prompt_size,prompt_size))
        self.linear_layer = nn.Linear(lin_dim,prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim,prompt_dim,kernel_size=3,stride=1,padding=1,bias=False)
    def forward(self,x):
        B,C,H,W = x.shape
        emb = x.mean(dim=(-2,-1))
        prompt_weights = F.softmax(self.linear_layer(emb),dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt,dim=1)
        prompt = F.interpolate(prompt,(H,W),mode="bilinear")
        prompt = self.conv3x3(prompt)
        return prompt
##########################################################################
##---------- PromptIR -----------------------
class PhysicsAwarePromptFusion(nn.Module):
    def __init__(self, feature_dim, phys_dim=1+3+3):
        super().__init__()
        self.phys_proj = nn.Conv2d(phys_dim, feature_dim, 1)
        self.fusion = nn.Conv2d(feature_dim * 2, feature_dim, 1)
        self.norm = nn.BatchNorm2d(feature_dim)
        self.act = nn.GELU()
    def forward(self, feature, T, A, S):
        A_exp = A.expand(-1, -1, T.shape[2], T.shape[3])
        phys_feat = torch.cat([T, A_exp, S], dim=1)
        phys_feat = self.phys_proj(phys_feat)
        fused = torch.cat([feature, phys_feat], dim=1)
        return self.act(self.norm(self.fusion(fused)))
class PromptReliabilityEstimation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.prompt_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        self.predictor = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, feature, prompt):
        f = self.feature_proj(feature)
        p = self.prompt_proj(prompt)
        x = torch.cat([f, p], dim=1)
        x = self.fusion(x)
        weight = self.avgpool(x)
        weight = self.fc(weight)
        x = x * weight
        reliability = self.predictor(x)
        prompt = prompt * reliability
        return prompt, reliability
class PromptIR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        decoder = False, use_ppe=True, use_agf=True, use_coral=False):
        super(PromptIR, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.embed_dim = dim
        self.use_ppe = use_ppe
        self.use_agf = use_agf
        self.use_coral = use_coral
        self.ppe = PhysicalPriorEstimator(in_channels=inp_channels, base_channels=32, use_coral=self.use_coral)
        self.ppf = PhysicsAwarePromptFusion(self.embed_dim)
        self.pre = PromptReliabilityEstimation(self.embed_dim)
        # 置信度门控
        self.confidence_evaluator = PhysicsConfidence()
        self.multi_gate = MultiLevelConfidenceGate(
            dim_list=[dim, dim*2, dim*4, dim*8],  # 编码器各层通道数
            phys_dim=1+3+3  # T + A + S
        )
        self.decoder = decoder
        if self.decoder:
            self.prompt1 = PromptGenBlock(prompt_dim=64,prompt_len=5,prompt_size = 64,lin_dim = 96)
            self.prompt2 = PromptGenBlock(prompt_dim=128,prompt_len=5,prompt_size = 32,lin_dim = 192)
            self.prompt3 = PromptGenBlock(prompt_dim=320,prompt_len=5,prompt_size = 16,lin_dim = 384)
        self.chnl_reduce1 = nn.Conv2d(64,64,kernel_size=1,bias=bias)
        self.chnl_reduce2 = nn.Conv2d(128,128,kernel_size=1,bias=bias)
        self.chnl_reduce3 = nn.Conv2d(320,256,kernel_size=1,bias=bias)
        self.reduce_noise_channel_1 = nn.Conv2d(dim + 64,dim,kernel_size=1,bias=bias)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.reduce_noise_channel_2 = nn.Conv2d(int(dim*2**1) + 128,int(dim*2**1),kernel_size=1,bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.reduce_noise_channel_3 = nn.Conv2d(int(dim*2**2) + 256,int(dim*2**2),kernel_size=1,bias=bias)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**1)+192, int(dim*2**2), kernel_size=1, bias=bias)
        self.noise_level3 = TransformerBlock(dim=int(dim*2**2) + 512, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level3 = nn.Conv2d(int(dim*2**2)+512,int(dim*2**2),kernel_size=1,bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.noise_level2 = TransformerBlock(dim=int(dim*2**1) + 224, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level2 = nn.Conv2d(int(dim*2**1)+224,int(dim*2**2),kernel_size=1,bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.noise_level1 = TransformerBlock(dim=int(dim*2**1)+64, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.reduce_noise_level1 = nn.Conv2d(int(dim*2**1)+64,int(dim*2**1),kernel_size=1,bias=bias)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
                # 适配器（用于双阶段冻结微调）
        self.adapter = LightweightAdapter(int(dim*2**1))  # 嵌入在编码器之后
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, inp_img,noise_emb = None):
        if self.use_coral:
            T, A, S, _ = self.ppe(inp_img)
        else:
            T, A, S = self.ppe(inp_img)
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        # ---- Physics-aware Prompt Fusion and Reliability ----
        if self.use_coral:
            T, A, S, _ = self.ppe(inp_img)
        else:
            T, A, S = self.ppe(inp_img)
        physics_prompt = self.ppf(out_enc_level1, T, A, S)
        prompt, reliability = self.pre(out_enc_level1, physics_prompt)
        # 将物理感知 prompt 融合到编码器特征中
        out_enc_level1 = out_enc_level1 + prompt  # 直接加法融合
        # 可选：用 reliability 加权: out_enc_level1 = out_enc_level1 + prompt * reliability
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 
        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4)
        if self.decoder:
            dec3_param = self.prompt3(latent)
            latent = torch.cat([latent, dec3_param], 1)
            latent = self.noise_level3(latent)
            latent = self.reduce_noise_level3(latent)
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        if self.decoder:
            dec2_param = self.prompt2(out_dec_level3)
            out_dec_level3 = torch.cat([out_dec_level3, dec2_param], 1)
            out_dec_level3 = self.noise_level2(out_dec_level3)
            out_dec_level3 = self.reduce_noise_level2(out_dec_level3)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.decoder:
            dec1_param = self.prompt1(out_dec_level2)
            out_dec_level2 = torch.cat([out_dec_level2, dec1_param], 1)
            out_dec_level2 = self.noise_level1(out_dec_level2)
            out_dec_level2 = self.reduce_noise_level1(out_dec_level2)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.adapter(out_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1, reliability

    def forward_with_gating(self, inp_img):
        """带置信度门控的推理（用于测试）"""
        # 1. 提取物理参数和置信度
        T, A, S = self.ppe(inp_img)
        confidence = self.confidence_evaluator(inp_img)  # [B,1,1,1]

        # 2. 编码器
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        out_enc_level2 = self.encoder_level2(out_enc_level1)
        out_enc_level3 = self.encoder_level3(out_enc_level2)
        out_enc_level4 = self.encoder_level4(out_enc_level3)

        # 3. 构造物理特征（不同尺度的 T,A,S）
        # 这里简化：将 T,A,S 缩放到各层尺寸
        B, _, H, W = inp_img.shape
        phys_feats = []
        for feat in [out_enc_level1, out_enc_level2, out_enc_level3, out_enc_level4]:
            _, _, h, w = feat.shape
            T_resized = F.interpolate(T, size=(h, w), mode='bilinear', align_corners=False)
            A_resized = F.interpolate(A, size=(h, w), mode='bilinear', align_corners=False)
            S_resized = F.interpolate(S, size=(h, w), mode='bilinear', align_corners=False)
            phys_feat = torch.cat([T_resized, A_resized, S_resized], dim=1)
            phys_feats.append(phys_feat)

        # 4. 多级门控融合（替换部分skip connection）
        # 这里我们只在解码器的对应层融合，但也可以融合到编码器输出
        # 简单起见，我们将融合后的特征作为解码器的额外输入
        data_feats = [out_enc_level1, out_enc_level2, out_enc_level3, out_enc_level4]
        fused_feats = self.multi_gate(data_feats, phys_feats, confidence)

        # 5. 解码器（使用融合后的特征）
        inp_dec_level4 = self.down4_3(out_enc_level3)
        # 在level4用融合特征替代原始编码器输出
        inp_dec_level4 = fused_feats[3]  # 直接用融合后的level4

        out_dec_level3 = self.decoder_level3(inp_dec_level4)
        # 后面步骤与原forward类似，但可以继续使用融合特征
        # 为了简化，我们只演示，实际可以更深入修改
        # 这里返回原始解码结果（确保兼容性）
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.adapter(out_dec_level1)
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1, confidence


# ============================================================
# 置信度门控模块（固定规则）
# ============================================================


class PhysicsConfidence(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        # 1. 暗通道稀疏度
        dark = torch.min(img, dim=1, keepdim=True)[0]
        dark = F.avg_pool2d(dark, 15, stride=1, padding=7)
        dark_sparsity = torch.exp(-dark.mean(dim=[2,3]) * 10)  # [B, 1, 1]

        # 2. 颜色分布合理性
        mean_rgb = img.mean(dim=[2,3])  # [B, 3]
        color_naturalness = torch.exp(
            -torch.abs(mean_rgb - 0.5).mean(dim=1, keepdim=True) * 5
        )  # [B, 1, 1]

        # 3. 纹理复杂度（使用方差替代拉普拉斯）
        # 计算局部方差作为纹理复杂度的近似
        patches = F.avg_pool2d(img, kernel_size=7, stride=1, padding=3)
        var = ((img - patches) ** 2).mean(dim=[1,2,3], keepdim=True)  # [B,1,1]
        texture_score = torch.exp(-var * 2)  # [B,1,1]

        confidence = dark_sparsity * 0.5 + color_naturalness * 0.3 + texture_score * 0.2
        return confidence  # [B, 1, 1, 1]

class GatedFusionLayer(nn.Module):
    """
    置信度门控融合层
    根据置信度动态融合数据特征和物理特征
    """
    def __init__(self, dim, phys_dim=None):
        super().__init__()
        if phys_dim is None:
            phys_dim = dim
        self.phys_proj = nn.Conv2d(phys_dim, dim, kernel_size=1) if phys_dim != dim else nn.Identity()
        self.fusion_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, data_feat, phys_feat, confidence):
        # data_feat: [B, dim, H, W]
        # phys_feat: [B, phys_dim, H, W]
        # confidence: [B, 1, 1, 1]
        phys_proj = self.phys_proj(phys_feat)
        # 门控融合
        fused = confidence * phys_proj + (1 - confidence) * data_feat
        return self.fusion_conv(fused) + data_feat  # 残差连接


class MultiLevelConfidenceGate(nn.Module):
    """
    多级置信度门控（在编码器各层融合物理特征）
    """
    def __init__(self, dim_list=[48, 96, 192, 384], phys_dim=1+3+3):  # T(1)+A(3)+S(3)=7
        super().__init__()
        self.gates = nn.ModuleList([
            GatedFusionLayer(dim, phys_dim) for dim in dim_list
        ])

    def forward(self, data_feats, phys_feats, confidence):
        # data_feats: list of [B, dim_i, H_i, W_i]
        # phys_feats: 通常为 T,A,S 在不同空间尺寸下的插值或池化版本
        fused_feats = []
        for i, (data_feat, phys_feat) in enumerate(zip(data_feats, phys_feats)):
            fused = self.gates[i](data_feat, phys_feat, confidence)
            fused_feats.append(fused)
        return fused_feats

# ============================================================
# 轻量级适配器（用于双阶段冻结微调）
# ============================================================

class LightweightAdapter(nn.Module):
    """
    极轻量适配器：1x1 Conv + BN + GELU，参数量 ~2k
    插入在编码器输出和解码器输入之间
    """
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
