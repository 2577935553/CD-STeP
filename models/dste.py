"""Decoupled Spatial-Temporal Enhancement (DSTE) module.

Implements the three sub-modules of paper Sec. 2.3:

    CDE  - Causal Dynamics Estimation     (ImprovedTemporalModule)
    SCA  - Spatial Context Aggregator     (LightweightSpatialModule)
    DMBF - Dynamic Multi-Branch Fusion    (AdaptiveFusion)

The top-level :class:`DSTE` wraps the three sub-modules with a content-aware
gate and a residual connection, matching Eq. (21) in the paper.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ImprovedTemporalModule(nn.Module):

    def __init__(self, dim: int,reduc_time=8):
        super().__init__()
        self.reduction = dim // reduc_time
        
        self.dim_reduce = nn.Sequential(
            nn.Conv3d(dim, self.reduction, kernel_size=1),
            nn.GroupNorm(min(8, self.reduction), self.reduction),
            nn.GELU()
        )
        
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(self.reduction, self.reduction * 2, 
                     kernel_size=(3, 1, 1), padding=(0, 0, 0)),
            nn.GroupNorm(min(16, self.reduction * 2), self.reduction * 2),
            nn.GELU(),
            nn.Conv3d(self.reduction * 2, self.reduction, kernel_size=1)
        )
        
        self.temporal_attn = nn.Sequential(
            nn.Linear(self.reduction, self.reduction),
            nn.LayerNorm(self.reduction),
            nn.GELU()
        )
        

        self.motion_intensity_head = nn.Sequential(
            nn.Linear(self.reduction * 2, self.reduction),
            nn.LayerNorm(self.reduction),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.reduction, 1)
        )
        

        self.velocity_smooth_head = nn.Sequential(
            nn.Linear(self.reduction * 3, self.reduction),
            nn.LayerNorm(self.reduction),
            nn.GELU(),
            nn.Linear(self.reduction, 1)
        )
        

        self.dim_restore = nn.Sequential(
            nn.Conv3d(self.reduction, dim, kernel_size=1),
            nn.GroupNorm(min(32, dim), dim)
        )
        
    def forward(self, x: torch.Tensor,activation={}):
        """
        x: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        

        x_in = rearrange(x, 'b t c h w -> b c t h w')
        x_reduced = self.dim_reduce(x_in)  # [B, reduction, T, H, W]
        activation['dste_reduc']=x_reduced
        

        x_padded = F.pad(x_reduced, (0, 0, 0, 0, 2, 0))
        x_conv = self.temporal_conv(x_padded)  # [B, reduction, T, H, W]
        activation['dste_causal_conv']=x_conv
        

        x_temporal = x_reduced.mean(dim=[3, 4])  # [B, reduction, T]
        x_conv_pooled = x_conv.mean(dim=[3, 4])  # [B, reduction, T]
        
        

        x_attn = rearrange(x_conv_pooled, 'b c t -> b t c')
        x_attn = self.temporal_attn(x_attn)
        x_attn = rearrange(x_attn, 'b t c -> b c t')
        

        x_temporal = x_temporal + x_attn
        

        loss_motion = 0.0
        loss_smooth = 0.0
        
        if T > 1 and self.training:
            motion_pairs = []
            motion_targets = []
            
            for i in range(T - 1):
                feat_i = x_temporal[:, :, i]
                feat_j = x_temporal[:, :, i + 1]
                

                true_motion = torch.norm(feat_j - feat_i, p=2, dim=1, keepdim=True)
                true_motion = true_motion / (true_motion.max() + 1e-6)
                motion_pairs.append(torch.cat([feat_i, feat_j], dim=1))
                motion_targets.append(true_motion)
            
            if motion_pairs:
                motion_pairs = torch.stack(motion_pairs, dim=1).view(-1, self.reduction * 2)
                motion_targets = torch.cat(motion_targets, dim=0).view(-1, 1)
                
                pred_motion = self.motion_intensity_head(motion_pairs)
                loss_motion = F.smooth_l1_loss(pred_motion, motion_targets.detach())
                activation['motion_preds']=pred_motion
                activation['motion_targets']=motion_targets
                activation['motion_pairs']=motion_pairs
        
        x_temporal = x_temporal.unsqueeze(-1).unsqueeze(-1)  # [B, reduction, T, 1, 1]
        x_temporal = x_temporal.expand(-1, -1, -1, H, W)
        
        x_out = self.dim_restore(x_temporal)  # [B, C, T, H, W]
        x_out = rearrange(x_out, 'b c t h w -> b t c h w')
        
        return x_out, loss_motion


class LightweightSpatialModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = dim // 16
        
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(dim, self.reduction, kernel_size=1),
            nn.GroupNorm(8, self.reduction),
            nn.GELU()
        )
        
        self.spatial_conv = nn.Sequential(
            # 3x3 depthwise
            nn.Conv2d(self.reduction, self.reduction, 3, padding=1, groups=self.reduction),
            nn.GroupNorm(8, self.reduction),
            nn.GELU(),
            # 5x5 depthwise (dilated)
            nn.Conv2d(self.reduction, self.reduction, 3, padding=2, dilation=2, groups=self.reduction),
            nn.GroupNorm(8, self.reduction),
            nn.GELU(),
            # pointwise
            nn.Conv2d(self.reduction, self.reduction, 1)
        )
        
        self.spatial_pool = nn.AdaptiveAvgPool2d(7)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=self.reduction,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.spatial_unpool = nn.Upsample(size=14, mode='bilinear', align_corners=False)
        
        self.contrast_head = nn.Sequential(
            nn.Linear(self.reduction, self.reduction // 2),
            nn.GELU(),
            nn.Linear(self.reduction // 2, self.reduction // 2)
        )
        
        self.dim_restore = nn.Sequential(
            nn.Conv2d(self.reduction, dim, kernel_size=1),
            nn.GroupNorm(32, dim)
        )
        
        self.temperature=0.1
        
    def forward(self, x: torch.Tensor,activation={}):
        """
        x: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape
        
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        
        x_reduced = self.dim_reduce(x_flat)  # [BT, reduction, H, W]
        activation['dste_spatial_reduc']=x_reduced
        
        x_conv = self.spatial_conv(x_reduced)
        x_spatial = x_reduced + x_conv
        activation['dste_spatial_reduc_s1']=x_spatial
        
        x_pooled = self.spatial_pool(x_spatial)  # [BT, reduction, 7, 7]
        x_pooled_flat = rearrange(x_pooled, 'bt c h w -> bt (h w) c')
        activation['dste_spatial_reduc_pool']=x_pooled
        
        x_attn, attn_weights = self.spatial_attn(x_pooled_flat, x_pooled_flat, x_pooled_flat)
        x_attn = rearrange(x_attn, 'bt (h w) c -> bt c h w', h=7, w=7)
        activation['dste_spatial_attn']=x_attn
        activation['spatial_attn_weights'] = attn_weights
        
        x_attn_up = self.spatial_unpool(x_attn)  # [BT, reduction, H, W]
        x_spatial = x_spatial + x_attn_up
        
        loss_contrast = 0.0
        if self.training:
            grid_size = 4
            patch_h, patch_w = H // grid_size, W // grid_size
            

            global_feat = x_spatial.mean(dim=[2, 3])  # [BT, reduction]
            global_proj = self.contrast_head(global_feat)  # [BT, reduction//2]
            

            patch_feats = []
            for i in range(grid_size):
                for j in range(grid_size):
                    h_start = i * patch_h
                    w_start = j * patch_w
                    patch = x_spatial[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w]
                    patch_feat = patch.mean(dim=[2, 3])
                    patch_feats.append(self.contrast_head(patch_feat))
            
            patch_feats = torch.stack(patch_feats, dim=1)  # [BT, grid_size^2, reduction//2]
            


            # global_proj_norm = F.normalize(global_proj, dim=1)  # [BT, reduction//2]
            # patch_feats_norm = F.normalize(patch_feats, dim=2)  # [BT, grid_size^2, reduction//2]
            

            logits = torch.sigmoid(torch.bmm(patch_feats, global_proj.unsqueeze(2)).squeeze(2)/self.temperature)  # [BT, grid_size^2]

            loss_contrast = 1.0 - logits.mean()
        

        x_out = self.dim_restore(x_spatial)
        activation['dste_spatial_out']=x_out
        x_out = rearrange(x_out, '(b t) c h w -> b t c h w', b=B)
        
        return x_out, loss_contrast


class AdaptiveFusion(nn.Module):
    """自适应融合模块 - 简单但有效"""
    def __init__(self, dim: int):
        super().__init__()

        self.gate_conv = nn.Sequential(
            nn.Conv3d(dim * 3, dim // 4, kernel_size=1),
            nn.GroupNorm(8, dim // 4),
            nn.GELU(),
            nn.Conv3d(dim // 4, 3, kernel_size=1)
        )
        
    def forward(self, x_temporal, x_spatial, x_orig,activation={}):
        """
        自适应融合三路特征
        """
        B, T, C, H, W = x_temporal.shape
        

        x_all = torch.stack([x_temporal, x_spatial, x_orig], dim=2)  # [B, T, 3, C, H, W]
        x_all = rearrange(x_all, 'b t n c h w -> b (n c) t h w')
        

        gates = self.gate_conv(x_all)  # [B, 3, T, H, W]
        gates = F.softmax(gates, dim=1)
        activation['gates']=gates
        

        gates = rearrange(gates, 'b n t h w -> b t n 1 h w')
        x_all = rearrange(x_all, 'b (n c) t h w -> b t n c h w', n=3)
        
        x_fused = (x_all * gates).sum(dim=2)  # [B, T, C, H, W]
        activation['fused_out']=x_fused
        
        return x_fused

class DSTE(nn.Module):
    """
    重新设计的DSTE模块
    核心改进：
    1. 大幅降低计算量（降维到1/16）
    2. 简化自监督任务，确保可收敛
    3. 移除复杂的Transformer，使用轻量级操作
    4. 清晰的梯度流设计
    """
    def __init__(self, dim: int, t_branch=True,s_branch=True,reduc_time=32,use_temperal=True,use_spatial=True,use_Gate=True):
        super().__init__()
        self.dim = dim
        self.t_branch = t_branch
        self.s_branch = s_branch
        self.use_gate=use_Gate
        

        # self.temporal_branch = LightweightTemporalModule(dim) # ---title1
        self.temporal_branch =ImprovedTemporalModule(dim,reduc_time)
        self.spatial_branch = LightweightSpatialModule(dim)
        

        self.fusion = AdaptiveFusion(dim)
        # self.fusion = AdaptiveWeightedFusion(dim)
        

        self.output_proj = nn.Sequential(
            nn.Conv3d(dim, dim // 4, kernel_size=1),
            nn.GroupNorm(8, dim // 4),
            nn.GELU(),
            nn.Conv3d(dim // 4, dim, kernel_size=1)
        )
        

        # self.residual_weight = nn.Parameter(torch.tensor(0.1))
        self.residual_weight=1.0
        
    def forward(self, x: torch.Tensor,activation={}):
        """
        x: [B, T, C, H, W]
        返回: enhanced_x, losses_dict
        """
        B, T, C, H, W = x.shape
        
        if self.t_branch:
            x_t, loss_temporal = self.temporal_branch(x,activation)
        else:
            x_t = x
            loss_temporal = torch.tensor(0.0, device=x.device)
        
        if self.s_branch:
            x_s, loss_spatial = self.spatial_branch(x,activation)
        else:
            x_s = x
            loss_spatial = torch.tensor(0.0, device=x.device)
        
        if self.use_gate:
            x_fused = self.fusion(x_t, x_s, x,activation)
            
            x_fused = rearrange(x_fused, 'b t c h w -> b c t h w')
            x_proj = self.output_proj(x_fused)
            x_proj = rearrange(x_proj, 'b c t h w -> b t c h w')
            
            x_out = x + self.residual_weight * x_proj
        else:
            x_out=x+self.residual_weight*(x_t+x_s)
        
        losses = {
            'loss_temporal_consistency': loss_temporal,
            'loss_spatial_contrast': loss_spatial,
            'loss_total_self_supervised': loss_temporal + loss_spatial
        }
        
        return x_out, losses




# ---------------------------------------------------------------------------
# Paper-aligned aliases.
# ---------------------------------------------------------------------------
CDE  = ImprovedTemporalModule       # Causal Dynamics Estimation (Sec. 2.3.1)
SCA  = LightweightSpatialModule     # Spatial Context Aggregator  (Sec. 2.3.2)
DMBF = AdaptiveFusion               # Dynamic Multi-Branch Fusion (Sec. 2.3.3)


__all__ = [
    'DSTE',
    'CDE', 'SCA', 'DMBF',
    'ImprovedTemporalModule', 'LightweightSpatialModule', 'AdaptiveFusion',
]
