import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
import numpy as np
from typing import *

class LGAFlow(nn.Module):
    """
    Local-Global Attention Flow (LGA-Flow) Module
    融合了局部注意力和全局上下文,层级光流预测
    embed_dim=
    """
    def __init__(self, feature_dim: int, embed_dim: int = 128, search_radius: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.search_radius = search_radius
        
        # 1. 特征投影
        self.query_proj = nn.Conv2d(feature_dim, embed_dim, 1)
        self.key_proj = nn.Conv2d(feature_dim, embed_dim, 1)
        
        # 将全局上下文融合回特征的MLP
        self.context_fusion_mlp = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim,1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim,1)
        )
        
        
        
        # 预先计算相对坐标并注册为buffer
        rel_coords_y, rel_coords_x = torch.meshgrid(
            torch.arange(-self.search_radius, self.search_radius + 1),
            torch.arange(-self.search_radius, self.search_radius + 1),
            indexing='ij'
        )
        rel_coords = torch.stack([rel_coords_x, rel_coords_y], dim=-1).float() # [K, K, 2]
        self.register_buffer('rel_coords', rel_coords.view(-1, 2), persistent=False) # [K*K, 2]

    def forward(self, feat_t:torch.Tensor,context:torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = feat_t.shape
        # print(feat_t.shape,context.shape)
        flows=[]
        for b in range(B):
            query=self.query_proj(feat_t[b][:T-1])
            key = self.key_proj(feat_t[b][1:])
            if T-1>1:
                cont=context[b].repeat(T-1,1,1,1) #T-1,C,H,W
            else:
                cont=context[b]
            # print('qkv:',query.shape,key.shape,cont.shape)
            fused_query_flat = self.context_fusion_mlp(torch.cat([query, cont], dim=1)) # T-1,C,H,W
            key_unfolded = F.unfold(
                key, 
                kernel_size=(self.search_radius * 2 + 1, self.search_radius * 2 + 1),
                padding=self.search_radius
            ) # [B, C * K*K, H*W] where K = search_radius*2+1,这里做了padding，滑动次数等于次数HW
            key_local = rearrange(key_unfolded, 'b (c k) n -> b n k c', c=self.embed_dim) # T-1,HW,K^2,C
            query_pointwise = rearrange(fused_query_flat, 'b c h w -> b (h w) c').unsqueeze(2)# T-1,HW,1,C
            local_corr = torch.sum(query_pointwise * key_local, dim=-1)/np.sqrt(self.embed_dim) # [B, N, K*K]
            local_corr = F.softmax(local_corr, dim=-1) # [B, N, K*K]
            flow = torch.matmul(local_corr, self.rel_coords)
            flow = rearrange(flow, 'b (h w) c -> b c h w', h=H, w=W) #像素动量,T-1,2,H,W
            flows.append(flow)
        return torch.stack(flows,dim=-0) # B,T-1,2,H,W
    
class DisplacementWarper(nn.Module):
    """
    接收一个以像素为单位的位移场 (flow)，并使用它来扭曲源张量。
    flow 的定义: source_coords = target_coords + flow(target_coords)
    """
    def __init__(self):
        super().__init__()
        self.grid_cache = {}

    def _create_grid(self, B: int, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cache_key = (B, H, W, device, dtype)
        if cache_key in self.grid_cache:
            return self.grid_cache[cache_key]

        coords_y, coords_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        pixel_grid = torch.stack([coords_x, coords_y], dim=-1)
        
        pixel_grid = pixel_grid.unsqueeze(0).expand(B, -1, -1, -1) # [B, H, W, 2]
        
        self.grid_cache[cache_key] = pixel_grid
        return pixel_grid

    def forward(self, source: torch.Tensor, flow: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
        """
        使用位移场 `flow` 扭曲 `source` 张量。

        Args:
            source (torch.Tensor): 要扭曲的张量 (图像, 特征, 标签), shape [B, C, H, W]
            flow (torch.Tensor): 位移场, shape [B, 2, H, W], flow[0]是dx, flow[1]是dy
            mode (str): grid_sample的插值模式。'bilinear'用于图像/特征，'nearest'用于标签。

        Returns:
            torch.Tensor: 扭曲后的张量, shape [B, C, H, W]
        """
        B, C, H, W = source.shape

        # 1. 获取目标像素坐标网格 (单位: 像素)
        # grid_target_pixels: [B, H, W, 2] in (x, y) order
        grid_target_pixels = self._create_grid(B, H, W, source.device, source.dtype)

        # 2. 计算源像素坐标
        # flow: [B, 2, H, W] -> [B, H, W, 2]
        flow_permuted = flow.permute(0, 2, 3, 1) # (dx, dy)
        # grid_source_pixels = grid_target_pixels + flow_permuted
        grid_source_pixels = grid_target_pixels + flow_permuted

        # 3. 将源像素坐标标准化到 [-1, 1] 范围
        flow_x = grid_source_pixels[..., 0]
        flow_y = grid_source_pixels[..., 1]
        
        # 处理 W=1 或 H=1 的边缘情况
        W_denom = max(W - 1, 1)
        H_denom = max(H - 1, 1)

        grid_source_norm = torch.stack([
            2.0 * flow_x / W_denom - 1.0,
            2.0 * flow_y / H_denom - 1.0
        ], dim=-1) # [B, H, W, 2]

        # 4. 执行扭曲操作
        warped_source = F.grid_sample(
            source,
            grid_source_norm,
            mode=mode,
            padding_mode='border',
            align_corners=True
        )

        return warped_source

class TemporalWarper(nn.Module):
    """
    一个封装了光流估计和多步扭曲功能的完整模块。
    
    工作流程:
    1. 输入整个视频序列的特征和上下文。
    2. 使用LGAFlow计算所有相邻帧的前向和后向光流。
    3. 提供一个 `warp` 方法，可以处理任意源帧到目标帧的多步扭曲。
    """
    def __init__(self, feature_dim: int, embed_dim: int = 128, search_radius: int = 4):
        super().__init__()
        self.flow_estimator = LGAFlow(feature_dim, embed_dim, search_radius)
        self.feature_warper = DisplacementWarper() # 使用上一节的简单扭曲模块
        self.grid_cache = {}

    def _create_pixel_grid(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        cache_key = (B, H, W, device)
        if cache_key in self.grid_cache:
            return self.grid_cache[cache_key]
        
        coords_y, coords_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        grid = torch.stack([coords_x, coords_y], dim=-1)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        self.grid_cache[cache_key] = grid
        return grid

    def _normalize_grid(self, grid: torch.Tensor, H: int, W: int) -> torch.Tensor:
        W_denom = max(W - 1, 1)
        H_denom = max(H - 1, 1)
        return torch.stack([
            2.0 * grid[..., 0] / W_denom - 1.0,
            2.0 * grid[..., 1] / H_denom - 1.0
        ], dim=-1)
    
    
    def forward(self, features_seq: torch.Tensor, context_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算整个序列的前向和后向光流场。

        Args:
            features_seq (torch.Tensor): 视频特征序列, shape [B, T, C, H, W]
            context_seq (torch.Tensor): 全局时序上下文, shape [B, 1, C, H, W]

        Returns:
            Dict[str, torch.Tensor]: 包含前向和后向光流的字典。
        """
        forward_flows = self.flow_estimator(features_seq, context_seq) # [B, T-1, 2, H, W]
        # 计算后向光流: t+1 -> t
        features_seq_reversed = torch.flip(features_seq, [1])
        backward_flows_reversed = self.flow_estimator(features_seq_reversed, context_seq)
        # 将结果反转回来以匹配原始时间顺序
        backward_flows = torch.flip(backward_flows_reversed, [1]) # [B, T-1, 2, H, W] 2->1;3->2;4->3...
        
        return {
            "forward_flows": forward_flows,
            "backward_flows": backward_flows
        }

    def warp(self, source: torch.Tensor, source_idx: int, target_idx: int, 
             forward_flows: torch.Tensor, backward_flows: torch.Tensor, 
             mode: str = 'bilinear') -> torch.Tensor:
        """
        将 `source` 从时间点 `source_idx` 扭曲到 `target_idx`。

        Args:
            source (torch.Tensor): 要扭曲的张量 (图像, 特征, 标签), shape [B, C, H, W]
            source_idx (int): 源帧的索引。
            target_idx (int): 目标帧的索引。
            forward_flows (torch.Tensor): [B, T-1, 2, H, W]
            backward_flows (torch.Tensor): [B, T-1, 2, H, W]
        Returns:
            torch.Tensor: 扭曲后的张量。
        """
        if source_idx == target_idx:
            return source

        current_data = source
        
        if target_idx > source_idx:
            # 前向扭曲
            for i in range(source_idx, target_idx):
                flow = forward_flows[:, i]
                # 上采样光流以匹配数据分辨率
                flow = F.interpolate(flow, size=current_data.shape[-2:], mode='bilinear', align_corners=True)
                current_data = self.feature_warper(current_data, flow, mode=mode)
        else:
            # 后向扭曲
            for i in range(source_idx, target_idx, -1):
                # 后向扭曲使用 backward_flows 的 i-1 索引
                flow = backward_flows[:, i-1]
                flow = F.interpolate(flow, size=current_data.shape[-2:], mode='bilinear', align_corners=True)
                current_data = self.feature_warper(current_data, flow, mode=mode)
                
        return current_data
    
    @torch.no_grad()
    def get_long_range_flow(self,
                            forward_flows: torch.Tensor,
                            backward_flows: torch.Tensor,
                            source_idx: int,
                            target_idx: int) -> torch.Tensor:
        """
        通过追踪坐标网格，计算从 source_idx 到 target_idx 的累积长程光流 (支持双向)。
        """
        if source_idx == target_idx:
            B, T,C, H, W = forward_flows.shape
            return torch.zeros(B, 2, H, W, device=forward_flows.device)

        B,T, _, H, W = forward_flows.shape
        grid_initial = self._create_pixel_grid(B, H, W, forward_flows.device)
        grid_current = grid_initial

        if target_idx > source_idx:
            # --- 前向累积 ---
            for i in range(source_idx, target_idx):
                flow_i = forward_flows[:, i]
                grid_current_norm = self._normalize_grid(grid_current, H, W)
                particle_velocities = F.grid_sample(
                    flow_i, grid_current_norm, mode='bilinear', padding_mode='border', align_corners=True
                )
                grid_current = grid_current + particle_velocities.permute(0, 2, 3, 1)
        else: # target_idx < source_idx
            # --- 后向累积 ---
            for i in range(source_idx, target_idx, -1):
                # 使用 backward_flows 的 i-1 索引
                flow_i = backward_flows[:, i-1]
                grid_current_norm = self._normalize_grid(grid_current, H, W)
                particle_velocities = F.grid_sample(
                    flow_i, grid_current_norm, mode='bilinear', padding_mode='border', align_corners=True
                )
                grid_current = grid_current + particle_velocities.permute(0, 2, 3, 1)

        total_displacement = grid_current - grid_initial
        return total_displacement.permute(0, 3, 1, 2)

    def get_direct_long_range_flow(self,
                                   features_seq: torch.Tensor,
                                   context_seq: torch.Tensor,
                                   source_idx: int,
                                   target_idx: int) -> torch.Tensor:
        """
        直接使用 flow_estimator 估计长程光流 (支持双向)。
        """
        if source_idx == target_idx:
            B, _, C, H, W = features_seq.shape
            return torch.zeros(B, 2, H, W, device=features_seq.device)
        
        # 确保 source_feat 始终是时间上的第一帧
        source_feat = features_seq[:, source_idx:source_idx+1]
        target_feat = features_seq[:, target_idx:target_idx+1]
        # 估计 t -> t+1 的光流
        two_frame_seq = torch.cat([source_feat, target_feat], dim=1)
        long_range_flow = self.flow_estimator(two_frame_seq, context_seq)[:, 0]
        

        return long_range_flow
