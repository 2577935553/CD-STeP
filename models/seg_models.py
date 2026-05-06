"""Segmentation models for CD-STeP.

Defines two networks used by the paper:
    SingleUnet        - per-frame slice network (paper: SegSlc)
    SegSeq            - sequence network with SSTCR + DSTE (paper: SegSeq)

The SegSeq class additionally exposes the two contrastive losses
described in the paper:
    cstc_loss   = compute_temporal_similarity()  (CSTC, Sec. 2.2.1)
    cggc_loss   = compute_global_similarity()    (CGGC, Sec. 2.2.2)
"""
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.base import SegmentationModel, SegmentationHead
from networks.base.init_func import initialize_decoder, initialize_head
from networks.encoders import resnet50, get_encoder
from networks.decoders import UnetDecoder
from networks.cbam import SpatialGate, ChannelAtten, CrossAtten_channel, CrossAtten_spatial, SelfAtten
from einops import rearrange

from .dste import DSTE


class SingleUnet(SegmentationModel):
    def __init__(
            self,
            encoder_name: str = 'resnet50',
            encoder_weight='imagenet',
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 1,
            classes: int = 1,
            **kwargs,
    ):
        super(SingleUnet, self).__init__()

        self.encoder = get_encoder(encoder_name, in_channels=in_channels, weights=encoder_weight, **kwargs)
        self.decoder = UnetDecoder(
            decoder_channels=decoder_channels,
            encoder_channels=self.encoder.get_out_channels()
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.initialize()


class ResidualCrossAttention(nn.Module):
    """Residual Cross-Attention module for bidirectional attention"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, query, key, value):
        B, N, C = query.shape
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.out_proj(out)
        
        # Residual connection
        return self.norm(out + query)


class ContrastiveLoss(nn.Module):
    """Improved contrastive loss with multiple options"""
    def __init__(self, temperature=0.2, loss_type='focal'):
        """
        Args:
            temperature: Temperature for scaling similarities
            loss_type: 'infonce', 'normalized_ce', 'bce', or 'focal'
        """
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.eps = 1e-10
        
    def forward(self, similarity, target_similarity):
        """
        similarity: [B, H*W, H*W] - predicted similarity matrix
        target_similarity: [B, H*W, H*W] - binary matrix (1 for same object, 0 for different)
        """
        
        if self.loss_type == 'infonce':
            return self.infonce_loss(similarity, target_similarity)
        elif self.loss_type == 'normalized_ce':
            return self.normalized_ce_loss(similarity, target_similarity)
        elif self.loss_type == 'bce':
            return self.bce_loss(similarity, target_similarity)
        elif self.loss_type == 'focal':
            return self.focal_contrastive_loss(similarity, target_similarity)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def infonce_loss(self, similarity, target_similarity):
        """
        InfoNCE-style contrastive loss
        Treats pixels from same object as positive pairs, others as negatives
        """
        # Apply temperature scaling
        similarity = similarity / self.temperature+ self.eps
        
        # Mask for positive pairs (same object)
        pos_mask = target_similarity.bool()
        neg_mask=~pos_mask
        
        # For each query pixel, we want to maximize similarity with positive pairs
        # and minimize with negative pairs
        # exp_sim = torch.exp(similarity)
        
        # Sum of exp similarities with positive samples
        pos_sim = torch.where(pos_mask, similarity, torch.zeros_like(similarity))
        pos_sim_sum = pos_sim.sum(dim=-1, keepdim=True)
        neg_sim=torch.where(neg_mask, similarity, torch.zeros_like(similarity))
        neg_sim_sum=neg_sim.sum(dim=-1,keepdim=True)
        
        P=pos_sim_sum/(pos_mask.sum(dim=-1,keepdim=True)+1)
        N=neg_sim_sum/(neg_mask.sum(dim=-1,keepdim=True)+1)
        
        # Sum of all exp similarities (positive + negative)
        # all_sim_sum = exp_sim.sum(dim=-1, keepdim=True)
        
        # InfoNCE loss: -log(positive / all)
        # Average over all positive pairs
        # num_positives = pos_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        loss = -torch.log(torch.exp(P) / (torch.exp(P)+torch.exp(N)+self.eps))
        loss = loss.mean()
        
        return loss
    
    def normalized_ce_loss(self, similarity, target_similarity):
        """
        Cross-entropy loss with properly normalized target distribution
        """
        # Apply temperature scaling
        similarity = similarity / self.temperature
        
        # Normalize target_similarity to be a valid probability distribution
        # Each row sums to 1
        target_probs = target_similarity / (target_similarity.sum(dim=-1, keepdim=True) + self.eps)
        
        # Compute log_softmax of similarities
        log_probs = F.log_softmax(similarity, dim=-1)
        
        # Cross-entropy loss
        loss = -(target_probs * log_probs).sum(dim=-1).mean()
        
        return loss
    
    def bce_loss(self, similarity, target_similarity):
        """
        Binary cross-entropy loss treating each pair independently
        """
        # Normalize similarity to [0, 1] using sigmoid
        similarity_probs = torch.sigmoid(similarity / self.temperature)
        
        # Binary cross-entropy
        loss = F.binary_cross_entropy(similarity_probs, target_similarity)
        
        return loss
    
    def focal_contrastive_loss(self, similarity, target_similarity, gamma=2.0):
        """
        Focal loss variant for handling class imbalance
        (typically more negative pairs than positive pairs)
        """
        # Apply temperature scaling and sigmoid
        similarity_probs = torch.sigmoid(similarity / self.temperature)
        
        # Focal loss weights
        p_t = torch.where(target_similarity.bool(), similarity_probs, 1 - similarity_probs)
        focal_weight = (1 - p_t) ** gamma
        
        # Binary cross-entropy with focal weighting
        bce = F.binary_cross_entropy(similarity_probs, target_similarity.float(), reduction='none')
        loss = (focal_weight * bce).mean()
        
        return loss


class SegSeq(nn.Module):
    def __init__(
            self,
            encoder_name: str = 'resnet50',
            encoder_weight='imagenet',
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            in_channels: int = 1,
            classes: int = 1,
            frame_number: int = 20,
            reduction_rate: int = 32,
            reduc_time: int=32,
            use_contrast:bool=True,
            use_dste:bool=True,
            TB=True,
            SB=True,
            Gate=True,
            K_sample=4,
            bkg_sample_size=100,
            tau_bkg=0.5,
            **kwargs,
    ):
        super(SegSeq, self).__init__()
        self.use_contrast=use_contrast
        self.use_dste=use_dste
        
        self.activation = {}
        self.K_sample=K_sample
        self.bkg_sample_size=bkg_sample_size
        self.tau_bkg=tau_bkg

        self.encoder = get_encoder(encoder_name, in_channels=in_channels, weights=encoder_weight, **kwargs)
        self.cc = CrossAtten_channel(2048 * frame_number // reduction_rate)
        self.cs = CrossAtten_spatial(2048 * frame_number // reduction_rate)
        self.selfatten = SelfAtten(2048 * frame_number, reduction_ratio=reduction_rate)
        self.dste_module=DSTE(dim=2048,reduc_time=reduc_time,use_temperal=TB,use_spatial=SB,use_Gate=Gate)
        # self.classifier = Perceptron(2048 * frame_number // 256)
        self.classes=classes

        network = resnet50(None, in_channels=1 + classes, deep_stem=32, out_channels=[64, 256, 512, 1024, 2048])
        self.projector = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            nn.Conv2d(256, 64, 1, 1, 0)
        )
        self.Spatial_atten = nn.Sequential(
            nn.Identity(),
            SpatialGate(frame_number * 64, 1)
        )
        self.Channel_Atten = nn.Sequential(
            nn.Identity(),
            ChannelAtten(frame_number * 64, 1)
        )

        self.convblock = nn.Sequential(
            nn.Conv2d(2048 * frame_number // reduction_rate, 2048 * frame_number, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(),
            nn.ReLU(),
        )
        
        self.memory_conv = nn.Sequential(
            nn.Conv2d(512, 128, 1, 1, 0),
            nn.ReLU()
        )
        
        self.memory_conv_ = nn.Sequential(
            nn.Conv2d(256, 128, 1, 1, 0),
            nn.ReLU()
        )
        self.featFusion = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.decoder = UnetDecoder(
            decoder_channels=decoder_channels,
            encoder_channels=self.encoder.get_out_channels()
        )

        self.cross_atten_mk_qk=ResidualCrossAttention(dim=256)
        self.cross_atten_qk_mk=ResidualCrossAttention(dim=256)
        
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=None,
            kernel_size=3,
        )
        self.frame_number = frame_number
        
        
        self.mk_gconv= nn.Sequential(
            nn.Conv2d(64*self.frame_number, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            )
        
        self.qk_gconv= nn.Sequential(
            nn.Conv2d(64*self.frame_number, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            )
        self.qk_proj=nn.Sequential(
            nn.Conv2d(64,256,kernel_size=1,stride=1,padding=0),
            nn.ReLU(),
        )
        
        self.contrastive_loss=ContrastiveLoss()
        self.eps=1e-8
        self.init_fuc()

    def init_fuc(self):
        initialize_decoder(self.decoder)
        initialize_head(self.segmentation_head)

    def compute_global_similarity(self, mk, qk, gt_first_frame=None):
        """
        Compute global spatiotemporal similarity with GT supervision
        mk: [B, T*C, H, W]
        qk: [B, T*C, H, W]
        gt_first_frame: [B, 1, H, W] - ground truth for first frame
        """
        B, TC, H, W = mk.shape
        
        # Reshape to [B, T*C, H*W]
        mk_flat = mk.view(B, TC, H * W)
        qk_flat = qk.view(B, TC, H * W)
        
        # Compute similarity: Sim(Qk^T @ Mk)
        similarity = torch.bmm(qk_flat.transpose(1, 2), mk_flat)  # [B, H*W, H*W]
        similarity = similarity / math.sqrt(TC)+self.eps
        
        # If GT is provided, compute target similarity for supervision
        target_similarity = None
        if gt_first_frame is not None:
            gt_first_frame_=F.interpolate(gt_first_frame.float().unsqueeze(1),(H,W),mode='nearest')
            gt_flat = gt_first_frame_.view(B, -1, 1)  # [B, H*W, 1]
            target_similarity = (gt_flat==gt_flat.transpose(1, 2))  # [B, H*W, H*W]
        
        return similarity, target_similarity

    def compute_temporal_similarity(self, mk_t, qk_t, pseudo_probs=None,return_sims=False,K_sample=4,bkg_sample_size=100,tau_bkg=0.5):
        """
        Compute temporal similarity with pseudo label supervision
        Merges all T temporal foreground pixels for (N*T)^2 complexity computation
        
        mk_t: [B*T, C, H, W]
        qk_t: [B*T, C, H, W]
        pseudo_labels: [B*T, 1, H, W] - pseudo labels from SingleUnet
        """
        BT, C, H, W = mk_t.shape
        B = BT // self.frame_number
        T = self.frame_number
        
        # Reshape to separate batch and time dimensions
        mk_t = mk_t.view(B, T, C, H, W)
        qk_t = qk_t.view(B, T, C, H, W)
        
        similarities = []
        target_similarities = []
        
        
        if pseudo_probs is not None:
            pseudo_labels=pseudo_probs.argmax(dim=1)
            pseudo_labels = pseudo_labels.view(B, T, 1, H, W)
            pseudo_probs_=pseudo_probs.view(B, T, self.classes, H, W)
        
        
        temporal_loss=0
        valid_pairs = 0
        # Process each batch separately
        for b in range(B):
            mk_b = mk_t[b]  # [T, C, H, W]
            qk_b = qk_t[b]  # [T, C, H, W]
            
            if pseudo_labels is not None:
                pseudo_b = pseudo_labels[b]  # [T, 1, H, W]
                
                # Get foreground mask for all frames
                fg_mask = (pseudo_probs_[b][:,0] < tau_bkg).float()  # [T, 1, H, W],通过背景取前景
                fg_mask = fg_mask.view(T, H * W)  # [T, H*W]
                
                # Collect sparse indices and features across all time steps
                all_mk_features = []
                all_qk_features = []
                all_pseudo_values = []
                frame_indices = []  # Track which frame each pixel belongs to
                
                has_foreground = False
                
                for t in range(T):
                    indices = torch.nonzero(fg_mask[t]).squeeze(-1)
                    
                    if len(indices) > 0:
                        has_foreground = True
                        # Extract features for sparse foreground pixels
                        mk_t_flat = mk_b[t].view(C, H * W)  # [C, H*W]
                        qk_t_flat = qk_b[t].view(C, H * W)  # [C, H*W]
                        
                        mk_sparse = mk_t_flat[:, indices]  # [C, N_sparse_t]
                        qk_sparse = qk_t_flat[:, indices]  # [C, N_sparse_t]
                        pseudo_sparse = pseudo_b[t].view(-1)[indices]  # [N_sparse_t]
                        
                        all_mk_features.append(mk_sparse)
                        all_qk_features.append(qk_sparse)
                        all_pseudo_values.append(pseudo_sparse)
                        frame_indices.append(torch.full((len(indices),), t, device=mk_t.device))
                
                
                if has_foreground:
                    # sample mk & qk
                    choose_m=min(len(all_mk_features),K_sample)
                    choose_q=min(len(all_qk_features),K_sample)
                    choice_m=sorted(np.random.choice(len(all_mk_features),choose_m,replace=False))
                    choice_q=sorted(np.random.choice(len(all_qk_features),choose_q,replace=False))
                    
                    # Origin:
                    # Concatenate all sparse features across time
                    all_mk = torch.cat([all_mk_features[t] for t in choice_m], dim=1)  # [C, N_total]
                    all_qk = torch.cat([all_qk_features[t] for t in choice_q], dim=1)  # [C, N_total]
                    all_pseudo_m = torch.cat([all_pseudo_values[t] for t in choice_m], dim=0)  # [N_total]
                    all_pseudo_q = torch.cat([all_pseudo_values[t] for t in choice_q], dim=0)  # [N_total]
                    all_frame_idx = torch.cat(frame_indices, dim=0)  # [N_total]
                    
                    # Compute similarity matrix for all sparse pixels across time
                    # Complexity: (N*T)^2 where N is average sparse pixels per frame
                    sim = torch.mm(all_mk.T, all_qk) / math.sqrt(C)+self.eps  # [N_total, N_total]

                    target_sim = (all_pseudo_m.unsqueeze(1) == all_pseudo_q.unsqueeze(0)).float()  # [N_total, N_total]
                    
                    similarities.append(sim)
                    target_similarities.append(target_sim)
                    
                else:
                    bg_sample_size = min(bkg_sample_size, H * W)  # Sample limited background pixels
                    
                    all_mk_bg = []
                    all_qk_bg = []
                    all_pseudo_bg = []
                    
                    for t in range(T):
                        # Randomly sample background pixels
                        bg_indices = torch.randperm(H * W, device=mk_t.device)[:bg_sample_size]
                        
                        mk_t_flat = mk_b[t].view(C, H * W)
                        qk_t_flat = qk_b[t].view(C, H * W)
                        
                        mk_bg = mk_t_flat[:, bg_indices]  # [C, bg_sample_size]
                        qk_bg = qk_t_flat[:, bg_indices]  # [C, bg_sample_size]
                        pseudo_bg = pseudo_b[t].view(-1)[bg_indices]  # [bg_sample_size]
                        
                        all_mk_bg.append(mk_bg)
                        all_qk_bg.append(qk_bg)
                        all_pseudo_bg.append(pseudo_bg)
                    
                    # Concatenate background features
                    all_mk = torch.cat(all_mk_bg, dim=1)  # [C, T*bg_sample_size]
                    all_qk = torch.cat(all_qk_bg, dim=1)  # [C, T*bg_sample_size]
                    all_pseudo = torch.cat(all_pseudo_bg, dim=0)  # [T*bg_sample_size]
                    
                    # Compute similarity for background
                    sim = torch.mm(all_mk.T, all_qk) / math.sqrt(C)+self.eps
                    # target_sim = all_pseudo.unsqueeze(0) * all_pseudo.unsqueeze(1)
                    # 背景分支修正
                    target_sim = (all_pseudo.unsqueeze(1) == all_pseudo.unsqueeze(0)).float()  # [N, N]
                    
                    similarities.append(sim)
                    target_similarities.append(target_sim)
                    
            # else:
            #     # No pseudo labels provided - compute dense similarity
            #     # This is expensive, consider sampling
            #     mk_b_flat = mk_b.view(T * C, H * W)  # [T*C, H*W]
            #     qk_b_flat = qk_b.view(T * C, H * W)  # [T*C, H*W]
                
            #     sim = torch.mm(mk_b_flat.T, qk_b_flat) / math.sqrt(T * C)+self.eps  # [H*W, H*W]
            #     similarities.append(sim)
        # if valid_pairs > 0:
        #     temporal_loss = temporal_loss / valid_pairs
        # if return_sims:
        return similarities, target_similarities if pseudo_labels is not None else None
        # else:
        #     # return loss:
        #     return temporal_loss

    def forward(self, x, x_prev,gt_first_frame=None,return_features=False):
        
        features = self.encoder(x)
        out_features=[f.detach() for f in features]
        bottle_feature = features[-1] # (B*T)*C*H*W
        # pseudo_labels=F.interpolate(x_prev.argmax(dim=1).float(),(56,56),mode='nearest')
        H_,W_=features[1].shape[-2:]
        pseudo_probs=F.interpolate(x_prev[:,1:].softmax(dim=1).float(),(H_,W_),mode='bilinear')
        
        # Extract memory keys from previous frames
        memory_key = self.projector(x_prev)  # [B*T, 64, H', W']
        
        # Extract query keys from current frame features
        query_key =  self.conv1x1(features[1])  # [B*T, 64, H', W']
        
        B_total = query_key.shape[0]
        batch = B_total // self.frame_number
        _, C, H_feat, W_feat = query_key.shape
        
        # Further project keys
        # mk = self.mk_proj(memory_key)  # [B*T, 64, H', W']
        mk = memory_key
        qk = query_key  # [B*T, 64, H', W']
        
        # Reshape for global processing
        # 初始版本没有gconv
        mk_global = self.mk_gconv(mk.view(batch, self.frame_number * C, H_feat, W_feat))  # [B, T*C, H, W]
        qk_global = self.qk_gconv(qk.view(batch, self.frame_number * C, H_feat, W_feat))  # [B, T*C, H, W]
        
        # Store activations
        self.activation['memory_key'] = memory_key
        self.activation['query_key'] = query_key
        
        self.activation['memory_global'] = mk_global
        self.activation['query_global'] = qk_global
        
        # 1. Global spatiotemporal contrastive learning
        global_sim, target_global_sim = self.compute_global_similarity(
            mk_global, qk_global, gt_first_frame
        )
        
        self.activation['global_sim']=global_sim
        self.activation['target_global_sim']=target_global_sim
        
        # 2. Temporal contrastive learning with pseudo labels
        target_temporal_sims=None
        temporal_sims=None
        # if return_sims:
        temporal_sims, target_temporal_sims = self.compute_temporal_similarity(
            mk, qk, pseudo_probs,K_sample=self.K_sample,bkg_sample_size=self.bkg_sample_size,tau_bkg=self.tau_bkg
        )
    # else:
        #     temperal_closs=self.compute_temporal_similarity(
        #         mk, qk, pseudo_probs,return_sims
        #     )
        # if return_sims:
        self.activation['temporal_sims']=temporal_sims
        self.activation['target_temporal_sims']=target_temporal_sims
        
        # print(temperal_closs)
        losses = {}
        if self.training:
            if target_global_sim is not None:
                losses['global_contrastive'] = self.contrastive_loss(global_sim, target_global_sim)
            
            if target_temporal_sims is not None:
                temporal_loss = 0
                for sim, target_sim in zip(temporal_sims, target_temporal_sims):
                    if (target_sim is not None) and (sim is not None):
                        temporal_loss += self.contrastive_loss(sim, target_sim)
                    else:
                        continue
                losses['temporal_contrastive'] = temporal_loss / len(temporal_sims)
            # else:
            #     losses['temporal_contrastive'] = temperal_closs
        
        # 3. Bidirectional cross-attention and fusion
        # readout = self.bidirectional_cross_attention(mk_global, qk_global)
        # 
        # Reshape readout for decoder
        # readout = readout.unsqueeze(1).repeat(1, self.frame_number, 1, 1, 1)
        # memory = readout.view(B_total, 128, H_feat, W_feat)
        
        # use qk directly
        qk=rearrange(self.qk_proj(qk),'(b t) c h w -> b t c h w',b=batch)
        readout=(qk+qk_global.unsqueeze(1)).flatten(0,1) # B,256,H,W
        memory=self.memory_conv_(readout)
        
        bottle_feature = bottle_feature.view(bottle_feature.size(0) // self.frame_number,
                                             self.frame_number , 2048, bottle_feature.size(2), bottle_feature.size(3))
        
        self.activation['bottleneck'] = bottle_feature
        x_atten,losses_dste=self.dste_module(bottle_feature,self.activation)
        x_atten=x_atten.flatten(0,1)
        # chann_feat, spat_feat = self.selfatten(
        #     bottle_feature)  # [b, 2048*frame/reduce, 1, 1], #[b, 2048*frame/reduce, h,w]
        # self.activation['chann_feat'], self.activation['spat_feat'] = chann_feat, spat_feat
        # x_chann = self.cc(chann_feat, spat_feat)  # [b, 2048*frame/reduce, h, w]
        # x_spat = self.cs(chann_feat, spat_feat)  # [b, 2048*frame/reduce, h, w]
        # self.activation['T@S'] = x_chann
        # self.activation['S@T'] = x_spat
        # x_atten = self.convblock(x_chann + x_spat)  # [b, 2048*frame, h, w]
        # x_atten = x_atten.view(x_atten.size(0) * self.frame_number, -1, x_atten.size(2),
        #                        x_atten.size(3))  # [b*frame, 2048, h, w]

        # frame_weight = self.classifier(chann_feat, spat_feat)
        if self.use_dste:
            features[-1] = x_atten
        self.activation['f1'] = features[1]
        if self.use_contrast:
            features[1] = self.featFusion(torch.cat([features[1], memory], dim=1))
        self.activation['fusion_feat'] = features[1]
        decoder_output = self.decoder(*features)
        if return_features:
            decoder_output,dec_feats=self.decoder(*features,return_feats=return_features)
        mask = self.segmentation_head(decoder_output)

        if return_features:
            return mask,out_features,dec_feats
        else:
            return mask,losses,losses_dste

    @torch.no_grad()
    def predict(self, x, x_prev):
        if self.training:
            self.eval()
        mask = self.forward(x, x_prev)

        return mask



# ---------------------------------------------------------------------------
# Paper-aligned aliases.
#
# The training scripts and the rest of the codebase historically refer to
# the two networks by these older names; the aliases below let new code
# import the paper-aligned names without breaking older imports.
# ---------------------------------------------------------------------------
SegSlc = SingleUnet                # per-frame slice segmentation network
TempSeg_Mem_New_ALL = SegSeq       # backward-compat alias for old training scripts


__all__ = [
    'SingleUnet', 'SegSlc',
    'SegSeq', 'TempSeg_Mem_New_ALL',
    'ContrastiveLoss',
    'ResidualCrossAttention',
]
