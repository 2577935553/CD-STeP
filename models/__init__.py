"""Top-level model interface for CD-STeP.

Paper-aligned names (preferred):
    SegSlc      - per-frame slice network
    SegSeq      - sequence network with SSTCR + DSTE
    DSTE        - Decoupled Spatial-Temporal Enhancement
    CDE / SCA / DMBF - the three sub-modules of DSTE

Legacy names (kept for backward compatibility with the existing training
scripts; resolve to the same classes):
    SingleUnet, TempSeg_Mem_New_ALL,
    ImprovedTemporalModule, LightweightSpatialModule, AdaptiveFusion
"""
from .seg_models import (
    SingleUnet, SegSlc,
    SegSeq, TempSeg_Mem_New_ALL,
    ContrastiveLoss, ResidualCrossAttention,
)
from .dste import (
    DSTE,
    CDE, SCA, DMBF,
    ImprovedTemporalModule, LightweightSpatialModule, AdaptiveFusion,
)

__all__ = [
    'SegSlc', 'SegSeq',
    'SingleUnet', 'TempSeg_Mem_New_ALL',
    'ContrastiveLoss', 'ResidualCrossAttention',
    'DSTE', 'CDE', 'SCA', 'DMBF',
    'ImprovedTemporalModule', 'LightweightSpatialModule', 'AdaptiveFusion',
]
