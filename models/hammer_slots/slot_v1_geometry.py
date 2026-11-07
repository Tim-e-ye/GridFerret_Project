#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[SYM-GF-SLOT-V1] Slot V1: Homography-Anchored Sub-Pixel Deformable Phase Locking (`subpixel_deform_V1A`)
vs. Global Affine Homography (`affine_homography_V1B`) vs. Rigid Cartesian Grid (`original_rigid`).
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


class HammerSlotV1(nn.Module):
    """
    @bce_contract
    slot_id: SYM-GF-SLOT-V1
    target_nail: Perspective Off-by-One Phase Drift on 19x19 Oblique Grids (Nail V-B)
    hard_invariants:
      - Gate 1: Masked/occluded intersections must have zero feature leakage (<1e-6)
      - Gate 4: Negative-biased micro-gate init b0 = -2.2 (sigmoid(-2.2) ≈ 0.0997)
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.deform_conv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.deform_conv[-1].weight)
        nn.init.zeros_(self.deform_conv[-1].bias)
        self.micro_gate = nn.Parameter(torch.tensor(-2.2))  # sigmoid(-2.2) ≈ 0.0997
        self.homography_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        v1_type: str = "original_rigid",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m = mask.unsqueeze(-1)
        x = x * m
        if v1_type in ("original_rigid", "base"):
            return x, torch.zeros(x.shape[0], x.shape[1], 2, device=x.device)
        if v1_type == "subpixel_deform_V1A":
            gate = torch.sigmoid(self.micro_gate)
            delta = torch.tanh(self.deform_conv(x))
            out = (x + gate * delta) * m
            pred_offset = out[..., :2] * 0.25
            return out, pred_offset
        if v1_type == "affine_homography_V1B":
            global_ctx = x.sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp(min=1.0)
            out = (x + 0.08 * torch.tanh(self.homography_proj(global_ctx))) * m
            return out, out[..., :2] * 0.15
        return x, torch.zeros(x.shape[0], x.shape[1], 2, device=x.device)
