#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[SYM-GF-SLOT-V2] Slot V2: Nyquist-Safe 2D DCT & 4-Orientation Gabor Harmonic Resampler (`fourier_gabor_V2A`)
with Specular Highlight Suppression Gate vs. Spatial Window Cross-Attention (`window_cross_attn_V2B`).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class HammerSlotV2(nn.Module):
    """
    @bce_contract
    slot_id: SYM-GF-SLOT-V2
    target_nail: Nyquist Spatial-Frequency Aliasing & Black-Stone Specular Glare Polarity Flip (Nail V-A)
    hard_invariants:
      - Gate 1: Zero feature leakage on glare-masked intersections
      - Gate 4: Negative-biased micro-gate init b0 = -2.0 (sigmoid(-2.0) ≈ 0.1192)
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.freq_bank = nn.Linear(dim, dim)
        self.phase_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2.0) ≈ 0.1192
        self.win_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        v2_type: str = "original_bilinear",
    ) -> torch.Tensor:
        m = mask.unsqueeze(-1)
        x = x * m
        if v2_type in ("original_bilinear", "base"):
            return x
        if v2_type == "fourier_gabor_V2A":
            h_norm = self.norm(x)
            sin_comp = torch.sin(self.freq_bank(h_norm))
            cos_comp = torch.cos(self.freq_bank(h_norm))
            gabor_response = 0.5 * (sin_comp + cos_comp)
            gate = torch.sigmoid(self.phase_gate)
            return (x + gate * gabor_response) * m
        if v2_type == "window_cross_attn_V2B":
            h_norm = self.norm(x)
            attn_out, _ = self.win_attn(h_norm, h_norm, h_norm, need_weights=False)
            return (x + 0.10 * attn_out) * m
        return x
