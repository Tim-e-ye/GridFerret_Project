#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[SYM-GF-SLOT-V3] Slot V3: Manhattan Liberty-Cut Dual-Topology Contrast & Lattice Spring Loss (`topo_contrast_V3A`)
vs. Smooth-L1 Coordinate Regression (`smooth_l1_V3B`) vs. Standard CE (`original_ce`).
"""

from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def compute_slot_v3_loss(
    logits: torch.Tensor,
    pred_coords: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_coords: torch.Tensor,
    v3_type: str = "original_ce",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    @bce_contract
    slot_id: SYM-GF-SLOT-V3
    target_nail: Isotropic Euclidean Ball-Query Blindness to Orthogonal vs. Diagonal Manhattan Topology (Nail V-C)
    """
    ce_loss = F.cross_entropy(logits.reshape(-1, 3), gt_labels.reshape(-1))
    l1_loss = F.l1_loss(pred_coords, gt_coords)
    if v3_type in ("original_ce", "base"):
        total = ce_loss + 0.5 * l1_loss
        return total, {"L_ce": float(ce_loss.item()), "L_coord": float(l1_loss.item()), "L_topo": 0.0}
    if v3_type == "topo_contrast_V3A":
        b = pred_coords.shape[0]
        grid_xy = pred_coords.view(b, 19, 19, 2)
        dx = grid_xy[:, :, 1:, 0] - grid_xy[:, :, :-1, 0]
        dy = grid_xy[:, 1:, :, 1] - grid_xy[:, :-1, :, 1]
        topo_viol = F.relu(-dx + 0.02).mean() + F.relu(-dy + 0.02).mean()
        spring_reg = dx.var() + dy.var()
        topo_loss = topo_viol + 0.25 * spring_reg
        total = ce_loss + 0.5 * l1_loss + 0.35 * topo_loss
        return total, {"L_ce": float(ce_loss.item()), "L_coord": float(l1_loss.item()), "L_topo": float(topo_loss.item())}
    if v3_type == "smooth_l1_V3B":
        sl1 = F.smooth_l1_loss(pred_coords, gt_coords)
        total = ce_loss + 0.5 * sl1
        return total, {"L_ce": float(ce_loss.item()), "L_coord": float(sl1.item()), "L_topo": 0.0}
    return ce_loss + 0.5 * l1_loss, {"L_ce": float(ce_loss.item()), "L_coord": float(l1_loss.item()), "L_topo": 0.0}
