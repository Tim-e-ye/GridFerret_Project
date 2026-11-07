#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared tensor compatibility & 4-Gate Impedance-Matching base for GridFerret `models/hammer_slots/`."""

from __future__ import annotations

import torch
import torch.nn as nn


class HammerRedLineViolationError(RuntimeError):
    """Raised when a candidate vision hammer card violates a locked reviewer red-line constraint."""


def assert_zero_mask_leakage(clean_out: torch.Tensor, noisy_out: torch.Tensor, tol: float = 1e-6) -> float:
    """Gate 1: Verify zero glare/occlusion mask leakage across spatial resampling & topology layers."""
    leak = float((clean_out - noisy_out).abs().max().item())
    if leak > tol:
        raise HammerRedLineViolationError(f"Gate 1 violation: mask leakage {leak:.2e} > {tol:.2e}")
    return leak


def preserve_manifold_norm(h_mod: torch.Tensor, h_ref: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Gate 3: Preserve pre-trained visual token L2 norm distribution to prevent LLM projector saturation."""
    ref_norm = h_ref.norm(dim=-1, keepdim=True).clamp(min=eps)
    mod_norm = h_mod.norm(dim=-1, keepdim=True).clamp(min=eps)
    return h_mod * (ref_norm / mod_norm)
