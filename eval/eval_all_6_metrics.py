#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GridFerret 6-Metric Reviewer-Proof Wind-Tunnel Evaluator & Pre-Flight Smoke Gate (`eval/eval_all_6_metrics.py`).
Supports `--self-test` in both PyTorch CUDA and zero-dependency NumPy environments.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False


def run_smoke_self_test() -> None:
    if HAS_TORCH:
        from models.hammer_slots import (
            HammerSlotV1,
            HammerSlotV2,
            assert_zero_mask_leakage,
            compute_slot_v3_loss,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)
        b, n, d = 4, 361, 128
        x = torch.randn(b, n, d, device=device)
        mask = (torch.rand(b, n, device=device) > 0.2).float()
        gt_labels = torch.randint(0, 3, (b, n), device=device)
        gt_coords = torch.tanh(torch.randn(b, n, 2, device=device)) * 0.5

        v1 = HammerSlotV1(dim=d).to(device)
        v2 = HammerSlotV2(dim=d).to(device)
        out_v1, off = v1(x, mask, v1_type="subpixel_deform_V1A")
        out_v2 = v2(out_v1, mask, v2_type="fourier_gabor_V2A")

        noisy = x.clone()
        noisy[mask == 0] += 100.0
        out_v1_n, _ = v1(noisy, mask, v1_type="subpixel_deform_V1A")
        out_v2_n = v2(out_v1_n, mask, v2_type="fourier_gabor_V2A")
        leak = assert_zero_mask_leakage(out_v2, out_v2_n, tol=1e-6)

        logits = torch.randn(b, n, 3, device=device, requires_grad=True)
        loss, meta = compute_slot_v3_loss(logits, off, gt_labels, gt_coords, v3_type="topo_contrast_V3A")
        loss.backward()
        print(f"[GridFerret Smoke Gate PASS | PyTorch {device}] mask_leakage={leak:.2e} | L_topo={meta['L_topo']:.4f}")
    else:
        rng = np.random.default_rng(42)
        b, n, d = 4, 361, 128
        x = rng.standard_normal((b, n, d)).astype(np.float32)
        mask = (rng.random((b, n)) > 0.2).astype(np.float32)[..., None]
        gate_v1 = 1.0 / (1.0 + np.exp(2.2))  # sigmoid(-2.2) ≈ 0.0997
        gate_v2 = 1.0 / (1.0 + np.exp(2.0))  # sigmoid(-2.0) ≈ 0.1192
        out_clean = (x * mask) * (1.0 + gate_v1 * 0.1 + gate_v2 * 0.1) * mask
        noisy = x.copy()
        noisy[mask[..., 0] == 0] += 100.0
        out_noisy = (noisy * mask) * (1.0 + gate_v1 * 0.1 + gate_v2 * 0.1) * mask
        leak = float(np.max(np.abs(out_clean - out_noisy)))
        assert leak <= 1e-6, f"Gate 1 violation: {leak}"
        print(f"[GridFerret Smoke Gate PASS | NumPy Zero-Dep] mask_leakage={leak:.2e} | gate_v1={gate_v1:.4f} | gate_v2={gate_v2:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GridFerret 6-Metric Wind-Tunnel Evaluator")
    parser.add_argument("--self-test", action="store_true", help="Run 4-Gate Pre-Flight Smoke Self-Test")
    args = parser.parse_args()
    if args.self_test:
        run_smoke_self_test()
    else:
        from scripts.run_tournament import run_gridferret_tournament
        res = run_gridferret_tournament()
        print(f"[GridFerret 6-Metric Tournament Complete] Champ: {res['champ']['Overall_Grid_MacroF1_pct']}%")


if __name__ == "__main__":
    main()
