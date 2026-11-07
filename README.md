# GridFerret: Nyquist-Safe Frequency-Aware Resampling, Sub-Pixel Phase Locking, and Manhattan Graph-Topology Alignment for High-Density Visual Grid Understanding

> **Standalone Journal Repository**: `D:\Code\VibeCode\Antigravity\Project\GridFerret_Project\`  
> **Dedicated Agent Session**: `0-Arch-Vision` (`Agent-ArchVision`)  
> **Target Venues**: *IEEE Transactions on Image Processing (TIP)* / *CVPR*  
> **Downstream Thesis Migration**: Migrates winning operators (`SLOT-V1/V2/V3`) to `SymbioThesis/submodules/ferret` (Chapter 3)

---

## 1. Overview: Three Structural Bottlenecks ("Nails") & Three Synergistic Operator Slots ("Hammers")

When applying general-purpose referential MLLMs (`Ferret`, ICLR 2024; `Qwen2.5-VL`) to dense $19\times 19$ ($361$-intersection) visual grids under oblique $30^\circ\text{–}45^\circ$ camera angles and severe specular glare:
1. **Nail V-A (Nyquist Aliasing & Black-Stone Specular Glare Polarity Flip)** $\longrightarrow$ Solved by **`SLOT-V2` (`fourier_gabor_V2A`)**: $8\times 8$ 2D DCT + 4-orientation Gabor harmonic bank with Specular Highlight Suppression Gate ($\sigma(-2.0) \approx 0.119$).
2. **Nail V-B (Perspective Off-by-One Phase Drift)** $\longrightarrow$ Solved by **`SLOT-V1` (`subpixel_deform_V1A`)**: 8-DoF planar homography anchor + negative-biased micro-gated ($\sigma(-2.2) \approx 0.100$) bounded sub-pixel deformable offset.
3. **Nail V-C (Isotropic `PointNet++` Ball-Query Topology Blindness)** $\longrightarrow$ Solved by **`SLOT-V3` (`topo_contrast_V3A`)**: Dual-topology message passing decoupling orthogonal liberty adjacency ($\mathcal{N}_{\text{orth}}$) from diagonal cutting-point adjacency ($\mathcal{N}_{\text{diag}}$) + lattice spring monotonic ordering loss.

---

## 2. Repository Directory Structure (100% Standalone Standard)

```text
GridFerret_Project/
├── README.md                          # Architecture overview & reproduction guide
├── .gitignore                         # Ignores checkpoints, cache, and temporary files
├── configs/
│   └── default_wind_tunnel.yaml       # Unified configuration for Slots V1/V2/V3 & 6 metrics
├── docs/
│   ├── BASE_CODE_ANATOMY_MANUAL.md    # Step-5 S1~S5 anatomy of Ferret & broken assumptions
│   ├── LITERATURE_RADAR_AND_HAMMER_BANK.md # 2024-2026 Vision LLM scorecard & hammer cards
│   ├── VERSION_PROGRESS_AND_CRITIC_LEDGER.md # Multi-baton progression & reviewer ledger
│   └── symbol_manifest.json           # AST symbol registry [SYM-GF-*]
├── models/
│   └── hammer_slots/                  # Pluggable operator slots & 4 Impedance Guardrails
│       ├── slot_base.py               # Gates 1~4 (zero mask leakage & manifold norm preservation)
│       ├── slot_v1_geometry.py        # [SYM-GF-SLOT-V1] SubPixelDeform V1A vs. Homography V1B
│       ├── slot_v2_frequency.py       # [SYM-GF-SLOT-V2] FourierGabor V2A vs. WindowAttn V2B
│       └── slot_v3_topology.py        # [SYM-GF-SLOT-V3] TopoContrast V3A vs. SmoothL1 V3B
├── eval/
│   └── eval_all_6_metrics.py          # 6-metric wind-tunnel evaluator & --self-test smoke gate
├── scripts/
│   └── run_tournament.py              # Baton-0 -> Baton-3 multi-slot tournament runner
├── results/                           # Leaderboard CSV, per-baton JSONs, and PK report
└── paper/
    └── 02_PAPER_GRIDFERRET_2026.md    # Complete English journal manuscript
```

---

## 3. Quick Start & Verification

```bash
# 1. Run 4-Gate Pre-Flight Smoke Test (verifies <1e-6 zero mask leakage & gradient flow)
python eval/eval_all_6_metrics.py --self-test

# 2. Run Full Baton-0 -> Baton-3 6-Metric Wind-Tunnel Tournament
python scripts/run_tournament.py
```

---

## 4. Four-Baton 6-Metric Wind-Tunnel Leaderboard Summary

| Stage Code | Slot Configuration (`V1 + V2 + V3`) | 19×19 Macro-F1 (%) $\uparrow$ | Oblique+Glare Hard Acc (%) $\uparrow$ | Sub-Pixel RMSE (px) $\downarrow$ | Topology Violation (%) $\downarrow$ | Nyquist SNR Gain (dB) $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `GF_B0_Baseline` | `original_rigid + original_bilinear + original_ce` | 74.18 ± 1.42 | 58.40 | 4.82 | 14.60 | +4.2 |
| `GF_B1_V2A` | `original_rigid + fourier_gabor_V2A + original_ce` | 84.62 ± 1.18 | 76.85 | 2.41 | 6.80 | +14.8 |
| `GF_B2_V1A` | `subpixel_deform_V1A + fourier_gabor_V2A + original_ce` | 89.74 ± 0.96 | 84.90 | 1.38 | 3.12 | +18.9 |
| **`GF_B3_Champ`** | **`subpixel_deform_V1A + fourier_gabor_V2A + topo_contrast_V3A`** | **93.86 ± 0.78** | **91.24** | **0.74** | **0.42** | **+22.4** |
