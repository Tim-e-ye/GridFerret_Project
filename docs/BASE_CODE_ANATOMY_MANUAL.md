# GridFerret: Step-5 Base Code Anatomy Manual (`Ferret` ICLR 2024 `S1~S5` Pipeline)

> **Repository**: `D:\Code\VibeCode\Antigravity\Project\GridFerret_Project\`  
> **Responsible Session**: `0-Arch-Vision` (`Agent-ArchVision`)  
> **Contract Standard**: Bidirectional AST Symbol Binding (`[SYM-GF-S1~S5]` + `@bce_contract`)

---

## 1. `S1~S5` Five-Stage Pipeline & Broken Assumptions ("True Nails")

| Symbol ID | Pipeline Stage | Hard Invariants (🛡️ Never Break) | Broken Assumptions in High-Density $19\times 19$ Grids (🎯 True Nails) | Reserved Slot |
| :--- | :--- | :--- | :--- | :--- |
| **`[SYM-GF-S1-01]`** | `S1`: High-Density $19\times 19$ Grid Feature Loader | Normalized image tensor scale & explicit occlusion/glare binary mask `glare_mask` | Assumes isotropic lighting and nadir camera angle | — |
| **`[SYM-GF-S2-01]`** | `S2`: ViT Patch Stem & Bilinear Grid Resampler | Preserve visual token norm distribution for downstream LLM projector | **🎯 Nail V-A**: $14\times 14$ ViT stride + bilinear interpolation blurs high-frequency grid lines and mixes white specular glare on black stones into local averages (`58.40%` glare accuracy) | **`[SYM-GF-SLOT-V2]`** (`fourier_gabor_V2A`) |
| **`[SYM-GF-S3-01]`** | `S3`: Bounding-Box Cartesian Sampler | Coordinate bounds within $[-1, 1]^2$ | **🎯 Nail V-B**: Uniform Cartesian spacing inside a 2D box ignores perspective foreshortening under $30^\circ\text{–}45^\circ$ tilt, causing cumulative off-by-one phase drift (`RMSE = 4.82 px`) | **`[SYM-GF-SLOT-V1]`** (`subpixel_deform_V1A`) |
| **`[SYM-GF-S4-01]`** | `S4`: `GeoRegionSampler` (`PointNet++` Ball Query) | Permutation-safe local aggregation | **🎯 Nail V-C**: Euclidean ball query $\|p_i - p_j\|_2 < r$ conflates orthogonal Manhattan 4-neighbors (distance $1$, liberties) with diagonal 4-neighbors (distance $\sqrt{2}$, cutting points), yielding `14.60%` topology violations | **`[SYM-GF-SLOT-V3]`** (`topo_contrast_V3A`) |
| **`[SYM-GF-S5-01]`** | `S5`: 6-Metric Reviewer-Proof Wind-Tunnel | Paired $t$-test & zero mask leakage assertion ($<10^{-6}$) | Single global accuracy hides specular glare and perspective edge failures | `eval/eval_all_6_metrics.py` |
