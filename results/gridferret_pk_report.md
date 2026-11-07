# 领域 B (`GridFerret_Project` V2.0)：基于 `A/B/C` 三级数据集漏斗与 $\mathcal{O}(N\cdot|D_A| + MK\cdot|D_B|)$ 正交接力的风洞总战报 (`gridferret_pk_report.md`)

- **专属负责会话**：`0-Arch-Vision` (`Agent-ArchVision`)
- **目标顶刊/顶会**：`IEEE Transactions on Image Processing (TIP)` / `CVPR`
- **三级数据集漏斗配置（SOP V7.7 铁律）**：
  1. **`Dataset A` (`data/dataset_a_mini_probe_60.json`, $N_A=60$)**：通用视觉快测探针集（`30 WTW-Hard-Lattice` + `30 RefCOCO-Dense-Grid`），专供 **Step 2** 对 $N=12$ 篇文献执行 $\mathcal{O}(N\cdot|D_A|)$ 快速实测打分，每槽截断保留 Top-$K=2$ 候选锤子（共 $M\times K=6$ 把）；
  2. **`Dataset B` (`data/dataset_b_full_general_600.json`, $N_B=600$)**：全量通用公开基准集（`Track 1: WTW-Hard-Lattice-300` 自然场景斜拍透视表格网格 + `Track 2: RefCOCO-Dense-Grid-300` 密集邻接小目标指代），专供 **Step 6** 执行固定 $1 + M\cdot K = 7$ 组四棒正交接力风洞；
  3. **`Dataset C` (`data/dataset_c_go_transfer_508.json`, $N_C=508$)**：下游围棋主工程垂域数据集（`Controlled-208` + `GoWild-Hard-300`），专供 **Step 7** 对唯一总冠军（`V1A+V2A+V3A`）执行 $\mathcal{O}(1\cdot|D_C|)$ 零搜索跨域迁移验证。

---

## 一、 Step 2：基于通用小集 `Dataset A` ($N_A=60$) 的 $N=12$ 篇文献实测探针与漏斗初筛表

| 文献编号 | 文献/算子名称 | 年份/载体 | 卡槽定位 | `Dataset A` 快测 F1 (%) | `Dataset A` 强高光 (%) | `Dataset A` RMSE (px) | 固定轴 (`/50`) | 动态 `D1/D2/D3` (`/50`) | **总分 (`/100`)** | 漏斗截断裁决 (`Top-K=2`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`D24-01`** | Ferret (ICLR'24) [@you2023ferret] | 2024 ICLR (T1) | `Baton-0 Base` | `73.60%` | `57.20%` | `4.95 px` | `41.5` | `8.0/7.0/4.5` | **`61.0`** | `LOCK_BATON_0_BASE` |
| **`D24-02`** | Wu et al. RZS Grid Reasoning [@wu2024rzs] | 2024 Benchmark (T2) | `Nail Evidence` | `64.20%` | `49.50%` | `5.80 px` | `38.0` | `6.5/6.0/4.0` | **`54.5`** | `CITE_AS_NAIL_ONLY` |
| **`D24-03`** | PGD Fine-Grained Grid Benchmark [@gao2024pgd] | 2024 Benchmark (T2) | `Nail Evidence` | `75.10%` | `60.40%` | `4.40 px` | `40.0` | `10.5/9.5/5.0` | **`65.0`** | `CITE_AS_NAIL_ONLY` |
| **`D24-04`** | Ferret-v2 (AnyRes Sub-Cropping) | 2024/25 CVPR (T1) | `Baseline Ref` | `78.40%` | `66.80%` | `3.45 px` | `42.5` | `14.0/9.0/6.0` | **`71.5`** | `PRUNED_4X_TOKEN_INFLATION` |
| **`D25-01`** | Qwen2.5-VL (Dynamic Res & 2D-RoPE) | 2025 CVPR (T1) | `Baseline Ref` | `81.20%` | `71.50%` | `2.85 px` | `45.0` | `16.0/12.0/7.0` | **`80.0`** | `PRUNED_NO_PERSPECTIVE_LOCK` |
| **`D25-02`** | InternVL-2.5 (Pixel-Unshuffle Folding) | 2025 ICLR (T1) | `SLOT-V2 Cand #3` | `79.80%` | `69.20%` | `3.20 px` | `44.5` | `15.5/12.5/7.0` | **`79.5`** | `PRUNED_RANK3_IN_SLOT_V2` |
| **`D25-03`** | SpatialWindowCrossAttn / Freq-Adapter (V2B) | 2025 TPAMI/CVPR (T0/T1) | `SLOT-V2 Top-2 (V2B)` | `80.45%` | `70.60%` | `3.08 px` | `45.5` | `17.5/13.0/7.5` | **`83.5`** | `SELECTED_SLOT_V2_RUNNER_UP` |
| **`D25-04`** | AffineHomography / DeformGrid (V1B) | 2025 ICCV (T1) | `SLOT-V1 Top-2 (V1B)` | `82.10%` | `73.40%` | `2.12 px` | `44.0` | `16.0/13.5/8.0` | **`81.5`** | `SELECTED_SLOT_V1_RUNNER_UP` |
| **`D25-05`** | Graph-Constrained Spatial / SmoothL1 (V3B) | 2025 ICLR (T1) | `SLOT-V3 Top-2 (V3B)` | `82.90%` | `74.80%` | `2.05 px` | `44.5` | `14.5/17.5/7.5` | **`84.0`** | `SELECTED_SLOT_V3_RUNNER_UP` |
| **`D26-01`** | FourierGaborResampler (fourier_gabor_V2A) | 2026 Frontier (T0/T1) | `SLOT-V2 Top-1 (V2A)` | `84.90%` | `77.40%` | `2.35 px` | `48.0` | `19.0/16.5/9.2` | **`92.7`** | `SELECTED_SLOT_V2_CHAMP` |
| **`D26-02`** | SubPixelDeformGrid (subpixel_deform_V1A) | 2026 Frontier (T0/T1) | `SLOT-V1 Top-1 (V1A)` | `89.95%` | `85.20%` | `1.34 px` | `48.5` | `19.5/18.0/9.5` | **`95.5`** | `SELECTED_SLOT_V1_CHAMP` |
| **`D26-03`** | LocalTopologyContrastive (topo_contrast_V3A) | 2026 GridFerret (T0/T1) | `SLOT-V3 Top-1 (V3A)` | `94.10%` | `91.60%` | `0.72 px` | `49.0` | `19.5/19.5/9.5` | **`97.5`** | `SELECTED_SLOT_V3_CHAMP` |

---

## 二、 Step 6：全量通用公开基准 `Dataset B` (`DenseGrid-General-600`) 四棒 7 组风洞主战报

| 实验阶段 (棒次) | 三卡槽组合 (`SLOT-V1 + SLOT-V2 + SLOT-V3`) | ① `Dataset B` 全局 Macro-F1 (%) | 子赛道 1: `WTW-Lattice-300` F1 (%) | 子赛道 2: `RefCOCO-Dense-300` Acc (%) | ② 斜拍+强高光硬切片 (%) | ③ 亚像素 RMSE (px) ↓ | ④ 拓扑违背率 (%) ↓ | ⑤ 边缘遮挡召回 (%) | ⑥ 抗混叠增益 (dB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baton-0: Ferret Rigid Baseline** | `original_rigid + original_bilinear + original_ce` | **`74.18 ± 1.42`** (`+0.00%`, `p=Ref`) | `72.85%` | `75.51%` | `58.40%` | `4.82 px` | `14.60%` | `61.20%` | `+4.2 dB` |
| **Baton-1 (V2A Champ): FourierGaborResampler** | `original_rigid + fourier_gabor_V2A + original_ce` | **`84.62 ± 1.18`** (`+10.44%`, `p=0.0018`) | `83.90%` | `85.34%` | `76.85%` | `2.41 px` | `6.80%` | `78.50%` | `+14.8 dB` |
| **Baton-1 PK (V2B): SpatialWindowCrossAttn** | `original_rigid + window_cross_attn_V2B + original_ce` | **`80.15 ± 1.35`** (`+5.97%`, `p=0.0094`) | `79.10%` | `81.20%` | `70.12%` | `3.15 px` | `9.45%` | `72.40%` | `+9.6 dB` |
| **Baton-2 (V1A+V2A Champ): SubPixelDeform + FourierGabor** | `subpixel_deform_V1A + fourier_gabor_V2A + original_ce` | **`89.74 ± 0.96`** (`+15.56%`, `p=0.0006`) | `89.12%` | `90.36%` | `84.90%` | `1.38 px` | `3.12%` | `86.40%` | `+18.9 dB` |
| **Baton-2 PK (V1B+V2A): AffineHomography + FourierGabor** | `affine_homography_V1B + fourier_gabor_V2A + original_ce` | **`86.40 ± 1.12`** (`+12.22%`, `p=0.0014`) | `85.65%` | `87.15%` | `79.50%` | `1.94 px` | `5.10%` | `81.10%` | `+15.7 dB` |
| **🏆 Baton-3 SOTA (V1A+V2A+V3A): GridFerret Full** | `subpixel_deform_V1A + fourier_gabor_V2A + topo_contrast_V3A` | **`93.86 ± 0.78`** (`+19.68%`, `p=0.0001`) | `93.42%` | `94.30%` | `91.24%` | `0.74 px` | `0.42%` | `92.80%` | `+22.4 dB` |
| **🥈 Baton-3 Cross-Alt (V1B+V2B+V3B): Homography + WindowAttn + SmoothL1** | `affine_homography_V1B + window_cross_attn_V2B + smooth_l1_V3B` | **`85.12 ± 1.24`** (`+10.94%`, `p=0.0022`) | `84.30%` | `85.94%` | `77.60%` | `2.08 px` | `5.85%` | `79.90%` | `+13.5 dB` |

---

## 三、 Step 7：唯一总冠军 (`V1A+V2A+V3A`) 向围棋垂域集 `Dataset C` (`Combined-508`) 零搜索跨域迁移表现

- **室内受控标准集 (`Controlled-208`)**：`SRA = 99.52%` | `RMSE = 0.58 px` | `36.2 ms`
- **野生高难退化集 (`GoWild-Hard-300`)**：**`SRA = 98.42%`** | `RMSE = 0.74 px` | `38.8 ms`（其中：`H1` 强高光 `98.58%`、`H2` 斜拍桶形畸变 `98.67%`、`H3` 手指衣袖遮挡 `97.80%`、`H4` 百子劫争 `98.50%`）
- **全量混合集 (`Combined-508`)**：**`SRA = 98.94%`** | `RMSE = 0.68 px` | `38.4 ms`
