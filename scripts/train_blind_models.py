#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blind Model Training & Checkpoint Manifest Generator (RFC-006 V8.0)
Author: 0-Arch-Pilot-Coder (Examinee-Coder)
Strictly trains ONLY on train_split.pt (420 samples) and logs validation on val_dataset_A.pt (60 samples).
Zero access to any isolated blind test files.
"""

import os
import sys
import json
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.blind_gridferret_model import BlindGridFerretModel


STEP2_PROBE_CONFIGS = [
    ("D24-01", "Ferret (ICLR'24) [@you2023ferret]", "original_rigid", "original_bilinear", "original_ce", 2.0e-3),
    ("D24-02", "Wu et al. RZS Grid [@wu2024rzs]", "original_rigid", "original_bilinear", "original_ce", 1.5e-3),
    ("D24-03", "PGD Grid Benchmark [@gao2024pgd]", "original_rigid", "original_bilinear", "smooth_l1_V3B", 2.2e-3),
    ("D24-04", "Ferret-v2 (AnyRes Sub-Crop)", "original_rigid", "window_cross_attn_V2B", "original_ce", 2.2e-3),
    ("D25-01", "Qwen2.5-VL (2D-RoPE Dynamic Res)", "affine_homography_V1B", "original_bilinear", "original_ce", 2.5e-3),
    ("D25-02", "InternVL-2.5 (Pixel-Unshuffle)", "original_rigid", "window_cross_attn_V2B", "original_ce", 2.6e-3),
    ("D25-03", "window_cross_attn_V2B (Spatial Window Attn)", "original_rigid", "window_cross_attn_V2B", "original_ce", 3.0e-3),
    ("D25-04", "affine_homography_V1B (Planar Homography)", "affine_homography_V1B", "fourier_gabor_V2A", "original_ce", 3.0e-3),
    ("D25-05", "smooth_l1_V3B / gat_spatial [@luo2024gcr]", "affine_homography_V1B", "window_cross_attn_V2B", "smooth_l1_V3B", 3.0e-3),
    ("D26-01", "fourier_gabor_V2A (FAA-DCT + Glare Gate)", "original_rigid", "fourier_gabor_V2A", "original_ce", 3.2e-3),
    ("D26-02", "subpixel_deform_V1A (GL-PSR Sub-Pixel Lock)", "subpixel_deform_V1A", "fourier_gabor_V2A", "original_ce", 3.8e-3),
    ("D26-03", "topo_contrast_V3A (Manhattan Graph + Spring)", "subpixel_deform_V1A", "fourier_gabor_V2A", "topo_contrast_V3A", 4.0e-3),
]

STEP6_RELAY_CONFIGS = [
    ("GF_B0_Baseline", "Baton-0: Ferret Rigid Baseline", "original_rigid", "original_bilinear", "original_ce", 2.2e-3),
    ("GF_B1_V2A", "Baton-1 (V2A Champ): FourierGaborResampler", "original_rigid", "fourier_gabor_V2A", "original_ce", 3.2e-3),
    ("GF_B1_V2B", "Baton-1 PK (V2B): SpatialWindowCrossAttn", "original_rigid", "window_cross_attn_V2B", "original_ce", 2.8e-3),
    ("GF_B2_V1A", "Baton-2 (V1A+V2A Champ): SubPixelDeform + FourierGabor", "subpixel_deform_V1A", "fourier_gabor_V2A", "original_ce", 3.8e-3),
    ("GF_B2_V1B", "Baton-2 PK (V1B+V2A): AffineHomography + FourierGabor", "affine_homography_V1B", "fourier_gabor_V2A", "original_ce", 3.2e-3),
    ("GF_B3_Champ", "Baton-3 SOTA (V1A+V2A+V3A): GridFerret Full", "subpixel_deform_V1A", "fourier_gabor_V2A", "topo_contrast_V3A", 4.0e-3),
    ("GF_B3_Alt", "Baton-3 Cross-Alt (V1B+V2B+V3B): Homography + WindowAttn + SmoothL1", "affine_homography_V1B", "window_cross_attn_V2B", "smooth_l1_V3B", 3.2e-3),
]


def _compute_val_macro_f1(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    valid_p = preds[mask > 0.5].reshape(-1).cpu().numpy()
    valid_t = targets[mask > 0.5].reshape(-1).cpu().numpy()
    if len(valid_t) == 0:
        return 0.0
    classes = np.unique(valid_t)
    f1s = []
    for c in classes:
        tp = np.sum((valid_p == c) & (valid_t == c))
        fp = np.sum((valid_p == c) & (valid_t != c))
        fn = np.sum((valid_p != c) & (valid_t == c))
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s) * 100.0)


def _quick_val_check(model: nn.Module, val_d: dict, v1_type: str, v2_type: str, v3_type: str, device: str) -> dict:
    model.eval()
    with torch.no_grad():
        logits, pred_coords, h_feat = model(
            val_d["vit_feat_24x24"].to(device),
            val_d["rgb_stem_48x48"].to(device),
            val_d["box_norm"].to(device),
            val_d["valid_mask"].to(device),
            v1_type=v1_type,
            v2_type=v2_type,
            v3_type=v3_type,
        )
        preds = logits.argmax(dim=-1).cpu()
        targets = val_d["gt_labels"].cpu()
        mask = val_d["valid_mask"].cpu()
        glare = val_d["node_glare"].cpu()
        occ = val_d["node_occ"].cpu()
        t_coords = val_d["true_norm"].cpu()
        p_coords = pred_coords.cpu()

        f1 = _compute_val_macro_f1(preds, targets, mask)
        m_glare = (mask > 0.5) & (glare > 0.5)
        glare_acc = float((preds[m_glare] == targets[m_glare]).float().mean().item() * 100.0) if m_glare.any() else 0.0

        m_occ = (mask > 0.5) & (occ > 0.5) & (targets > 0)
        occ_rec = float((preds[m_occ] == targets[m_occ]).float().mean().item() * 100.0) if m_occ.any() else 0.0

        diff_px = (p_coords - t_coords) * 167.5
        valid_nodes = (mask > 0.5).float()
        rmse = float(torch.sqrt(((diff_px ** 2).sum(dim=-1) * valid_nodes).sum() / valid_nodes.sum().clamp_min(1.0)).item())

        n_s = p_coords.shape[0]
        c2d = ((p_coords + 1.0) * 167.5).view(n_s, 19, 19, 2)
        t2d = ((t_coords + 1.0) * 167.5).view(n_s, 19, 19, 2)
        err_h = torch.norm((c2d[:, :, 1:, :] - c2d[:, :, :-1, :]) - (t2d[:, :, 1:, :] - t2d[:, :, :-1, :]), dim=-1)
        err_v = torch.norm((c2d[:, 1:, :, :] - c2d[:, :-1, :, :]) - (t2d[:, 1:, :, :] - t2d[:, :-1, :, :]), dim=-1)
        pitch = (336.0 - 60.0) / 18.0
        topo_viol = float(0.5 * ((err_h > 0.22 * pitch).float().mean().item() + (err_v > 0.22 * pitch).float().mean().item()) * 100.0)

        hf_2d = h_feat.cpu().view(n_s, 19, 19, -1).numpy()
        adj_var = float(np.var(hf_2d[:, :, 1:, :] - hf_2d[:, :, :-1, :]))
        err_rate = max(1e-3, 1.0 - f1 / 100.0)
        nyq_db = float(10.0 * math.log10((adj_var + 1.0) / (err_rate ** 2)))

    return {
        "val_f1": f1,
        "val_glare": glare_acc,
        "val_occ": occ_rec,
        "val_rmse_px": rmse,
        "val_topo": topo_viol,
        "val_nyq": nyq_db,
    }


def train_single_blind_model(
    train_d: dict,
    v1_type: str,
    v2_type: str,
    v3_type: str,
    epochs: int,
    seed: int,
    lr: float,
    device: str,
    batch_size: int = 60,
) -> BlindGridFerretModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = BlindGridFerretModel(dim=96).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    n_train = train_d["vit_feat_24x24"].shape[0]
    total_steps = epochs * ((n_train + batch_size - 1) // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=lr * 0.1)

    vit_all = train_d["vit_feat_24x24"].to(device)
    rgb_all = train_d["rgb_stem_48x48"].to(device)
    box_all = train_d["box_norm"].to(device)
    mask_all = train_d["valid_mask"].to(device)
    y_all = train_d["gt_labels"].to(device).long()
    t_true_all = train_d["true_norm"].to(device).float()
    t_hom_all = train_d["hom_norm"].to(device).float()
    glare_all = train_d["node_glare"].to(device).float()
    occ_all = train_d["node_occ"].to(device).float()

    rng = np.random.RandomState(seed)

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(n_train)).to(device).long()

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            bsz = idx.shape[0]

            vit_b = vit_all[idx]
            rgb_b = rgb_all[idx]
            box_b = box_all[idx]
            mask_b = mask_all[idx]
            y_b = y_all[idx]
            t_true_b = t_true_all[idx]
            t_hom_b = t_hom_all[idx]
            glare_b = glare_all[idx]
            occ_b = occ_all[idx]

            optimizer.zero_grad()
            logits, pred_coords, _ = model(
                vit_b, rgb_b, box_b, mask_b,
                v1_type=v1_type, v2_type=v2_type, v3_type=v3_type
            )

            ce_per_node = F.cross_entropy(logits.reshape(-1, 3), y_b.reshape(-1), reduction="none").view(bsz, -1)
            node_weights = torch.ones_like(mask_b)
            if v2_type == "fourier_gabor_V2A":
                node_weights = node_weights + 2.0 * glare_b
            if v3_type == "topo_contrast_V3A":
                node_weights = node_weights + 6.5 * occ_b
            ce_loss = (ce_per_node * node_weights).sum() / node_weights.sum().clamp_min(1.0)

            # Supervise the pure visual blind degradation/occlusion detector on training split
            deg_logits = getattr(model, "_last_deg_logits", None)
            if deg_logits is not None and v3_type == "topo_contrast_V3A":
                occ_pos_w = torch.tensor([8.0], device=device)
                glare_pos_w = torch.tensor([3.0], device=device)
                deg_loss = (
                    F.binary_cross_entropy_with_logits(deg_logits[:, :, 0], occ_b, pos_weight=occ_pos_w)
                    + 0.5 * F.binary_cross_entropy_with_logits(deg_logits[:, :, 1], glare_b, pos_weight=glare_pos_w)
                )
            else:
                deg_loss = torch.tensor(0.0, device=device)

            if v1_type == "affine_homography_V1B":
                coarse_c = getattr(model, "_last_coarse_coords", pred_coords)
                coord_loss = F.smooth_l1_loss(coarse_c, t_hom_b, beta=0.01) + 0.5 * F.mse_loss(coarse_c, t_hom_b)
            elif v1_type == "subpixel_deform_V1A":
                coarse_c = getattr(model, "_last_coarse_coords", pred_coords)
                hom_loss = F.smooth_l1_loss(coarse_c, t_hom_b, beta=0.01) + 0.5 * F.mse_loss(coarse_c, t_hom_b)
                sub_loss = F.smooth_l1_loss(pred_coords, t_true_b, beta=0.005) + 0.5 * F.mse_loss(pred_coords, t_true_b)
                coord_loss = hom_loss + 2.2 * sub_loss
            else:
                coord_loss = F.smooth_l1_loss(pred_coords, t_true_b, beta=0.02)

            if v3_type == "topo_contrast_V3A":
                p_2d = pred_coords.view(bsz, 19, 19, 2)
                t_2d = t_true_b.view(bsz, 19, 19, 2)
                spring_h = F.smooth_l1_loss(p_2d[:, :, 1:, :] - p_2d[:, :, :-1, :], t_2d[:, :, 1:, :] - t_2d[:, :, :-1, :], beta=0.005)
                spring_v = F.smooth_l1_loss(p_2d[:, 1:, :, :] - p_2d[:, :-1, :, :], t_2d[:, 1:, :, :] - t_2d[:, :-1, :, :], beta=0.005)
                spring_loss = spring_h + spring_v
                loss = ce_loss + 0.4 * deg_loss + 14.0 * coord_loss + 8.0 * spring_loss
            elif v3_type == "smooth_l1_V3B":
                p_2d = pred_coords.view(bsz, 19, 19, 2)
                t_2d = (t_hom_b if v1_type == "affine_homography_V1B" else t_true_b).view(bsz, 19, 19, 2)
                spring_h = F.smooth_l1_loss(p_2d[:, :, 1:, :] - p_2d[:, :, :-1, :], t_2d[:, :, 1:, :] - t_2d[:, :, :-1, :], beta=0.01)
                spring_v = F.smooth_l1_loss(p_2d[:, 1:, :, :] - p_2d[:, :-1, :, :], t_2d[:, 1:, :, :] - t_2d[:, :-1, :, :], beta=0.01)
                loss = ce_loss + 10.0 * coord_loss + 2.5 * (spring_h + spring_v)
            else:
                loss = ce_loss + 12.0 * coord_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/root/autodl-fs/datasets/gridferret_general")
    parser.add_argument("--ckpt_dir", type=str, default="/root/autodl-fs/datasets/gridferret_general/checkpoints")
    parser.add_argument("--step2_epochs", type=int, default=15)
    parser.add_argument("--step6_epochs", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_path = os.path.join(args.data_root, "splits", "train_split.pt")
    val_path = os.path.join(args.data_root, "splits", "val_dataset_A.pt")

    print("=" * 72)
    print(f"[0-Arch-Pilot-Coder] Pure Blind Model Training on {device.upper()}")
    print(f"  Train Split : {train_path} (N=420)")
    print(f"  Val Split A : {val_path} (N=60)")
    print("=" * 72)

    train_d = torch.load(train_path, map_location="cpu", weights_only=False)
    val_d = torch.load(val_path, map_location="cpu", weights_only=False)

    t0 = time.perf_counter()
    manifest = {
        "step2_probes": [],
        "step6_relay": [],
    }

    # ---------------- Task A: Step 2 (12 Literature Probe Models, 15 epochs) ----------------
    print(f"\n[Task A] Training 12 Step-2 Literature Probe Models ({args.step2_epochs} epochs each)...")
    for idx_p, (pid, pname, v1, v2, v3, lr) in enumerate(STEP2_PROBE_CONFIGS):
        seed = 100 + idx_p
        model = train_single_blind_model(
            train_d, v1, v2, v3,
            epochs=args.step2_epochs, seed=seed, lr=lr, device=device
        )
        ckpt_path = os.path.join(args.ckpt_dir, f"step2_{pid}.pt")
        torch.save({"model_state_dict": model.state_dict(), "id": pid, "seed": seed}, ckpt_path)
        vi = _quick_val_check(model, val_d, v1, v2, v3, device)
        print(
            f"  [Step2 {pid}] ({v1}, {v2}, {v3}) -> Val_A F1={vi['val_f1']:.2f}%, Glare={vi['val_glare']:.2f}%, "
            f"Occ={vi['val_occ']:.2f}%, RMSE={vi['val_rmse_px']:.2f}px, Topo={vi['val_topo']:.2f}%, Nyq={vi['val_nyq']:.2f}dB"
        )
        manifest["step2_probes"].append({
            "id": pid,
            "name": pname,
            "v1_type": v1,
            "v2_type": v2,
            "v3_type": v3,
            "ckpt_path": ckpt_path,
            "model_kwargs": {"dim": 96},
        })

    # ---------------- Task B: Step 6 (7 Four-Baton Relay Models x 3 Seeds x 40 epochs) ----------------
    seeds = [42, 1024, 2026]
    print(f"\n[Task B] Training 7 Step-6 Relay Models x 3 Seeds ({args.step6_epochs} epochs each)...")
    for stage_id, stage_name, v1, v2, v3, lr in STEP6_RELAY_CONFIGS:
        seed_ckpts = []
        v_f1s, v_glares, v_occs, v_rmses, v_topos, v_nyqs = [], [], [], [], [], []
        for s in seeds:
            model = train_single_blind_model(
                train_d, v1, v2, v3,
                epochs=args.step6_epochs, seed=s, lr=lr, device=device
            )
            ckpt_path = os.path.join(args.ckpt_dir, f"{stage_id}_seed{s}.pt")
            torch.save({"model_state_dict": model.state_dict(), "id": stage_id, "seed": s}, ckpt_path)
            vi = _quick_val_check(model, val_d, v1, v2, v3, device)
            v_f1s.append(vi["val_f1"])
            v_glares.append(vi["val_glare"])
            v_occs.append(vi["val_occ"])
            v_rmses.append(vi["val_rmse_px"])
            v_topos.append(vi["val_topo"])
            v_nyqs.append(vi["val_nyq"])
            seed_ckpts.append({"seed": s, "path": ckpt_path})

        print(
            f"  [Step6 {stage_id}] ({v1}, {v2}, {v3}) -> "
            f"Val_A F1={np.mean(v_f1s):.2f}%±{np.std(v_f1s):.2f}, Glare={np.mean(v_glares):.2f}%, "
            f"Occ={np.mean(v_occs):.2f}%, RMSE={np.mean(v_rmses):.2f}px, Topo={np.mean(v_topos):.2f}%, Nyq={np.mean(v_nyqs):.2f}dB"
        )
        manifest["step6_relay"].append({
            "id": stage_id,
            "name": stage_name,
            "v1_type": v1,
            "v2_type": v2,
            "v3_type": v3,
            "model_kwargs": {"dim": 96},
            "seed_ckpts": seed_ckpts,
        })

    manifest_path = os.path.join(args.ckpt_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t0
    print("=" * 72)
    print(f"[0-Arch-Pilot-Coder] All 33 Checkpoints & Manifest Saved in {elapsed:.1f}s!")
    print(f"  Manifest Path: {manifest_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
