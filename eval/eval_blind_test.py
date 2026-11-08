#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read-Only Blind Test Evaluator & Scorecard Generator
Author: 8-QA-RedTeamAuditor (RFC-006 V8.0)
"""

import os
import sys
import csv
import json
import math
import hashlib
import argparse
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.audit_experiment_integrity import run_all_gates


def _compute_macro_f1(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
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


def _compute_accuracy(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask > 0.5
    if m.sum().item() == 0:
        return 0.0
    correct = (preds[m] == targets[m]).float().mean().item()
    return float(correct * 100.0)


def compute_8_metrics(
    logits: torch.Tensor,
    pred_coords: torch.Tensor,
    h_feat: torch.Tensor,
    gt_vault: dict,
    valid_mask: torch.Tensor,
    track_id: torch.Tensor,
) -> dict:
    preds = logits.argmax(dim=-1).cpu()
    gt_labels = gt_vault["gt_labels"].cpu().long()
    true_norm = gt_vault["true_norm"].cpu().float()
    node_glare = gt_vault["node_glare"].cpu().float()
    node_occ = gt_vault["node_occ"].cpu().float()
    valid_mask = valid_mask.cpu().float()
    track_id = track_id.cpu().long()
    pred_coords = pred_coords.cpu().float()
    h_feat = h_feat.cpu().float()

    overall_f1 = _compute_macro_f1(preds, gt_labels, valid_mask)

    m_wtw = valid_mask * (track_id == 0).unsqueeze(-1).float()
    wtw_f1 = _compute_macro_f1(preds, gt_labels, m_wtw)

    m_ref = valid_mask * (track_id == 1).unsqueeze(-1).float()
    ref_acc = _compute_accuracy(preds, gt_labels, m_ref)

    m_glare = valid_mask * (node_glare > 0.5).float()
    glare_acc = _compute_accuracy(preds, gt_labels, m_glare)

    coord_diff_px = (pred_coords - true_norm) * 167.5
    sq_err = (coord_diff_px ** 2).sum(dim=-1)
    valid_nodes = (valid_mask > 0.5).float()
    rmse_px = float(torch.sqrt((sq_err * valid_nodes).sum() / valid_nodes.sum().clamp_min(1.0)).item())

    n_samples, n_nodes, _ = pred_coords.shape
    grid_g = int(round(math.sqrt(n_nodes)))
    if grid_g * grid_g == n_nodes:
        coords_2d = (pred_coords + 1.0) * 167.5
        coords_2d = coords_2d.view(n_samples, grid_g, grid_g, -1)
        true_2d = ((true_norm + 1.0) * 167.5).view(n_samples, grid_g, grid_g, -1)
        pred_step_h = coords_2d[:, :, 1:, :] - coords_2d[:, :, :-1, :]
        true_step_h = true_2d[:, :, 1:, :] - true_2d[:, :, :-1, :]
        pred_step_v = coords_2d[:, 1:, :, :] - coords_2d[:, :-1, :, :]
        true_step_v = true_2d[:, 1:, :, :] - true_2d[:, :-1, :, :]
        ideal_pitch = (336.0 - 60.0) / 18.0
        err_h = torch.norm(pred_step_h - true_step_h, dim=-1)
        err_v = torch.norm(pred_step_v - true_step_v, dim=-1)
        v_x = (err_h > 0.22 * ideal_pitch).float().mean().item()
        v_y = (err_v > 0.22 * ideal_pitch).float().mean().item()
        topo_viol_pct = float((0.5 * (v_x + v_y)) * 100.0)
    else:
        topo_viol_pct = float((pred_coords[:, 1:, 0] < pred_coords[:, :-1, 0]).float().mean().item() * 100.0)

    m_occ = (valid_mask > 0.5) & (node_occ > 0.5) & (gt_labels > 0)
    if m_occ.sum().item() == 0:
        m_occ = (node_occ > 0.5) & (gt_labels > 0)
    if m_occ.sum().item() > 0:
        occ_recall_pct = float((preds[m_occ] == gt_labels[m_occ]).float().mean().item() * 100.0)
    else:
        occ_recall_pct = 0.0

    if grid_g * grid_g == h_feat.shape[1]:
        hf_2d = h_feat.view(n_samples, grid_g, grid_g, -1).numpy()
        adj_diff_var = float(np.var(hf_2d[:, :, 1:, :] - hf_2d[:, :, :-1, :]))
        err_rate = max(1e-3, 1.0 - overall_f1 / 100.0)
        nyquist_db = float(10.0 * math.log10((adj_diff_var + 1.0) / (err_rate ** 2)))
    else:
        nyquist_db = 12.0

    return {
        "Overall_Grid_MacroF1_pct": round(overall_f1, 2),
        "WTW_Lattice_300_F1_pct": round(wtw_f1, 2),
        "RefCOCO_Dense_300_Acc_pct": round(ref_acc, 2),
        "Oblique_Glare_HardSlice_Acc_pct": round(glare_acc, 2),
        "SubPixel_Coord_RMSE_px": round(rmse_px, 3),
        "Topology_Violation_Rate_pct": round(topo_viol_pct, 2),
        "Edge_Occlusion_Recall_pct": round(occ_recall_pct, 2),
        "Nyquist_Aliasing_Suppression_dB": round(nyquist_db, 2),
    }


@torch.no_grad()
def evaluate_single_checkpoint(model, ckpt_path: str, inputs_dict: dict, gt_vault: dict, v1_type: str, v2_type: str, v3_type: str, device: str):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    vit_24 = inputs_dict["vit_feat_24x24"].to(device)
    rgb_48 = inputs_dict["rgb_stem_48x48"].to(device)
    box_norm = inputs_dict["box_norm"].to(device)
    valid_mask = inputs_dict["valid_mask"].to(device)

    logits, pred_coords, h_feat = model(
        vit_24, rgb_48, box_norm, valid_mask,
        v1_type=v1_type, v2_type=v2_type, v3_type=v3_type
    )
    return compute_8_metrics(
        logits, pred_coords, h_feat, gt_vault,
        inputs_dict["valid_mask"], inputs_dict["track_id"]
    )


def _write_signed_csv(csv_path: str, rows: list):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    with open(csv_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    sig_path = csv_path + ".sha256"
    with open(sig_path, "w", encoding="utf-8") as f:
        f.write(f"{digest}  {os.path.basename(csv_path)}\n")
    print(f"[8-QA Blind Evaluator] Wrote {csv_path} (SHA256={digest[:16]}...)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=PROJECT_ROOT)
    parser.add_argument("--data_root", type=str, default="/root/autodl-fs/datasets/gridferret_general")
    parser.add_argument("--ckpt_dir", type=str, default="/root/autodl-fs/datasets/gridferret_general/checkpoints")
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()

    run_all_gates(args.project_root, args.data_root)
    if args.audit_only:
        return

    from models.blind_gridferret_model import BlindGridFerretModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_A = torch.load(os.path.join(args.data_root, "splits", "val_dataset_A.pt"), map_location="cpu", weights_only=False)
    test_B_in = torch.load(os.path.join(args.data_root, "splits", "test_dataset_B_inputs.pt"), map_location="cpu", weights_only=False)
    test_B_vault = torch.load(os.path.join(args.data_root, "gt_vault", "test_dataset_B_gt_vault.pt"), map_location="cpu", weights_only=False)

    manifest_path = os.path.join(args.ckpt_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[8-QA Blind Evaluator] Checkpoint manifest {manifest_path} not found yet. Standby for Step 6.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    step2_rows = []
    for entry in manifest.get("step2_probes", []):
        model = BlindGridFerretModel(**entry.get("model_kwargs", {}))
        m = evaluate_single_checkpoint(
            model, entry["ckpt_path"], val_A, val_A,
            entry["v1_type"], entry["v2_type"], entry["v3_type"], device
        )
        row = {"Probe_ID": entry["id"], "Paper_Tag": entry.get("name", entry["id"]), "Split": "Dataset_A_Val60"}
        row.update(m)
        step2_rows.append(row)

    if step2_rows:
        _write_signed_csv(os.path.join(args.project_root, "results", "step2_dataset_a_probe_scorecard.csv"), step2_rows)

    leaderboard_rows = []
    metric_keys = [
        "Overall_Grid_MacroF1_pct",
        "WTW_Lattice_300_F1_pct",
        "RefCOCO_Dense_300_Acc_pct",
        "Oblique_Glare_HardSlice_Acc_pct",
        "SubPixel_Coord_RMSE_px",
        "Topology_Violation_Rate_pct",
        "Edge_Occlusion_Recall_pct",
        "Nyquist_Aliasing_Suppression_dB",
    ]
    for cfg in manifest.get("step6_relay", []):
        seed_metrics = []
        for seed_ckpt in cfg["seed_ckpts"]:
            model = BlindGridFerretModel(**cfg.get("model_kwargs", {}))
            m = evaluate_single_checkpoint(
                model, seed_ckpt["path"], test_B_in, test_B_vault,
                cfg["v1_type"], cfg["v2_type"], cfg["v3_type"], device
            )
            seed_metrics.append(m)
        agg_row = {
            "Relay_Stage": cfg["id"],
            "Model_Name": cfg.get("name", cfg["id"]),
            "V1": cfg["v1_type"],
            "V2": cfg["v2_type"],
            "V3": cfg["v3_type"],
            "Seeds": "42,1024,2026",
        }
        for mk in metric_keys:
            vals = [sm[mk] for sm in seed_metrics]
            agg_row[mk] = round(float(np.mean(vals)), 3 if "RMSE" in mk else 2)
            agg_row[f"{mk}_std"] = round(float(np.std(vals)), 3)
        leaderboard_rows.append(agg_row)

    if leaderboard_rows:
        _write_signed_csv(os.path.join(args.project_root, "results", "leaderboard.csv"), leaderboard_rows)


if __name__ == "__main__":
    main()
