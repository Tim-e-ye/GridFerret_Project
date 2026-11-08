#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1.5 Data Proctor: Physical Blind Split & GT Vault Isolation
Author: 8-QA-RedTeamAuditor (RFC-006 V8.0)
"""

import os
import sys
import hashlib
import argparse
import numpy as np
import torch


OBSERVATION_KEYS = {
    "vit_feat_24x24",
    "rgb_stem_48x48",
    "box_norm",
    "valid_mask",
    "sample_ids",
    "track_id",
}

GT_SUPERVISION_KEYS = {
    "gt_labels",
    "true_norm",
    "hom_norm",
    "node_glare",
    "node_occ",
}

VAULT_KEYS = {"sample_ids", "track_id"} | GT_SUPERVISION_KEYS


def _resolve_raw_feature_path(cli_path: str = None) -> str:
    candidates = []
    if cli_path:
        candidates.append(cli_path)
    candidates.extend([
        "/dev/shm/gridferret_cache/gridferret_clip_vit_l14_features_v2.pt",
        "/root/autodl-fs/datasets/gridferret_general/gridferret_clip_vit_l14_features_v2.pt",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "gridferret_clip_vit_l14_features_v2.pt"),
    ])
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot find gridferret_clip_vit_l14_features_v2.pt in candidates: {candidates}")


def _normalize_raw_dict(raw: dict) -> dict:
    """Normalize potential key aliases in raw feature pack while preserving tensors on CPU."""
    norm = {}
    alias_map = {
        "vit_feat_24x24": ["vit_feat_24x24", "vit_24", "vit_feat", "clip_vit_feat"],
        "rgb_stem_48x48": ["rgb_stem_48x48", "rgb_48", "rgb_stem", "rgb_patches"],
        "box_norm": ["box_norm", "obs_norm", "noisy_norm", "bbox_norm"],
        "valid_mask": ["valid_mask", "mask", "node_mask"],
        "gt_labels": ["gt_labels", "labels", "target_labels", "cell_labels"],
        "true_norm": ["true_norm", "gt_norm", "clean_norm", "target_coords"],
        "hom_norm": ["hom_norm", "rect_norm", "homography_norm"],
        "node_glare": ["node_glare", "glare_mask", "is_glare"],
        "node_occ": ["node_occ", "occ_mask", "is_occ"],
    }
    for target_key, aliases in alias_map.items():
        found = None
        for k in aliases:
            if k in raw and torch.is_tensor(raw[k]):
                found = raw[k].detach().cpu()
                break
        if found is None and target_key in raw:
            found = torch.as_tensor(raw[target_key]).cpu()
        if found is None:
            raise KeyError(f"Missing required tensor for '{target_key}' in raw feature file (available keys: {list(raw.keys())})")
        norm[target_key] = found

    n_samples = norm["vit_feat_24x24"].shape[0]
    assert n_samples == 600, f"Expected 600 samples (0..299 WTW, 300..599 RefCOCO), got {n_samples}"

    if "sample_ids" in raw:
        norm["sample_ids"] = torch.as_tensor(raw["sample_ids"], dtype=torch.long).cpu()
    else:
        norm["sample_ids"] = torch.arange(n_samples, dtype=torch.long)

    if "track_id" in raw:
        norm["track_id"] = torch.as_tensor(raw["track_id"], dtype=torch.long).cpu()
    else:
        track_id = torch.zeros(n_samples, dtype=torch.long)
        track_id[300:600] = 1
        norm["track_id"] = track_id

    return norm


def _slice_dict(data_dict: dict, indices: torch.Tensor, allowed_keys: set) -> dict:
    out = {}
    for k in sorted(allowed_keys):
        out[k] = data_dict[k][indices].clone()
    return out


def build_blind_splits(raw_path: str, out_root: str, seed: int = 2026):
    print(f"[8-QA Data Proctor] Loading raw CLIP-ViT-L/14@336 pack from: {raw_path}")
    raw_data = torch.load(raw_path, map_location="cpu", weights_only=False)
    data = _normalize_raw_dict(raw_data)

    rng = np.random.RandomState(seed)

    # Track 0: 0..299 (WTW-Hard-Lattice)
    perm_t0 = rng.permutation(np.arange(0, 300))
    t0_train, t0_val, t0_test = perm_t0[:210], perm_t0[210:240], perm_t0[240:300]

    # Track 1: 300..599 (RefCOCO-Dense-Grid)
    perm_t1 = rng.permutation(np.arange(300, 600))
    t1_train, t1_val, t1_test = perm_t1[:210], perm_t1[210:240], perm_t1[240:300]

    idx_train = torch.from_numpy(np.sort(np.concatenate([t0_train, t1_train]))).long()
    idx_val = torch.from_numpy(np.sort(np.concatenate([t0_val, t1_val]))).long()
    idx_test = torch.from_numpy(np.sort(np.concatenate([t0_test, t1_test]))).long()

    assert len(idx_train) == 420 and len(idx_val) == 60 and len(idx_test) == 120
    assert len(set(idx_train.tolist()) & set(idx_val.tolist())) == 0
    assert len(set(idx_train.tolist()) & set(idx_test.tolist())) == 0
    assert len(set(idx_val.tolist()) & set(idx_test.tolist())) == 0

    all_keys = OBSERVATION_KEYS | GT_SUPERVISION_KEYS
    train_split = _slice_dict(data, idx_train, all_keys)
    val_dataset_A = _slice_dict(data, idx_val, all_keys)
    test_dataset_B_inputs = _slice_dict(data, idx_test, OBSERVATION_KEYS)
    test_dataset_B_gt_vault = _slice_dict(data, idx_test, VAULT_KEYS)

    for forbidden in GT_SUPERVISION_KEYS:
        assert forbidden not in test_dataset_B_inputs, f"FATAL: {forbidden} leaked into test_dataset_B_inputs!"
    assert set(test_dataset_B_inputs.keys()) == OBSERVATION_KEYS

    splits_dir = os.path.join(out_root, "splits")
    vault_dir = os.path.join(out_root, "gt_vault")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(vault_dir, exist_ok=True)

    p_train = os.path.join(splits_dir, "train_split.pt")
    p_val = os.path.join(splits_dir, "val_dataset_A.pt")
    p_test_in = os.path.join(splits_dir, "test_dataset_B_inputs.pt")
    p_vault = os.path.join(vault_dir, "test_dataset_B_gt_vault.pt")

    torch.save(train_split, p_train)
    torch.save(val_dataset_A, p_val)
    torch.save(test_dataset_B_inputs, p_test_in)
    torch.save(test_dataset_B_gt_vault, p_vault)

    print(f"[8-QA Data Proctor] Saved train_split.pt (N={len(idx_train)}) -> {p_train}")
    print(f"[8-QA Data Proctor] Saved val_dataset_A.pt (N={len(idx_val)}) -> {p_val}")
    print(f"[8-QA Data Proctor] Saved test_dataset_B_inputs.pt (N={len(idx_test)}, keys={sorted(test_dataset_B_inputs.keys())}) -> {p_test_in}")
    print(f"[8-QA Data Proctor] Saved test_dataset_B_gt_vault.pt (N={len(idx_test)}, keys={sorted(test_dataset_B_gt_vault.keys())}) -> {p_vault}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, default=None)
    parser.add_argument("--out_root", type=str, default="/root/autodl-fs/datasets/gridferret_general")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    raw_file = _resolve_raw_feature_path(args.raw_path)
    build_blind_splits(raw_file, args.out_root, seed=args.seed)
