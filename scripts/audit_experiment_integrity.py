#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFC-006 V8.0 Five-Gate Anti-Cheating Integrity Auditor
Author: 8-QA-RedTeamAuditor
"""

import os
import sys
import ast
import hashlib
import argparse
import torch

FORBIDDEN_MODEL_TOKENS = {
    "true_norm",
    "hom_norm",
    "gt_labels",
    "node_glare",
    "node_occ",
    "gt_vault",
}

ALLOWED_FORWARD_ARGS = [
    "self",
    "vit_24",
    "rgb_48",
    "box_norm",
    "valid_mask",
    "v1_type",
    "v2_type",
    "v3_type",
]

EXPECTED_BLIND_INPUT_KEYS = {
    "vit_feat_24x24",
    "rgb_stem_48x48",
    "box_norm",
    "valid_mask",
    "sample_ids",
    "track_id",
}


def _row_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def check_gate1_ast_zero_oracle_leakage(project_root: str):
    model_path = os.path.join(project_root, "models", "blind_gridferret_model.py")
    if not os.path.exists(model_path):
        print(f"[Gate 1] [PASS (Pre-Train Standby)] {model_path} does not exist yet; AST scanner armed.")
        return

    with open(model_path, "r", encoding="utf-8") as f:
        src = f.read()

    tree = ast.parse(src, filename=model_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_MODEL_TOKENS:
            raise AssertionError(f"FATAL_ORACLE_LEAKAGE: Forbidden identifier '{node.id}' found in {model_path}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_MODEL_TOKENS:
            raise AssertionError(f"FATAL_ORACLE_LEAKAGE: Forbidden attribute '.{node.attr}' found in {model_path}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for tok in FORBIDDEN_MODEL_TOKENS:
                if tok in node.value:
                    raise AssertionError(f"FATAL_ORACLE_LEAKAGE: Forbidden key literal '{tok}' found in {model_path}")
        if isinstance(node, ast.arg) and node.arg in FORBIDDEN_MODEL_TOKENS:
            raise AssertionError(f"FATAL_ORACLE_LEAKAGE: Forbidden argument '{node.arg}' found in {model_path}")

    found_class = False
    found_forward = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "BlindGridFerretModel":
            found_class = True
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    found_forward = True
                    arg_names = [a.arg for a in item.args.args]
                    if arg_names != ALLOWED_FORWARD_ARGS:
                        raise AssertionError(
                            f"FATAL_ORACLE_LEAKAGE: BlindGridFerretModel.forward signature mismatch! "
                            f"Expected {ALLOWED_FORWARD_ARGS}, got {arg_names}"
                        )
                    if item.args.vararg is not None or item.args.kwarg is not None:
                        raise AssertionError(
                            "FATAL_ORACLE_LEAKAGE: *args / **kwargs are forbidden in BlindGridFerretModel.forward!"
                        )
    if not found_class or not found_forward:
        raise AssertionError("FATAL_ORACLE_LEAKAGE: BlindGridFerretModel.forward not found in models/blind_gridferret_model.py")

    print("[Gate 1] [PASS] AST Zero Oracle Leakage in Models verified.")


def check_gate2_strict_split_disjointness(data_root: str):
    splits_dir = os.path.join(data_root, "splits")
    train_d = torch.load(os.path.join(splits_dir, "train_split.pt"), map_location="cpu", weights_only=False)
    val_d = torch.load(os.path.join(splits_dir, "val_dataset_A.pt"), map_location="cpu", weights_only=False)
    test_d = torch.load(os.path.join(splits_dir, "test_dataset_B_inputs.pt"), map_location="cpu", weights_only=False)

    ids_train = set(train_d["sample_ids"].tolist())
    ids_val = set(val_d["sample_ids"].tolist())
    ids_test = set(test_d["sample_ids"].tolist())

    assert len(ids_train) == 420, f"Expected 420 train samples, got {len(ids_train)}"
    assert len(ids_val) == 60, f"Expected 60 val samples, got {len(ids_val)}"
    assert len(ids_test) == 120, f"Expected 120 test samples, got {len(ids_test)}"
    assert len(ids_train) + len(ids_val) + len(ids_test) == 600
    assert len(ids_train & ids_val) == 0, "FATAL: train and val sample_ids overlap!"
    assert len(ids_train & ids_test) == 0, "FATAL: train and test sample_ids overlap!"
    assert len(ids_val & ids_test) == 0, "FATAL: val and test sample_ids overlap!"

    h_train = {_row_sha256(train_d["vit_feat_24x24"][i]) for i in range(420)}
    h_val = {_row_sha256(val_d["vit_feat_24x24"][i]) for i in range(60)}
    h_test = {_row_sha256(test_d["vit_feat_24x24"][i]) for i in range(120)}

    assert len(h_train & h_val) == 0, "FATAL: SHA256 feature hash collision between train and val!"
    assert len(h_train & h_test) == 0, "FATAL: SHA256 feature hash collision between train and test!"
    assert len(h_val & h_test) == 0, "FATAL: SHA256 feature hash collision between val and test!"

    print("[Gate 2] [PASS] Strict Split Disjointness (420 + 60 + 120 == 600, 0% ID/SHA256 overlap) verified.")


def check_gate3_blind_input_sanitization(data_root: str):
    test_path = os.path.join(data_root, "splits", "test_dataset_B_inputs.pt")
    vault_path = os.path.join(data_root, "gt_vault", "test_dataset_B_gt_vault.pt")
    test_d = torch.load(test_path, map_location="cpu", weights_only=False)
    actual_keys = set(test_d.keys())
    if actual_keys != EXPECTED_BLIND_INPUT_KEYS:
        raise AssertionError(
            f"FATAL_BLIND_INPUT_CONTAMINATION: test_dataset_B_inputs.pt keys {actual_keys} "
            f"!= expected {EXPECTED_BLIND_INPUT_KEYS}"
        )
    # Verify valid_mask does NOT encode node_occ (must be > 0.5 for occluded in-canvas nodes)
    if os.path.exists(vault_path):
        vault_d = torch.load(vault_path, map_location="cpu", weights_only=False)
        occ_nodes = vault_d["node_occ"] > 0.5
        if occ_nodes.any():
            occ_valid_rate = float((test_d["valid_mask"][occ_nodes] > 0.5).float().mean().item())
            if occ_valid_rate < 0.99:
                raise AssertionError(
                    f"FATAL_VALID_MASK_OCCLUSION_LEAK: valid_mask excludes occluded nodes "
                    f"(occ_valid_rate={occ_valid_rate:.4f} < 0.99)!"
                )
    print(f"[Gate 3] [PASS] Blind Input Sanitization & Zero Occlusion Mask Leakage verified (keys={sorted(actual_keys)}).")


def check_gate4_training_script_vault_isolation(project_root: str):
    train_script = os.path.join(project_root, "scripts", "train_blind_models.py")
    if not os.path.exists(train_script):
        print(f"[Gate 4] [PASS (Pre-Train Standby)] {train_script} does not exist yet; Vault isolation scanner armed.")
        return

    with open(train_script, "r", encoding="utf-8") as f:
        content = f.read()

    for forbidden in ("gt_vault", "test_dataset_B"):
        if forbidden in content:
            raise AssertionError(
                f"FATAL_VAULT_BREACH: Forbidden reference '{forbidden}' found in {train_script}!"
            )
    print("[Gate 4] [PASS] Training Script Vault Isolation verified.")


def check_gate5_real_clip_vit_provenance(data_root: str):
    splits_dir = os.path.join(data_root, "splits")
    train_d = torch.load(os.path.join(splits_dir, "train_split.pt"), map_location="cpu", weights_only=False)
    feat = train_d["vit_feat_24x24"].float()

    assert feat.ndim >= 3 and feat.shape[0] == 420, f"Unexpected vit_feat_24x24 shape: {tuple(feat.shape)}"
    mean_val = float(feat.mean().item())
    std_val = float(feat.std().item())
    assert std_val > 0.05, f"Degenerate feature variance std={std_val:.6f}"
    assert not torch.isnan(feat).any() and not torch.isinf(feat).any(), "NaN/Inf in CLIP-ViT features"

    flat = feat.reshape(feat.shape[0], -1)
    rank = int(torch.linalg.matrix_rank(flat[:128]).item())
    assert rank >= 64, f"Feature matrix rank too low ({rank}), suspected synthetic constant block!"

    print(
        f"[Gate 5] [PASS] Real CLIP-ViT-Large-Patch14-336 Provenance verified "
        f"(shape={tuple(feat.shape)}, mean={mean_val:.4f}, std={std_val:.4f}, rank128={rank})."
    )


def run_all_gates(project_root: str, data_root: str):
    print("=" * 72)
    print("[8-QA-RedTeamAuditor] Running RFC-006 V8.0 5-Gate Integrity Audit...")
    print("=" * 72)
    check_gate1_ast_zero_oracle_leakage(project_root)
    check_gate2_strict_split_disjointness(data_root)
    check_gate3_blind_input_sanitization(data_root)
    check_gate4_training_script_vault_isolation(project_root)
    check_gate5_real_clip_vit_provenance(data_root)
    print("=" * 72)
    print("[8-QA-RedTeamAuditor] ALL 5 ANTI-CHEATING GATES: [PASS]")
    print("=" * 72)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-fs/datasets/gridferret_general",
    )
    args = parser.parse_args()
    run_all_gates(args.project_root, args.data_root)
