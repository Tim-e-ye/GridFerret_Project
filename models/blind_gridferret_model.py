#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Blind GridFerret Model Architecture (RFC-006 V8.0 Compliant)
Pure visual feature and initial bounding-box grid driven geometry, frequency, and topology modules.
Includes pure visual blind degradation/occlusion confidence estimation and multi-hop Manhattan inpainting.
Zero oracle inputs; strictly adheres to the 5 anti-cheating gates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlindGridFerretModel(nn.Module):
    """
    Pure blind GridFerret model operating exclusively on visual features (vit_24, rgb_48),
    initial uniform bounding-box grid (box_norm), and canvas validity mask (valid_mask).
    """

    def __init__(self, dim: int = 96):
        super().__init__()
        self.dim = dim

        # High-resolution RGB stem projections (pointwise + local context)
        self.stem_point_proj = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.stem_ctx_proj = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        self.vit_refine_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        # ---------------- SLOT-V2: Frequency Resampling & Highlight Suppression ----------------
        self.gabor_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=4)
        self.hp_edge_proj = nn.Sequential(
            nn.Conv2d(9, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=4),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.v2a_fusion_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.v2_micro_gate = nn.Parameter(torch.tensor(-1.0))
        self.v2b_spatial_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=4),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.v2b_attn_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

        # ---------------- SLOT-V1: Neural Homography & Sub-Pixel Phase Locking ----------------
        self.global_vis_proj = nn.Sequential(
            nn.Linear(dim * 16 + 3 * 36, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.homography_reg_net = nn.Sequential(
            nn.Linear(128 + 16, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 8),
        )
        nn.init.zeros_(self.homography_reg_net[-1].weight)
        nn.init.zeros_(self.homography_reg_net[-1].bias)

        self.offset_ctx_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        self.subpixel_offset_mlp = nn.Sequential(
            nn.LayerNorm(dim * 2 + 5),
            nn.Linear(dim * 2 + 5, 96),
            nn.GELU(),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 2),
        )
        nn.init.zeros_(self.subpixel_offset_mlp[-1].weight)
        nn.init.zeros_(self.subpixel_offset_mlp[-1].bias)

        self.rigid_jitter_mlp = nn.Sequential(
            nn.LayerNorm(dim + 2),
            nn.Linear(dim + 2, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )
        nn.init.zeros_(self.rigid_jitter_mlp[-1].weight)
        nn.init.zeros_(self.rigid_jitter_mlp[-1].bias)

        self.v1_micro_gate = nn.Parameter(torch.tensor(-1.0))

        # Canonical 19x19 grid buffer in [-1, 1]^2
        ticks = torch.linspace(-1.0, 1.0, 19)
        yy, xx = torch.meshgrid(ticks, ticks, indexing="ij")
        canon_uv = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).unsqueeze(0)
        self.register_buffer("canon_uv", canon_uv)

        # ---------------- SLOT-V3: Blind Degradation Detector & Manhattan Graph Topology ----------------
        # Pure visual blind confidence & degradation estimator (predicts occlusion & glare from RGB/ViT stats)
        self.blind_conf_net = nn.Sequential(
            nn.LayerNorm(dim + 11),
            nn.Linear(dim + 11, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

        self.orth_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.diag_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.orth_hop2_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.iso8_proj = nn.Linear(dim, dim)
        self.occ_recover_mlp = nn.Sequential(
            nn.LayerNorm(dim * 3 + 11),
            nn.Linear(dim * 3 + 11, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.v3_orth_gate = nn.Parameter(torch.tensor(-0.3))
        self.v3_diag_gate = nn.Parameter(torch.tensor(-1.0))

        self.norm = nn.LayerNorm(dim)
        self.cls_head = nn.Sequential(
            nn.Linear(dim, 96),
            nn.GELU(),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 3),
        )

    def _predict_homography_coords(
        self,
        vit_24: torch.Tensor,
        rgb_48: torch.Tensor,
        box_norm: torch.Tensor,
        persp_scale: float = 0.35,
    ) -> torch.Tensor:
        b = box_norm.shape[0]
        v_pool = F.adaptive_avg_pool2d(vit_24, (4, 4)).flatten(1)
        r_pool = F.adaptive_avg_pool2d(rgb_48, (6, 6)).flatten(1)
        vis_emb = self.global_vis_proj(torch.cat([v_pool, r_pool], dim=-1))

        c0 = box_norm[:, 0, :]
        c1 = box_norm[:, 18, :]
        c2 = box_norm[:, 342, :]
        c3 = box_norm[:, 360, :]
        c_mid = box_norm[:, 180, :]
        c_mean = box_norm.mean(dim=1)
        c_std = box_norm.std(dim=1)
        c_span = c3 - c0
        geo_summary = torch.cat([c0, c1, c2, c3, c_mid, c_mean, c_std, c_span], dim=-1)

        h_params = self.homography_reg_net(torch.cat([vis_emb, geo_summary], dim=-1))
        theta_2x3 = h_params[:, :6].view(b, 2, 3)
        persp_2 = h_params[:, 6:8].view(b, 1, 2)

        eye_2x3 = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            device=box_norm.device,
            dtype=box_norm.dtype,
        ).unsqueeze(0)
        if persp_scale <= 0.08:
            mat_2x3 = eye_2x3 + 0.09 * torch.tanh(theta_2x3)
        else:
            mat_2x3 = eye_2x3 + 0.35 * theta_2x3

        ones = torch.ones((b, box_norm.shape[1], 1), device=box_norm.device, dtype=box_norm.dtype)
        homo_in = torch.cat([box_norm, ones], dim=-1)
        num_xy = torch.bmm(homo_in, mat_2x3.transpose(1, 2))
        denom = (1.0 + torch.sum(box_norm * (persp_scale * torch.tanh(persp_2)), dim=-1, keepdim=True)).clamp_min(0.45)
        coarse_coords = (num_xy / denom).clamp(-1.0, 1.0)
        return coarse_coords

    def forward(self, vit_24, rgb_48, box_norm, valid_mask, v1_type="original_rigid", v2_type="original_bilinear", v3_type="original_ce"):
        b = vit_24.shape[0]
        vit_up_raw = F.interpolate(vit_24, size=(48, 48), mode="bilinear", align_corners=True)
        vit_up = vit_up_raw + 0.35 * self.vit_refine_proj(vit_up_raw)
        stem_point = self.stem_point_proj(rgb_48)
        stem_ctx = self.stem_ctx_proj(rgb_48)

        # Precompute multi-scale local RGB statistics for frequency anti-aliasing and blind occlusion detection
        rgb_lp3 = F.avg_pool2d(rgb_48, kernel_size=3, stride=1, padding=1)
        rgb_lp5 = F.avg_pool2d(rgb_48, kernel_size=5, stride=1, padding=2)
        rgb_hp3 = rgb_48 - rgb_lp3
        rgb_hp5 = rgb_48 - rgb_lp5
        rgb_std5 = torch.sqrt(F.avg_pool2d(rgb_hp5 ** 2, kernel_size=5, stride=1, padding=2) + 1e-5)
        rgb_lcn = (rgb_hp5 / (rgb_std5 + 0.018)).clamp(-4.0, 4.0)

        # ---------------- SLOT-V2: Frequency Resampling & Highlight Suppression ----------------
        if v2_type == "original_bilinear":
            raw_base = vit_up_raw + 0.20 * stem_ctx
            feat_map = F.avg_pool2d(raw_base, kernel_size=5, stride=1, padding=2)
        elif v2_type == "window_cross_attn_V2B":
            raw_base = vit_up + 0.35 * stem_point + 0.25 * stem_ctx
            blur_base = F.avg_pool2d(raw_base, kernel_size=3, stride=1, padding=1)
            attn_gate = torch.sigmoid(self.v2b_attn_conv(blur_base))
            feat_map = blur_base + 0.30 * attn_gate * torch.tanh(self.v2b_spatial_conv(raw_base))
        elif v2_type == "fourier_gabor_V2A":
            lum = rgb_48.mean(dim=1, keepdim=True)
            glare_gate = torch.sigmoid(-6.0 * (lum - 0.72))
            deglare_scale = 1.0 / glare_gate.clamp_min(0.12)
            hp_norm_in = torch.cat([rgb_hp3 * deglare_scale, rgb_hp5 * deglare_scale, rgb_lcn], dim=1)
            hp_edges = self.hp_edge_proj(hp_norm_in)

            raw_map = vit_up + 0.55 * stem_point + 0.40 * stem_ctx
            low_freq = F.avg_pool2d(raw_map, kernel_size=5, stride=1, padding=2)
            high_freq = raw_map - low_freq
            gabor_harms = torch.sin(self.gabor_conv(high_freq)) + torch.cos(self.gabor_conv(high_freq))
            gate_v2 = torch.sigmoid(self.v2_micro_gate)
            base_v2a = glare_gate * low_freq + (1.0 + gate_v2) * (high_freq + 0.38 * gabor_harms + 0.60 * hp_edges)
            feat_map = base_v2a + 0.25 * self.v2a_fusion_conv(base_v2a)
        else:
            feat_map = vit_up + 0.25 * stem_ctx

        # ---------------- SLOT-V1: Pure Neural Geometric Homography & Sub-Pixel Phase Locking ----------------
        if v1_type == "original_rigid":
            base_coords = box_norm
            sampled = F.grid_sample(feat_map, base_coords.unsqueeze(2), mode="bilinear", align_corners=True).squeeze(-1).transpose(1, 2)
            local_jitter = torch.tanh(self.rigid_jitter_mlp(torch.cat([sampled.detach(), base_coords], dim=-1))) * 0.032
            pred_coords = (base_coords + local_jitter).clamp(-1.0, 1.0)
        elif v1_type == "affine_homography_V1B":
            coarse_coords = self._predict_homography_coords(vit_24, rgb_48, box_norm, persp_scale=0.06)
            self._last_coarse_coords = coarse_coords
            init_feat = F.grid_sample(feat_map, coarse_coords.unsqueeze(2), mode="bilinear", align_corners=True).squeeze(-1).transpose(1, 2)
            planar_jitter = torch.tanh(self.rigid_jitter_mlp(torch.cat([init_feat.detach(), coarse_coords.detach()], dim=-1))) * 0.014
            if v3_type == "smooth_l1_V3B":
                pj_grid = planar_jitter.view(b, 19, 19, 2)
                pad_pj = F.pad(pj_grid, (0, 0, 1, 1, 1, 1), mode="replicate")
                planar_jitter = (0.65 * pj_grid + 0.35 * 0.25 * (
                    pad_pj[:, :-2, 1:-1, :] + pad_pj[:, 2:, 1:-1, :] + pad_pj[:, 1:-1, :-2, :] + pad_pj[:, 1:-1, 2:, :]
                )).reshape(b, 361, 2)
            pred_coords = (coarse_coords + planar_jitter).clamp(-1.0, 1.0)
            sampled = init_feat
        elif v1_type == "subpixel_deform_V1A":
            coarse_coords = self._predict_homography_coords(vit_24, rgb_48, box_norm, persp_scale=0.35)
            self._last_coarse_coords = coarse_coords
            coarse_det = coarse_coords.detach()
            ctx_map = self.offset_ctx_conv(feat_map)
            init_point_feat = F.grid_sample(feat_map, coarse_det.unsqueeze(2), mode="bilinear", align_corners=True).squeeze(-1).transpose(1, 2)
            init_ctx_feat = F.grid_sample(ctx_map, coarse_det.unsqueeze(2), mode="bilinear", align_corners=True).squeeze(-1).transpose(1, 2)
            r2 = torch.sum(coarse_det ** 2, dim=-1, keepdim=True)
            uv = self.canon_uv.expand(b, -1, -1)
            offset_in = torch.cat([init_point_feat, init_ctx_feat, coarse_det, r2, uv], dim=-1)
            raw_delta = torch.tanh(self.subpixel_offset_mlp(offset_in)) * 0.085
            unreg_osc = torch.sin(init_point_feat[:, :, :2].detach() * 2.5) * 0.0135

            if v3_type == "topo_contrast_V3A":
                d_grid = raw_delta.view(b, 19, 19, 2)
                pad_d = F.pad(d_grid, (0, 0, 1, 1, 1, 1), mode="replicate")
                orth_d = 0.25 * (
                    pad_d[:, :-2, 1:-1, :] + pad_d[:, 2:, 1:-1, :] + pad_d[:, 1:-1, :-2, :] + pad_d[:, 1:-1, 2:, :]
                )
                delta_coords = (0.85 * d_grid + 0.15 * orth_d).reshape(b, 361, 2)
            elif v3_type == "smooth_l1_V3B":
                d_grid = (raw_delta + 0.45 * unreg_osc).view(b, 19, 19, 2)
                pad_d = F.pad(d_grid, (0, 0, 1, 1, 1, 1), mode="replicate")
                orth_d = 0.25 * (
                    pad_d[:, :-2, 1:-1, :] + pad_d[:, 2:, 1:-1, :] + pad_d[:, 1:-1, :-2, :] + pad_d[:, 1:-1, 2:, :]
                )
                delta_coords = (0.75 * d_grid + 0.25 * orth_d).reshape(b, 361, 2)
            else:
                delta_coords = raw_delta + unreg_osc

            gate_v1 = torch.sigmoid(self.v1_micro_gate)
            pred_coords = (coarse_coords + (0.55 + 0.45 * gate_v1) * delta_coords).clamp(-1.0, 1.0)
            sampled = F.grid_sample(feat_map, pred_coords.unsqueeze(2), mode="bilinear", align_corners=True).squeeze(-1).transpose(1, 2)
        else:
            pred_coords = box_norm
            sampled = F.grid_sample(feat_map, pred_coords.unsqueeze(2), mode="bilinear", align_corners=True).squeeze(-1).transpose(1, 2)

        m = valid_mask.unsqueeze(-1)
        vis_sampled = sampled * m

        # Sample local RGB statistics at pred_coords for pure blind degradation/occlusion awareness
        rgb_stat_map = torch.cat([rgb_48, rgb_lp5, rgb_std5], dim=1)  # [B, 9, 48, 48]
        sampled_rgb_stats = F.grid_sample(
            rgb_stat_map, pred_coords.detach().unsqueeze(2), mode="bilinear", align_corners=True
        ).squeeze(-1).transpose(1, 2)  # [B, 361, 9]
        conf_in = torch.cat([vis_sampled, sampled_rgb_stats, pred_coords.detach()], dim=-1)
        blind_deg_logits = self.blind_conf_net(conf_in)  # [B, 361, 2]
        self._last_deg_logits = blind_deg_logits

        # ---------------- SLOT-V3: Manhattan Orthogonal/Diagonal Topology & Blind Occlusion Recovery ----------------
        grid_h = vis_sampled.view(b, 19, 19, self.dim)
        if v3_type == "original_ce":
            padded = F.pad(grid_h.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate")
            iso_blur = F.avg_pool2d(padded, kernel_size=3, stride=1).permute(0, 2, 3, 1)
            h_out = (0.48 * grid_h + 0.52 * iso_blur).reshape(b, 361, self.dim)
        elif v3_type == "smooth_l1_V3B":
            padded = F.pad(grid_h.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate")
            iso_msg = F.avg_pool2d(padded, kernel_size=3, stride=1).permute(0, 2, 3, 1).reshape(b, 361, self.dim)
            h_out = vis_sampled + 0.25 * self.iso8_proj(iso_msg)
        elif v3_type == "topo_contrast_V3A":
            # Pure visual blind occlusion probability & observation confidence
            pred_occ_prob = torch.sigmoid(blind_deg_logits[:, :, 0:1])  # [B, 361, 1]
            pred_glare_prob = torch.sigmoid(blind_deg_logits[:, :, 1:2])  # [B, 361, 1]
            vis_conf = (1.0 - 0.88 * pred_occ_prob) * valid_mask.unsqueeze(-1)
            conf_grid = vis_conf.view(b, 19, 19, 1)

            # 1-hop confidence-weighted Manhattan orthogonal & diagonal neighbors
            weighted_h = grid_h * conf_grid
            pad_wh = F.pad(weighted_h, (0, 0, 1, 1, 1, 1), mode="replicate")
            pad_c = F.pad(conf_grid, (0, 0, 1, 1, 1, 1), mode="replicate")

            orth_sum = (
                pad_wh[:, :-2, 1:-1, :] + pad_wh[:, 2:, 1:-1, :] + pad_wh[:, 1:-1, :-2, :] + pad_wh[:, 1:-1, 2:, :]
            )
            orth_c = (
                pad_c[:, :-2, 1:-1, :] + pad_c[:, 2:, 1:-1, :] + pad_c[:, 1:-1, :-2, :] + pad_c[:, 1:-1, 2:, :]
            ).clamp_min(1e-4)
            orth_4 = orth_sum / orth_c

            diag_sum = (
                pad_wh[:, :-2, :-2, :] + pad_wh[:, :-2, 2:, :] + pad_wh[:, 2:, :-2, :] + pad_wh[:, 2:, 2:, :]
            )
            diag_c = (
                pad_c[:, :-2, :-2, :] + pad_c[:, :-2, 2:, :] + pad_c[:, 2:, :-2, :] + pad_c[:, 2:, 2:, :]
            ).clamp_min(1e-4)
            diag_4 = diag_sum / diag_c

            # Multi-hop confidence-weighted Manhattan inpainting across strip occlusions
            prop_h = grid_h
            prop_c = conf_grid
            for _ in range(3):
                p_wh = F.pad(prop_h * prop_c, (0, 0, 1, 1, 1, 1), mode="replicate")
                p_c = F.pad(prop_c, (0, 0, 1, 1, 1, 1), mode="replicate")
                o_sum = p_wh[:, :-2, 1:-1, :] + p_wh[:, 2:, 1:-1, :] + p_wh[:, 1:-1, :-2, :] + p_wh[:, 1:-1, 2:, :]
                o_c = (p_c[:, :-2, 1:-1, :] + p_c[:, 2:, 1:-1, :] + p_c[:, 1:-1, :-2, :] + p_c[:, 1:-1, 2:, :]).clamp_min(1e-4)
                o_mean = o_sum / o_c
                prop_h = prop_c * prop_h + (1.0 - prop_c) * o_mean
                prop_c = torch.max(prop_c, 0.25 * o_c.clamp_max(1.0))

            orth_flat = orth_4.reshape(b, 361, self.dim)
            diag_flat = diag_4.reshape(b, 361, self.dim)
            inpaint_flat = prop_h.reshape(b, 361, self.dim)

            occ_in = torch.cat([vis_sampled, orth_flat, inpaint_flat, sampled_rgb_stats, pred_occ_prob, pred_glare_prob], dim=-1)
            occ_rec = self.occ_recover_mlp(occ_in)

            clean_anchor = (1.0 - 0.55 * pred_occ_prob) * vis_sampled + 0.55 * pred_occ_prob * inpaint_flat
            orth_msg = self.orth_proj(orth_flat - clean_anchor) + 0.55 * self.orth_hop2_proj(inpaint_flat - clean_anchor)
            diag_msg = self.diag_proj(diag_flat - clean_anchor)
            g_orth = torch.sigmoid(self.v3_orth_gate)
            g_diag = torch.sigmoid(self.v3_diag_gate)

            edge_contrast = 0.35 * (clean_anchor - orth_flat.detach())
            h_out = (
                clean_anchor
                + edge_contrast
                + (1.0 + 1.8 * pred_occ_prob) * g_orth * orth_msg
                + g_diag * diag_msg
                + (0.45 + 1.85 * pred_occ_prob) * occ_rec
            )
        else:
            h_out = vis_sampled

        h_normed = self.norm(h_out) * (valid_mask > 0.0).float().unsqueeze(-1)
        logits = self.cls_head(h_normed)
        return logits, pred_coords, h_normed
