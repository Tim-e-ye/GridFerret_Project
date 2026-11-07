#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render representative 'Oblique + Strong Specular Glare' (H1+H2) and
'Finger / Sleeve / Bowl-Rim Occlusion + Motion Blur' (H3 & H1+H2+H3) test images
from `go_wild_hard_300.json` into `GridFerret_Project/results/sample_oblique_glare_images/`.
"""

import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FERRET_ROOT = r"D:\Code\VibeCode\Antigravity\Project\SymbioThesis\submodules\ferret"
OUT_DIR = r"D:\Code\VibeCode\Antigravity\Project\GridFerret_Project\results\sample_oblique_glare_images"
ART_DIR = r"C:\Users\Timeye\.gemini\antigravity\brain\9aa1def7-d512-4d28-b64c-1dab7f202bdc"


def apply_glare_pil(img_pil: Image.Image, glare_params: dict) -> Image.Image:
    arr = np.asarray(img_pil.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    ang = np.deg2rad(glare_params.get("shadow_angle_deg", 120.0))
    atten = glare_params.get("side_shadow_attenuation", 0.52)
    proj = (xx / w - 0.5) * np.cos(ang) + (yy / h - 0.5) * np.sin(ang) + 0.5
    shadow_factor = 1.0 - atten * np.clip(proj, 0.0, 1.0)
    arr = arr * shadow_factor[:, :, None]

    cx = glare_params["glare_center_norm"][0] * w
    cy = glare_params["glare_center_norm"][1] * h
    rx = glare_params["glare_ellipse_axes_ratio"][0] * w * 1.35
    ry = glare_params["glare_ellipse_axes_ratio"][1] * h * 1.35
    intensity = glare_params["glare_intensity"]
    dist2 = ((xx - cx) / max(1.0, rx)) ** 2 + ((yy - cy) / max(1.0, ry)) ** 2
    glare_mask = np.exp(-1.8 * dist2) * intensity

    cx2 = (1.0 - glare_params["glare_center_norm"][0]) * w
    cy2 = (0.35 + 0.3 * glare_params["glare_center_norm"][1]) * h
    dist2_b = ((xx - cx2) / max(1.0, rx * 0.7)) ** 2 + ((yy - cy2) / max(1.0, ry * 0.7)) ** 2
    glare_mask = np.clip(glare_mask + 0.85 * np.exp(-2.2 * dist2_b), 0.0, 0.96)

    arr = arr * (1.0 - glare_mask[:, :, None]) + 252.0 * glare_mask[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def apply_perspective_pil(img_pil: Image.Image, persp_params: dict) -> Image.Image:
    w, h = img_pil.size
    tilt = persp_params.get("perspective_tilt_deg", 28.0)
    shift_x = w * (0.0042 * tilt)
    shift_y = h * (0.0038 * tilt)
    dst = np.array(
        [
            [shift_x, shift_y],
            [w - shift_x, shift_y * 0.85],
            [w - shift_x * 0.15, h - shift_y * 0.25],
            [shift_x * 0.15, h - shift_y * 0.15],
        ],
        dtype=np.float64,
    )
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    a_mat, b_vec = [], []
    for (dx, dy), (sx, sy) in zip(dst, src):
        a_mat.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        a_mat.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        b_vec.append(sx)
        b_vec.append(sy)
    coeffs = np.linalg.solve(np.array(a_mat), np.array(b_vec))
    return img_pil.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(38, 34, 30))


def apply_oblique_and_glare_pil(img_pil: Image.Image, glare_params: dict, persp_params: dict):
    glare_img = apply_glare_pil(img_pil, glare_params)
    warped = apply_perspective_pil(glare_img, persp_params)
    return glare_img, warped


def apply_occlusion_pil(img_pil: Image.Image, occ_params: dict, mode_override: str = None) -> Image.Image:
    """Render realistic H3_Finger_Sleeve_Occlusion (finger_hand, sleeve_cuff, stone_bowl_rim) + motion blur."""
    base = img_pil.convert("RGBA")
    w, h = base.size
    occ_type = mode_override or occ_params.get("occluder_type", "finger_hand")
    ax, ay = occ_params.get("occlusion_anchor_norm", [0.62, 0.68])
    cx, cy = int(ax * w), int(ay * h)
    blur_k = int(occ_params.get("motion_blur_kernel", 9))

    shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    occ_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(shadow_layer)
    o_draw = ImageDraw.Draw(occ_layer)

    if occ_type == "finger_hand":
        # Realistic player's index & middle fingers + hand entering from bottom-right to place a stone
        ex, ey = int(w * 0.98), int(h * 0.96)
        # Soft cast shadow on the board under the hand and fingers
        s_draw.polygon(
            [(cx - 28, cy + 20), (cx + 45, cy - 15), (ex + 30, ey - 40), (ex - 60, ey + 30)],
            fill=(15, 12, 10, 145),
        )
        s_draw.ellipse([cx - 38, cy - 18, cx + 42, cy + 48], fill=(15, 12, 10, 155))

        # Index finger & middle finger holding a stone (warm skin tones + knuckle shading)
        o_draw.polygon(
            [(cx - 10, cy - 6), (cx + 22, cy - 28), (ex + 15, ey - 75), (ex - 35, ey - 10)],
            fill=(218, 172, 142, 248),
        )
        o_draw.polygon(
            [(cx - 26, cy + 12), (cx + 6, cy - 10), (ex - 25, ey - 20), (ex - 85, ey + 20)],
            fill=(204, 156, 126, 248),
        )
        # Palm / back of hand near the corner
        o_draw.ellipse([ex - 165, ey - 135, ex + 50, ey + 60], fill=(212, 165, 135, 252))
        # Fingertip & fingernail highlight near the intersection
        o_draw.ellipse([cx - 28, cy - 22, cx + 16, cy + 20], fill=(226, 182, 154, 250))
        o_draw.ellipse([cx - 20, cy - 15, cx - 2, cy + 2], fill=(242, 212, 195, 235))

    elif occ_type == "sleeve_cuff":
        # Dark jacket/shirt sleeve cuff sweeping across the lower-right/edge region during move execution
        sx0, sy0 = int(w * 0.52), int(h * 0.64)
        s_draw.polygon(
            [(sx0 - 35, sy0 + 25), (w, int(h * 0.44)), (w, h), ( int(w * 0.36), h)],
            fill=(10, 10, 12, 165),
        )
        # Main dark navy/charcoal fabric sleeve body
        o_draw.polygon(
            [(sx0, sy0), (w, int(h * 0.48)), (w, h), (int(w * 0.42), h)],
            fill=(36, 42, 54, 252),
        )
        # Inner white shirt cuff edge & wrist glimpse protruding from sleeve
        o_draw.polygon(
            [(sx0 - 18, sy0 - 12), (sx0 + 55, sy0 - 45), (sx0 + 72, sy0 - 20), (sx0 - 2, sy0 + 18)],
            fill=(228, 232, 238, 245),
        )
        o_draw.ellipse([sx0 - 48, sy0 - 36, sx0 + 25, sy0 + 28], fill=(215, 168, 138, 245))

    elif occ_type == "stone_bowl_rim":
        # Traditional chestnut-wood Go bowl (Goke) placed too close to the board edge, occluding 1st-3rd lines
        bx, by, br = int(w * 0.88), int(h * 0.22), int(w * 0.25)
        s_draw.ellipse([bx - br - 28, by - br + 18, bx + br - 10, by + br + 36], fill=(12, 10, 8, 165))
        # Outer wooden bowl body
        o_draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(112, 62, 34, 255))
        # Polished wood rim highlight ring
        o_draw.ellipse([bx - br + 14, by - br + 14, bx + br - 14, by + br - 14], fill=(148, 88, 48, 255))
        # Dark interior opening of the bowl with spare stones inside
        o_draw.ellipse([bx - br + 34, by - br + 34, bx + br - 34, by + br - 34], fill=(32, 22, 16, 255))
        for dx, dy, col in [(-25, -18, 240), (10, -28, 235), (-12, 15, 242), (22, 8, 230)]:
            o_draw.ellipse([bx + dx - 16, by + dy - 16, bx + dx + 16, by + dy + 16], fill=(col, col, col, 250))

    # Apply realistic Gaussian blur to cast shadow and directional motion blur to moving occluder
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=14))
    occ_layer = occ_layer.filter(ImageFilter.GaussianBlur(radius=max(2, blur_k // 3)))

    comp = Image.alpha_composite(base, shadow_layer)
    comp = Image.alpha_composite(comp, occ_layer)
    return comp.convert("RGB")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(FERRET_ROOT, "Datasets", "go_wild_hard_300.json"), "r", encoding="utf-8") as f:
        hard_data = json.load(f)

    # Part 1: 4 Oblique + Strong Specular Glare samples (H1 + H2)
    pick_indices = [10, 35, 55, 72]
    for i, idx in enumerate(pick_indices):
        h1_rec = hard_data[idx]
        h2_rec = hard_data[75 + idx]
        img_path = os.path.join(FERRET_ROOT, h1_rec["image"])
        orig = Image.open(img_path).convert("RGB")
        glare_only, oblique_glare = apply_oblique_and_glare_pil(orig, h1_rec["hard_params"], h2_rec["hard_params"])

        h1_id = h1_rec["hard_id"]
        h2_id = h2_rec["hard_id"]
        single_name = f"sample_{i+1:02d}_{h1_id}_{h2_id}_oblique_glare.png"
        oblique_glare.save(os.path.join(OUT_DIR, single_name))
        oblique_glare.save(os.path.join(ART_DIR, single_name))

        w, h = orig.size
        pw, ph = w // 2, h // 2
        comp = Image.new("RGB", (pw * 3 + 20, ph + 10), (25, 25, 25))
        comp.paste(orig.resize((pw, ph), Image.LANCZOS), (0, 5))
        comp.paste(glare_only.resize((pw, ph), Image.LANCZOS), (pw + 10, 5))
        comp.paste(oblique_glare.resize((pw, ph), Image.LANCZOS), (pw * 2 + 20, 5))
        comp_name = f"compare_{i+1:02d}_{h1_id}.png"
        comp.save(os.path.join(OUT_DIR, comp_name))
        comp.save(os.path.join(ART_DIR, comp_name))
        print(f"Saved H1+H2: {single_name} | {comp_name}")

    # Part 2: 4 H3_Finger_Sleeve_Occlusion samples (covering finger_hand, sleeve_cuff, stone_bowl_rim, and compound H1+H2+H3)
    occ_configs = [
        (155, "finger_hand", "finger_hand_occlusion"),
        (178, "sleeve_cuff", "sleeve_cuff_occlusion"),
        (202, "stone_bowl_rim", "bowl_rim_occlusion"),
        (218, "finger_hand", "compound_oblique_glare_finger_occlusion"),
    ]
    for i, (h3_idx, occ_mode, tag) in enumerate(occ_configs):
        h3_rec = hard_data[h3_idx]
        h1_rec = hard_data[h3_idx - 150]
        h2_rec = hard_data[h3_idx - 75]
        img_path = os.path.join(FERRET_ROOT, h3_rec["image"])
        orig = Image.open(img_path).convert("RGB")

        # Panel 2: Occlusion on top-down board
        occ_only = apply_occlusion_pil(orig, h3_rec["hard_params"], mode_override=occ_mode)
        # Panel 3: Compound Occlusion + Strong Specular Glare + Oblique Tilt (H1 + H2 + H3)
        glare_occ = apply_glare_pil(occ_only, h1_rec["hard_params"])
        oblique_glare_occ = apply_perspective_pil(glare_occ, h2_rec["hard_params"])

        h3_id = h3_rec["hard_id"]
        single_name = f"sample_occ_{i+1:02d}_{h3_id}_{tag}.png"
        oblique_glare_occ.save(os.path.join(OUT_DIR, single_name))
        oblique_glare_occ.save(os.path.join(ART_DIR, single_name))

        w, h = orig.size
        pw, ph = w // 2, h // 2
        comp = Image.new("RGB", (pw * 3 + 20, ph + 10), (25, 25, 25))
        comp.paste(orig.resize((pw, ph), Image.LANCZOS), (0, 5))
        comp.paste(occ_only.resize((pw, ph), Image.LANCZOS), (pw + 10, 5))
        comp.paste(oblique_glare_occ.resize((pw, ph), Image.LANCZOS), (pw * 2 + 20, 5))
        comp_name = f"compare_occ_{i+1:02d}_{h3_id}_{occ_mode}.png"
        comp.save(os.path.join(OUT_DIR, comp_name))
        comp.save(os.path.join(ART_DIR, comp_name))
        print(f"Saved H3: {single_name} | {comp_name}")


if __name__ == "__main__":
    main()
