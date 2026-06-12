"""Stage 1 — globally-aligned per-frame camera poses.

Runs the NoDA-FT backbone over overlapping sliding windows. NoDA-FT is
depth-supervised, so each window's camera poses come out metric-scaled directly
from RGB — no calibration, scene scan, or URDF. The windows are then
stitched into one global frame with a rigid (6-DOF) Umeyama alignment on the camera
poses shared between consecutive windows.

Output: ``stage1_global_poses.npz`` with per-(camera, frame) world-to-camera
extrinsics in a single global, metric coordinate frame.

Usage:
    python stage1_pose.py \
        --frames-base /path/to/frames --cameras head,left_wrist,right_wrist \
        --frame-start 66 --frame-end 116 --finetuned-ckpt noda_ft.pt \
        --out /path/to/out/stage1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))            # depth_anything_3
sys.path.insert(0, str(Path(__file__).resolve().parent))  # utils

from utils.geometry import w2c_to_cam_center, apply_similarity_to_w2c, chain_align
from utils.io_utils import build_sliding_windows, window_image_list

DEFAULT_MODEL = "depth-anything/DA3-Giant"
DEFAULT_RES = 504


def load_model(model_id, finetuned_ckpt=None):
    from depth_anything_3.api import DepthAnything3
    print(f"  loading {model_id} ...")
    model = DepthAnything3.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.cuda()
        model.device = torch.device("cuda")
    if finetuned_ckpt:
        # NoDA-FT weights: LoRA adapters (rank 16) on the backbone attention Q/K/V
        # projections (A.1.3). Attach the adapters before loading; the fine-tuning
        # recipe is a one-time offline step out of scope for this inference release.
        ck = torch.load(finetuned_ckpt, map_location="cpu")
        saved = ck.get("trainable_state", ck)
        own = dict(model.model.named_parameters())
        n = 0
        for k, v in saved.items():
            if k in own and own[k].shape == v.shape:
                with torch.no_grad():
                    own[k].copy_(v.to(own[k].device, dtype=own[k].dtype))
                n += 1
        print(f"  loaded {n}/{len(saved)} NoDA-FT tensors from {finetuned_ckpt}")
        model.model.eval()
    print(f"  ready on {next(model.parameters()).device}")
    return model


def infer_window(model, img_paths, res):
    """One multi-view forward → per-view world-to-camera extrinsics (N,3,4)."""
    t0 = time.time()
    pred = model.inference(
        img_paths, process_res=res, ref_view_strategy="first",
        use_ray_pose=False, infer_gs=False,
    )
    if pred.extrinsics is None:
        raise RuntimeError("model returned no extrinsics")
    return np.asarray(pred.extrinsics), time.time() - t0


def run_stage1(frames_base, cameras, frame_start, frame_end, win_size, overlap,
               model_id, finetuned_ckpt, res, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cam = len(cameras)
    n_frames = frame_end - frame_start
    windows = build_sliding_windows(frame_start, frame_end, win_size, overlap)
    print(f"  cameras={cameras}  frames=[{frame_start},{frame_end})  windows={windows}")

    model = load_model(model_id, finetuned_ckpt)

    # --- per-window inference (metric poses straight from NoDA-FT) ---
    per_window_ext, per_window_centers = [], []
    for w_idx, (w_s, w_e) in enumerate(windows):
        n_f = w_e - w_s
        img_paths = window_image_list(frames_base, cameras, w_s, w_e)
        assert len(img_paths) == n_cam * n_f
        ext, dt = infer_window(model, img_paths, res)
        assert ext.shape[0] == n_cam * n_f, f"ext {ext.shape} != {n_cam * n_f}"
        per_window_ext.append(ext)
        per_window_centers.append(np.array([w2c_to_cam_center(e) for e in ext]))
        print(f"  W{w_idx:02d} [{w_s}-{w_e - 1}] {len(img_paths)} views  t={dt:.1f}s")

    # --- stitch windows into one global frame ---
    # Each window is already metric, so the inter-window transform is rigid 6-DOF,
    # solved in closed form (Umeyama) on the camera poses shared via overlap frames.
    _, drifts, transforms = chain_align(
        per_window_centers, overlap, n_cam=n_cam, with_scale=False)
    print(f"  chain drifts (%): {[f'{d:.2f}' for d in drifts]}")

    # --- assemble per-(camera, frame) global poses ---
    ext_g = {c: [None] * n_frames for c in cameras}
    cen_g = {c: [None] * n_frames for c in cameras}
    for w_idx, (w_s, w_e) in enumerate(windows):
        n_f = w_e - w_s
        ext = per_window_ext[w_idx]
        s_chain, R_chain, t_chain = transforms[w_idx]
        for c_i, c_name in enumerate(cameras):
            for fi in range(n_f):
                local = ext[c_i * n_f + fi]
                w2c_glob = apply_similarity_to_w2c(local, s_chain, R_chain, t_chain)
                gf = w_s - frame_start + fi  # latest window wins on overlap frames
                ext_g[c_name][gf] = w2c_glob
                cen_g[c_name][gf] = w2c_to_cam_center(w2c_glob)
    for c in cameras:
        assert all(v is not None for v in ext_g[c]), f"unfilled frames for {c}"

    ext_out = np.stack([np.stack(ext_g[c], 0) for c in cameras], 0)  # (n_cam, n_frame, 4, 4)
    cen_out = np.stack([np.stack(cen_g[c], 0) for c in cameras], 0)  # (n_cam, n_frame, 3)

    out_npz = out_dir / "stage1_global_poses.npz"
    np.savez(out_npz,
             cameras=np.array(cameras, dtype=object),
             frame_indices=np.arange(frame_start, frame_end),
             extrinsics_global=ext_out,
             cam_centers_global=cen_out)
    summary = {
        "cameras": list(cameras), "frame_range": [frame_start, frame_end],
        "n_windows": len(windows),
        "chain_drift_max_pct": float(max(drifts)) if drifts else 0.0,
    }
    (out_dir / "stage1_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  saved {out_npz}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: per-frame global poses")
    p.add_argument("--frames-base", required=True, help="{frames_base}/{cam}/{frame:06d}.png")
    p.add_argument("--cameras", default="head,left_wrist,right_wrist")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, required=True, help="exclusive")
    p.add_argument("--win", type=int, default=10)
    p.add_argument("--overlap", type=int, default=5)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--finetuned-ckpt", default=None,
                   help="NoDA-FT weights (metric poses depend on the depth-supervised fine-tune)")
    p.add_argument("--res", type=int, default=DEFAULT_RES)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    cams = [c.strip() for c in a.cameras.split(",") if c.strip()]
    run_stage1(a.frames_base, cams, a.frame_start, a.frame_end, a.win, a.overlap,
               a.model, a.finetuned_ckpt, a.res, a.out)


if __name__ == "__main__":
    main()
