"""Stage 1 — per-frame Gaussian Splat initialization (after pose estimation).

For each target frame we form a short sliding window (target + look-ahead) and run
one NoDA-FT forward over the 3xW views, reading the per-pixel Gaussian primitives
from the GS head. The window's primitives are lifted into Stage 1's global frame
with a closed-form Umeyama alignment on the shared camera poses, then combined with
the cross-frame opacity filter of Eq. 4 — target-frame primitives are kept
unconditionally; look-ahead primitives are kept only where their opacity clears
tau — into one Gaussian cloud per frame. NoDA-FT predicts low opacity on dynamic
pixels, so anything that moves within the window is dropped from the look-ahead
frames rather than smeared into the splat.

Output: ``per_frame_gaussians/frame_{:06d}.npz`` consumed by Stage 2.

Usage:
    python stage1_gaussians.py \
        --frames-base /path/to/frames --stage1 /path/to/out/stage1 \
        --finetuned-ckpt noda_ft.pt --frames 66-115 --out /path/to/out/gaussians
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.geometry import w2c_to_cam_center, umeyama
from utils.gaussians import apply_sim_to_gs
from utils.io_utils import build_per_frame_windows, frame_path, parse_frames

GIANT_MODEL = "depth-anything/DA3-Giant"
DEFAULT_RES = 504
EDGE_TRIM = 8 / 256  # fraction of H/W trimmed from each side


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
    return model


def forward_window(model, img_paths, res):
    """One NoDA-FT forward with GS. Returns ext, depth, and per-view Gaussians."""
    pred = model.inference(
        img_paths, process_res=res, ref_view_strategy="first",
        use_ray_pose=False, infer_gs=True,
    )
    depth = np.asarray(pred.depth)               # (V, H, W)
    V, H, W = depth.shape
    G = H * W
    gs = pred.gaussians
    out = dict(
        ext=np.asarray(pred.extrinsics),
        intr=np.asarray(pred.intrinsics),
        depth=depth,
        means=gs.means.view(V, G, 3).detach().cpu().numpy(),
        scales=gs.scales.view(V, G, 3).detach().cpu().numpy(),
        rotations=gs.rotations.view(V, G, 4).detach().cpu().numpy(),
        opacities=gs.opacities.view(V, G).detach().cpu().numpy(),
        harmonics_dc=gs.harmonics.view(V, G, 3, -1).detach().cpu().numpy()[..., 0],
    )
    return out, (V, H, W)


def run_stage1_gaussians(frames_base, stage1_dir, frames, context, model_id, res, out_dir,
                          ref_cam, prune_depth_pct, opacity_tau, finetuned_ckpt):
    stage1_npz = Path(stage1_dir) / "stage1_global_poses.npz"
    s1 = np.load(stage1_npz, allow_pickle=True)
    s1_cams = list(s1["cameras"])
    f_idx = [int(f) for f in s1["frame_indices"]]
    frame_start = f_idx[0]
    ext_global_all = s1["extrinsics_global"]   # (n_s1_cam, n_frame, 4, 4)
    cen_global_all = s1["cam_centers_global"]  # (n_s1_cam, n_frame, 3)

    if ref_cam:
        assert ref_cam in s1_cams, f"--ref-cam {ref_cam} not in stage1 {s1_cams}"
        cams, s1_idx = [ref_cam], [s1_cams.index(ref_cam)]
    else:
        cams, s1_idx = s1_cams[:], list(range(len(s1_cams)))
    n_cam = len(cams)
    ext_global = ext_global_all[s1_idx]
    cen_global = cen_global_all[s1_idx]

    out_dir = Path(out_dir)
    per_frame_dir = out_dir / "per_frame_gaussians"
    per_frame_dir.mkdir(parents=True, exist_ok=True)

    windows = build_per_frame_windows(frames[0], frames[-1] + 1, context)
    windows = [w for w in windows if w[0] in set(frames)]
    print(f"  cams={cams}  {len(windows)} per-frame windows  context={context}")

    model = load_model(model_id, finetuned_ckpt)

    summary, t_start = [], time.time()
    for i, (target_frame, frames_in_win, target_idx) in enumerate(windows):
        n_f = len(frames_in_win)
        img_paths = [frame_path(frames_base, c, f) for c in cams for f in frames_in_win]
        pred, (V, H, W) = forward_window(model, img_paths, res)
        ext, depth = pred["ext"], pred["depth"]

        # Dump per-cam intrinsics once (Stage 2 needs them).
        if i == 0:
            for c_i, c in enumerate(cams):
                np.save(out_dir / f"intrinsics_{c}.npy", pred["intr"][c_i * n_f + target_idx])

        # Lift this window into Stage 1's global frame: Umeyama on the camera poses
        # shared between the window and the Stage 1 trajectory.
        src, dst = [], []
        for c_i in range(n_cam):
            src.append(np.array([w2c_to_cam_center(ext[c_i * n_f + fi]) for fi in range(n_f)]))
            dst.append(cen_global[c_i, np.array(frames_in_win) - frame_start])
        s, R, t, res_um = umeyama(np.vstack(src), np.vstack(dst))

        # Assemble the per-frame splat with the Eq. 4 cross-frame filter:
        #   G_t = U_{t' in window} { p : t' = target  or  opacity(p) > tau }.
        # Target-frame primitives are kept unconditionally; primitives lifted from
        # the look-ahead frames are admitted only where opacity clears tau, which
        # drops the smeared geometry left behind by anything that moved.
        gh, gw = max(int(EDGE_TRIM * H), 1), max(int(EDGE_TRIM * W), 1)
        means, scales, rots, ops, harms, cam_ids = [], [], [], [], [], []
        for c_i in range(n_cam):
            for fi in range(n_f):
                v = c_i * n_f + fi
                mask = np.zeros((H, W), dtype=bool)
                mask[gh:H - gh, gw:W - gw] = True
                if prune_depth_pct < 1.0:  # drop far-field outliers
                    mask &= (depth[v] <= np.quantile(depth[v].flatten(), prune_depth_pct))
                idx = mask.flatten()

                m, sc, rq = apply_sim_to_gs(
                    pred["means"][v][idx], pred["scales"][v][idx], pred["rotations"][v][idx], s, R, t)
                op = pred["opacities"][v][idx]
                harm = pred["harmonics_dc"][v][idx]
                if fi != target_idx:  # look-ahead frames: keep only confident primitives
                    kp = op > opacity_tau
                    m, sc, rq, op, harm = m[kp], sc[kp], rq[kp], op[kp], harm[kp]
                means.append(m); scales.append(sc); rots.append(rq)
                ops.append(op); harms.append(harm)
                cam_ids.append(np.full(len(m), c_i, dtype=np.int32))

        fi = target_frame - frame_start
        n_gs = sum(len(m) for m in means)
        np.savez(
            per_frame_dir / f"frame_{target_frame:06d}.npz",
            frame_index=target_frame,
            cameras=np.array(cams, dtype=object),
            means=np.concatenate(means).astype(np.float32),
            scales=np.concatenate(scales).astype(np.float32),
            rotations=np.concatenate(rots).astype(np.float32),
            opacities=np.concatenate(ops).astype(np.float32),
            harmonics_dc=np.concatenate(harms).astype(np.float32),
            camera_idx=np.concatenate(cam_ids),
            extrinsics_global=ext_global[:, fi].astype(np.float32),
            cam_centers_global=cen_global[:, fi].astype(np.float32),
            umeyama_scale=float(s), umeyama_residual_max_mm=float(res_um.max() * 1000),
        )
        summary.append({"frame": target_frame, "n_gaussians": int(n_gs),
                        "umeyama_res_max_mm": float(res_um.max() * 1000)})
        if i % 5 == 0 or i == len(windows) - 1:
            print(f"  frame {target_frame:4d} [{i+1}/{len(windows)}] "
                  f"{n_gs:,} gs  umeyama_max={res_um.max()*1000:.1f}mm")

    (out_dir / "stage1_gaussians_summary.json").write_text(json.dumps(
        {"cameras": cams, "context": context, "n_frames": len(summary),
         "total_time_s": time.time() - t_start, "per_frame": summary}, indent=2))
    print(f"  saved {out_dir / 'stage1_gaussians_summary.json'}")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: per-frame Gaussian init")
    p.add_argument("--frames-base", required=True)
    p.add_argument("--stage1", required=True, help="Stage 1 output dir")
    p.add_argument("--frames", default=None, help="frame spec; default = all stage1 frames")
    p.add_argument("--context", type=int, default=4, help="look-ahead frames (window = 1+context)")
    p.add_argument("--ref-cam", default=None, help="single-camera ref pass (default: all stage1 cams)")
    p.add_argument("--model", default=GIANT_MODEL)
    p.add_argument("--res", type=int, default=DEFAULT_RES)
    p.add_argument("--prune-depth-pct", type=float, default=0.9)
    p.add_argument("--opacity-tau", type=float, default=0.01,
                   help="cross-frame primitive filter threshold tau (Eq. 4)")
    p.add_argument("--finetuned-ckpt", default=None, help="NoDA-FT weights")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    s1 = np.load(Path(a.stage1) / "stage1_global_poses.npz", allow_pickle=True)
    fi = [int(f) for f in s1["frame_indices"]]
    frames = parse_frames(a.frames) if a.frames else fi
    run_stage1_gaussians(a.frames_base, a.stage1, frames, a.context, a.model, a.res, a.out,
                          a.ref_cam, a.prune_depth_pct, a.opacity_tau, a.finetuned_ckpt)


if __name__ == "__main__":
    main()
