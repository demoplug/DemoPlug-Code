"""Stage 2 — per-frame photometric refinement.

Each frame's Gaussians (from the Stage 1 splat-init step) are refined by minimizing a
multi-camera L1 photometric loss against the ground-truth RGB. View matrices are taken
directly from Stage 1's global extrinsics (consistent across cameras), which avoids the
inter-camera misalignment that causes frame-edge break-up artifacts.

Output: ``per_frame_gaussians/frame_{:06d}.npz`` with refined Gaussians.

Usage:
    python stage2_refine.py \
        --stage1 /path/to/out/stage1 --init-from /path/to/out/gaussians \
        --gt-base /path/to/frames --frames 66-115 \
        --out /path/to/out/refine
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.gaussians import GSParams
from utils.io_utils import parse_frames

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Per-parameter learning rates (means need larger steps to migrate in space).
LR = {"means": 2e-4, "log_scales": 5e-3, "quats": 1e-3, "logit_op": 5e-2, "harm_dc": 2.5e-3}


def load_intrinsics(d, cameras):
    return {c: np.load(Path(d) / f"intrinsics_{c}.npy") for c in cameras}


def build_views(frame, s1_cams, s1_ext, frame_to_s1, cam_K, gt_base, H, W):
    """One view per camera: stage1 extrinsic as viewmat + intrinsics + GT image."""
    views = []
    for ci, c in enumerate(s1_cams):
        viewmat = s1_ext[ci, frame_to_s1[frame]].astype(np.float32)
        gt = Image.open(Path(gt_base) / c / f"{frame:06d}.png").resize((W, H), Image.BILINEAR)
        views.append({
            "cam": c,
            "viewmat": torch.from_numpy(viewmat).to(DEVICE),
            "K": torch.from_numpy(cam_K[c].astype(np.float32)).to(DEVICE),
            "gt": torch.from_numpy(np.asarray(gt).astype(np.float32) / 255.0).to(DEVICE),
        })
    return views


def refine_frame(frame, init_npz, views, steps, out_path, H, W, log_every=100):
    data = np.load(init_npz, allow_pickle=True)
    gs = GSParams(
        data["means"].astype(np.float32), data["scales"].astype(np.float32),
        data["rotations"].astype(np.float32), data["opacities"].astype(np.float32),
        data["harmonics_dc"].astype(np.float32),
    ).to(DEVICE)
    optim = torch.optim.Adam([
        {"params": [gs.means], "lr": LR["means"]},
        {"params": [gs.log_scales], "lr": LR["log_scales"]},
        {"params": [gs.quats], "lr": LR["quats"]},
        {"params": [gs.logit_op], "lr": LR["logit_op"]},
        {"params": [gs.harm_dc], "lr": LR["harm_dc"]},
    ], eps=1e-15)

    init_loss = None
    t0 = time.time()
    for step in range(steps):
        total = sum(torch.mean(torch.abs(gs.render(v["viewmat"], v["K"], H, W) - v["gt"]))
                    for v in views) / len(views)
        optim.zero_grad(); total.backward(); optim.step()
        if init_loss is None:
            init_loss = float(total.item())
        if step == 0 or (step + 1) % log_every == 0 or step == steps - 1:
            print(f"    step {step+1:4d}/{steps}: L1={total.item():.4f}")

    with torch.no_grad():
        per_cam = {v["cam"]: float(torch.mean(torch.abs(
            gs.render(v["viewmat"], v["K"], H, W) - v["gt"])).item()) for v in views}
    final_loss = float(np.mean(list(per_cam.values())))

    out = gs.export()
    np.savez(out_path, frame_index=frame,
             cameras=np.array([v["cam"] for v in views], dtype=object),
             refine_steps=steps, refine_init_loss=init_loss, refine_final_loss=final_loss,
             refine_per_cam_l1_final=np.array(list(per_cam.values()), dtype=np.float32),
             **out)
    return {"frame": frame, "init_loss": init_loss, "final_loss": final_loss,
            "n_gs": int(len(out["opacities"])), "per_cam_l1_final": per_cam,
            "elapsed_s": time.time() - t0}


def run_stage2(stage1_dir, init_from, gt_base, frames, steps, resize_hw, out_dir,
               log_every, skip_existing):
    H, W = resize_hw
    out_dir = Path(out_dir)
    per_frame_dir = out_dir / "per_frame_gaussians"
    per_frame_dir.mkdir(parents=True, exist_ok=True)

    s1 = np.load(Path(stage1_dir) / "stage1_global_poses.npz", allow_pickle=True)
    s1_cams = list(s1["cameras"])
    s1_ext = s1["extrinsics_global"]
    frame_to_s1 = {int(fr): i for i, fr in enumerate(s1["frame_indices"])}
    cam_K = load_intrinsics(init_from, s1_cams)

    # Copy intrinsics into the output dir so downstream tools can find them.
    for c in s1_cams:
        src = Path(init_from) / f"intrinsics_{c}.npy"
        if src.exists():
            (out_dir / f"intrinsics_{c}.npy").write_bytes(src.read_bytes())

    print(f"  cams={s1_cams}  frames={frames[0]}..{frames[-1]} ({len(frames)})  "
          f"{H}x{W}  steps={steps}")

    summary, t_start = [], time.time()
    for idx, f in enumerate(frames):
        out_npz = per_frame_dir / f"frame_{f:06d}.npz"
        if skip_existing and out_npz.exists():
            print(f"  [{idx+1}/{len(frames)}] frame {f} — skip (exists)")
            continue
        init_npz = Path(init_from) / "per_frame_gaussians" / f"frame_{f:06d}.npz"
        views = build_views(f, s1_cams, s1_ext, frame_to_s1, cam_K, gt_base, H, W)
        s = refine_frame(f, init_npz, views, steps, out_npz, H, W, log_every)
        summary.append(s)
        print(f"  [{idx+1}/{len(frames)}] frame {f}: L1 {s['init_loss']:.4f} → "
              f"{s['final_loss']:.4f}  ({s['elapsed_s']:.1f}s)")

    (out_dir / "refine_summary.json").write_text(json.dumps({
        "init_from": str(init_from), "stage1": str(stage1_dir), "steps": steps,
        "resize_hw": [H, W], "total_time_s": time.time() - t_start,
        "mean_init_loss": float(np.mean([s["init_loss"] for s in summary])) if summary else None,
        "mean_final_loss": float(np.mean([s["final_loss"] for s in summary])) if summary else None,
        "per_frame": summary,
    }, indent=2))
    print(f"  saved {out_dir / 'refine_summary.json'}")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: photometric refinement")
    p.add_argument("--stage1", required=True, help="Stage 1 pose output dir")
    p.add_argument("--init-from", required=True, help="Gaussian-init output dir (per_frame_gaussians/)")
    p.add_argument("--gt-base", required=True, help="{gt_base}/{cam}/{frame:06d}.png ground-truth RGB")
    p.add_argument("--frames", required=True)
    p.add_argument("--steps", type=int, default=2800, help="full optimisation; 800 = warm-start")
    p.add_argument("--resize-hw", default="280,504", help="render H,W")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    H, W = (int(x) for x in a.resize_hw.split(","))
    run_stage2(a.stage1, a.init_from, a.gt_base, parse_frames(a.frames), a.steps,
               (H, W), a.out, a.log_every, a.skip_existing)


if __name__ == "__main__":
    main()
