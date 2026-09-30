# DemoPlug: Plug-and-Play Demonstration Augmentation

This repository contains the released implementation of the core DemoPlug data
generation pipeline for spatially generalized manipulation. It provides two
reusable pieces:

1. a Cartesian trajectory utility that samples mobile-base perturbations while
   preserving the source end-effector reference during interaction; and
2. a three-stage RGB-to-Gaussian reconstruction pipeline that estimates globally
   aligned camera poses, initializes per-frame Gaussian primitives, and refines
   them with a photometric objective.

The released scripts are research utilities. Robot-specific action export,
task-object editing, and the experiment's controller interface depend on the
robot platform and policy stack and are kept outside this repository. The
input/output formats below are the stable interfaces used by the released code.

### Project page

<https://demoplug.github.io/DemoPlug/>

![DemoPlug pipeline](pipeline.png)

## Repository layout

```
trajectory_augment.py   # Cartesian base/EEF reference augmentation
stage1_pose.py          # globally aligned camera-pose estimation
stage1_gaussians.py     # per-frame Gaussian initialization
stage2_refine.py        # photometric Gaussian refinement
utils/
├── geometry.py          # pose centers, alignment, and window stitching
├── gaussians.py         # Gaussian transforms and the gsplat renderer
├── se_math.py           # SE(2)/SE(3) operations and interpolation
└── io_utils.py          # frame loading and window specifications
```

## Requirements

- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
  (`depth_anything_3` must be importable; the scripts also look in a sibling
  `src/` directory)
- `torch`, `numpy`, `scipy`, `opencv-python`, and `pillow`
- `gsplat` for `stage2_refine.py`

The reconstruction stages are normally run with a CUDA-enabled PyTorch build.
The repository does not distribute model weights. Pass a compatible adapted DA3
checkpoint with `--finetuned-ckpt` when one is available; omitting it runs the
base DA3 model.

## Input formats

RGB frames are stored one camera at a time:

```
{frames_base}/{camera}/{frame:06d}.png
```

The default rig uses three cameras (`head,left_wrist,right_wrist`), but the
`--cameras` argument accepts any ordered camera list supported by the recording.
The trajectory utility reads one demonstration `.npz` containing:

```
base_se2  (H, 3)       # mobile-base x, y, yaw for each frame
eef_se3   (H, 4, 4)    # world-frame end-effector pose
gripper   (H,)         # gripper command/opening
```

## Quick start

Generate Cartesian reference trajectories:

```bash
python trajectory_augment.py \
  --source /data/clip/ep000/trajectory.npz --num-aug 4 \
  --r 0.12 --phi-deg 10 --epsilon 0.1 \
  --out /data/out/trajectory
```

Estimate globally aligned poses, initialize Gaussians, and refine them:

```bash
python stage1_pose.py \
  --frames-base /data/clip/ep000 \
  --cameras head,left_wrist,right_wrist \
  --frame-start 66 --frame-end 116 \
  --finetuned-ckpt /models/adapted_da3.pt \
  --out /data/out/stage1

python stage1_gaussians.py \
  --frames-base /data/clip/ep000 --stage1 /data/out/stage1 \
  --finetuned-ckpt /models/adapted_da3.pt \
  --frames 66-115 --out /data/out/gaussians

python stage2_refine.py \
  --stage1 /data/out/stage1 --init-from /data/out/gaussians \
  --gt-base /data/clip/ep000 --frames 66-115 \
  --out /data/out/refine
```

`stage1_pose.py` estimates metric camera poses in overlapping windows and stitches
them with rigid alignment. `stage1_gaussians.py` lifts the per-view primitives
into that global frame, removes low-opacity look-ahead primitives, and writes one
cloud per target frame. `stage2_refine.py` keeps the estimated camera poses fixed
and optimizes Gaussian means, scales, rotations, opacity, and color. Each stage
reads the previous stage's files, so an interrupted run can be resumed.

## Outputs

| Script | Output | Contents |
| --- | --- | --- |
| `trajectory_augment.py` | `augmented_{i:02d}.npz` | augmented base and EEF references, gripper sequence, body-local perturbations, and next-step Cartesian deltas |
| `stage1_pose.py` | `stage1_global_poses.npz` | camera names, frame indices, globally aligned world-to-camera extrinsics, and camera centers |
| `stage1_gaussians.py` | `per_frame_gaussians/frame_{:06d}.npz` | Gaussian means, scales, rotations, opacities, DC colors, camera IDs, and corresponding global camera poses |
| `stage2_refine.py` | `per_frame_gaussians/frame_{:06d}.npz` | refined Gaussian parameters and per-camera photometric losses |

The Cartesian trajectory output is policy-agnostic: downstream code can map the
references to absolute targets, pose increments, or controller velocities as
required by a robot platform.
