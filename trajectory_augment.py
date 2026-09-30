"""DemoPlug Cartesian trajectory augmentation utility.

Given one source demonstration's proprioceptive trajectory — per-frame mobile-base
pose ``T^b`` in SE(2), end-effector world pose ``T^g`` in SE(3), and gripper opening
``g`` — this emits ``N`` augmented trajectories that visit new base viewpoints while
preserving the source gripper-object world trajectory through the contact phase.

The demonstration is split at a phase boundary ``t*`` detected from a sustained
end-effector translation burst. In the approach phase the base is perturbed and the
end-effector is re-routed to reach the same contact pose; in the contact phase the
end-effector world pose is held identical to the source and the base carries a small
frozen perturbation. The per-frame body-local perturbations are written alongside
the augmented poses so a downstream robot or camera interface can apply them.

Usage:
    python trajectory_augment.py \
        --source /path/to/trajectory.npz --num-aug 4 \
        --r 0.12 --phi-deg 10 --epsilon 0.1 --out /path/to/out/trajectory
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.se_math import se2, se2_mul, se2_inv, iota, interp_se2, interp_se3


def detect_contact_frame(eef, nu, n_nu):
    """Phase-boundary heuristic based on sustained end-effector translation.

    ``t*`` is the first frame whose end-effector world-translation speed exceeds the
    threshold ``nu`` for ``n_nu`` consecutive frames — a sustained burst rather than a
    single-frame spike.
    """
    trans = eef[:, :3, 3]
    H = len(eef)
    v = np.zeros(H)
    v[1:] = np.linalg.norm(np.diff(trans, axis=0), axis=1)  # v[t] = |trans_t - trans_{t-1}|
    for t in range(1, H - n_nu + 1):
        if np.all(v[t:t + n_nu] > nu):
            return t
    return H // 2  # fallback if no sustained burst is found


def sample_se2_perturbation(r, phi, rng):
    """Planar sampling: planar translation uniform on a disk of radius ``r``, yaw in ``[-phi, phi]``."""
    radius = r * np.sqrt(rng.uniform())     # sqrt → uniform over disk area, not radius
    angle = rng.uniform(0.0, 2.0 * np.pi)
    dyaw = rng.uniform(-phi, phi)
    return se2(radius * np.cos(angle), radius * np.sin(angle), dyaw)


def eef_in_base(base_pose, eef_pose):
    """End-effector expressed in the base frame: ``T^{g/b} = iota(T^b)^{-1} . T^g``."""
    return np.linalg.inv(iota(base_pose)) @ eef_pose


def augment_trajectory(base, eef, grip, t_star, dT1, dT_star):
    """Apply the trajectory augmentation rule with sampled endpoint perturbations ``dT1`` (approach start) and
    ``dT_star`` (contact frame). Returns the augmented poses and the per-frame
    body-local perturbations consumed by the renderer."""
    H = len(base)
    gb = np.stack([eef_in_base(base[t], eef[t]) for t in range(H)])  # source T^{g/b}

    aug_base = np.zeros_like(base)              # T_hat^b   (SE2)
    dpert_base = np.zeros_like(base)            # dT_t^b    (SE2, body-local)
    for t in range(H):
        if t <= t_star:
            # Approach phase: base freely perturbed, blending dT1 -> dT_star along the
            # SE(2) geodesic (the trajectory interpolation specification: split interpolant ~ geodesic to first order).
            lam = t / t_star if t_star > 0 else 1.0
            dpert_base[t] = interp_se2(dT1, dT_star, lam)
        else:
            # Contact phase: a single frozen, small base perturbation.
            dpert_base[t] = dT_star
        aug_base[t] = se2_mul(base[t], dpert_base[t])    # T_hat^b = T^b . dT_t

    # Approach-phase gripper endpoints are pinned: at t=0 it rides the perturbed base,
    # at t* it is locked to the source contact pose. The interior is split-SE(3)
    # interpolated so the augmented arm reaches the *same* contact point.
    g_start = iota(aug_base[0]) @ gb[0]
    g_contact = eef[t_star]
    aug_eef = np.zeros((H, 4, 4))               # T_hat^g   (SE3)
    for t in range(H):
        if t <= t_star:
            lam = t / t_star if t_star > 0 else 1.0
            aug_eef[t] = interp_se3(g_start, g_contact, lam)
        else:
            aug_eef[t] = eef[t]                  # world-frame end-effector held at source

    # Consistency: T^{g/b} re-expressed against the perturbed base.
    aug_gb = np.stack([eef_in_base(aug_base[t], aug_eef[t]) for t in range(H)])
    # Gripper-frame perturbation consumed by Camera-pose update: dT^g = (T^g)^{-1} . T_hat^g
    # (identity through the contact phase, where the end-effector world pose is fixed).
    dpert_eef = np.stack([np.linalg.inv(eef[t]) @ aug_eef[t] for t in range(H)])

    return dict(aug_base=aug_base, aug_eef=aug_eef, aug_eef_in_base=aug_gb,
                dpert_base=dpert_base, dpert_eef=dpert_eef, gripper=grip.copy())


def derive_actions(aug_base, aug_eef):
    """Re-derive per-frame actions from the augmented state.

    The exact state-to-action map is policy-specific (absolute target, delta, etc.).
    Here we use the common next-step delta in each body's own frame; swap this for
    your policy's map.
    """
    H = len(aug_base)
    action_base = np.zeros_like(aug_base)
    action_eef = np.tile(np.eye(4), (H, 1, 1))
    for t in range(H - 1):
        action_base[t] = se2_mul(se2_inv(aug_base[t]), aug_base[t + 1])
        action_eef[t] = np.linalg.inv(aug_eef[t]) @ aug_eef[t + 1]
    return action_base, action_eef


def augment_camera_pose(cam_world, cam_offset, body_pert):
    """Camera-pose update: re-render pose for a camera rigidly attached to a perturbed body.

    ``cam_world`` is the source camera pose, ``cam_offset = T^{k/m}`` is the constant
    camera-to-body offset (estimated once at t=1), and ``body_pert`` is the body-local
    perturbation in SE(3) — ``iota(dT^b)`` for a base (head) camera, ``dT^g`` for a
    gripper (wrist) camera. The conjugation carries no time-varying stitching drift,
    so the augmented camera inherits the same global frame as the source.
    """
    return cam_world @ np.linalg.inv(cam_offset) @ body_pert @ cam_offset


def run(source, num_aug, r, phi, epsilon, nu, n_nu, seed, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(source, allow_pickle=True)
    base = d["base_se2"].astype(float)      # (H, 3) x, y, yaw
    eef = d["eef_se3"].astype(float)        # (H, 4, 4)
    grip = d["gripper"].astype(float)       # (H,)
    H = len(base)
    t_star = detect_contact_frame(eef, nu, n_nu)
    print(f"  H={H}  contact frame t*={t_star}  num_aug={num_aug}")

    for i in range(num_aug):
        rng = np.random.default_rng(seed + i)
        dT1 = sample_se2_perturbation(r, phi, rng)                       # wide, approach start
        dT_star = sample_se2_perturbation(r * epsilon, phi * epsilon, rng)  # small, contact frame
        traj = augment_trajectory(base, eef, grip, t_star, dT1, dT_star)
        action_base, action_eef = derive_actions(traj["aug_base"], traj["aug_eef"])
        np.savez(out_dir / f"augmented_{i:02d}.npz",
                 t_star=t_star, dT1=dT1, dT_star=dT_star,
                 action_base_se2=action_base, action_eef_se3=action_eef, **traj)
        print(f"  aug {i:02d}: dT1=({dT1[0]:+.3f}, {dT1[1]:+.3f}, "
              f"{np.degrees(dT1[2]):+.1f}deg)")
    print(f"  saved {num_aug} augmentations to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="DemoPlug trajectory generator (base perturbation)")
    p.add_argument("--source", required=True,
                   help="npz with base_se2 (H,3), eef_se3 (H,4,4), gripper (H,)")
    p.add_argument("--num-aug", type=int, default=4, help="augmented demos per source (N)")
    p.add_argument("--r", type=float, default=0.12, help="translation radius for dT1 (m)")
    p.add_argument("--phi-deg", type=float, default=10.0, help="yaw radius for dT1 (deg)")
    p.add_argument("--epsilon", type=float, default=0.1, help="contact-frame perturbation scale")
    p.add_argument("--nu", type=float, default=5e-3,
                   help="contact velocity threshold (m/frame; tune to the control rate)")
    p.add_argument("--n-nu", type=int, default=3, help="sustained-burst length for t* detection")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    run(a.source, a.num_aug, a.r, np.radians(a.phi_deg), a.epsilon,
        a.nu, a.n_nu, a.seed, a.out)


if __name__ == "__main__":
    main()
