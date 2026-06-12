"""Rigid / similarity transforms, Umeyama alignment, and cross-window chaining.

All poses are world-to-camera (w2c) extrinsics. A camera center in world space is
``c = -R^T t`` where ``w2c = [R | t]``. A *similarity* maps local coordinates to
global ones as ``p_global = s * R @ p_local + t``.
"""
import numpy as np


def w2c_to_cam_center(w2c: np.ndarray) -> np.ndarray:
    """World-space camera center from a (3,4) or (4,4) world-to-camera matrix."""
    R, t = w2c[:3, :3], w2c[:3, 3]
    return -R.T @ t


def ensure_4x4(ext: np.ndarray) -> np.ndarray:
    """Promote a (3,4) extrinsic to a (4,4) homogeneous matrix."""
    if ext.shape == (4, 4):
        return ext.astype(np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3] = ext
    return out


def apply_similarity_to_w2c(w2c_3x4: np.ndarray, scale: float,
                            R_sim: np.ndarray, t_sim: np.ndarray) -> np.ndarray:
    """Transform a w2c pose by a similarity acting on world coordinates.

    The similarity ``p_global = scale * R_sim @ p_local + t_sim`` is applied to the
    camera center; the camera's world->cam rotation becomes ``R_wc @ R_sim^T`` and
    the translation is recomputed so the new center matches.
    """
    w2c = ensure_4x4(w2c_3x4)
    R_wc, t_wc = w2c[:3, :3], w2c[:3, 3]
    c_local = -R_wc.T @ t_wc
    c_global = scale * (R_sim @ c_local) + t_sim
    new_R_wc = R_wc @ R_sim.T
    new_t_wc = -new_R_wc @ c_global
    out = np.eye(4)
    out[:3, :3] = new_R_wc
    out[:3, 3] = new_t_wc
    return out


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity (s, R, t) mapping ``src`` onto ``dst``.

    Returns ``(scale, R, t, residuals)`` such that ``dst ≈ scale * src @ R.T + t``.
    Set ``with_scale=False`` to estimate a rigid 6-DOF transform only (use this when
    each input already carries a correct absolute metric scale).
    """
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    sigma = (sc ** 2).sum() / n
    H = (sc.T @ dc) / n
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))  # reflection guard
    R = Vt.T @ S @ U.T
    scale = (np.diag(D) @ S).trace() / sigma if with_scale else 1.0
    t = mu_d - scale * R @ mu_s
    residuals = np.linalg.norm(scale * (src @ R.T) + t - dst, axis=1)
    return scale, R, t, residuals


def chain_align(windows_centers, overlap: int, n_cam: int = 1, with_scale: bool = True):
    """Stitch a list of per-window camera-center sets into one global frame.

    ``windows_centers[i]`` is ``(n_cam * n_frame_i, 3)`` with cameras concatenated
    along axis 0. Each window is aligned to the running global frame using the
    overlap frames shared with the previous window.

    Returns ``(aligned_centers, drifts_pct, transforms)`` where ``transforms[i]`` is
    the ``(scale, R, t)`` similarity mapping window ``i`` local coords to global.
    """
    K = len(windows_centers)
    aligned = [windows_centers[0].copy()]
    drifts = []
    transforms = [(1.0, np.eye(3), np.zeros(3))]

    for i in range(1, K):
        prev_w_len = len(aligned[-1]) // n_cam
        curr_w_len = len(windows_centers[i]) // n_cam
        eff_ov = min(overlap, prev_w_len, curr_w_len)
        prev_pts, curr_pts = [], []
        for c in range(n_cam):
            p = aligned[-1][c * prev_w_len:(c + 1) * prev_w_len][-eff_ov:]
            q = windows_centers[i][c * curr_w_len:(c + 1) * curr_w_len][:eff_ov]
            prev_pts.append(p)
            curr_pts.append(q)
        prev_pts, curr_pts = np.vstack(prev_pts), np.vstack(curr_pts)

        if len(curr_pts) < 3:  # too few overlap points → keep as-is
            aligned.append(windows_centers[i].copy())
            drifts.append(0.0)
            transforms.append((1.0, np.eye(3), np.zeros(3)))
            continue

        s, R, t, _ = umeyama(curr_pts, prev_pts, with_scale=with_scale)
        drifts.append(abs(1.0 - s) * 100.0)
        aligned.append(s * (windows_centers[i] @ R.T) + t)
        transforms.append((s, R, t))

    return aligned, drifts, transforms
