"""3D Gaussian primitive transforms and a differentiable gsplat renderer.

Gaussians are stored as: ``means (N,3)``, ``scales (N,3)``, ``rotations (N,4)``
quaternions in ``wxyz`` order, ``opacities (N,)`` in [0,1], and ``harmonics_dc
(N,3)`` (the DC SH band, i.e. base RGB before the SH ``C0`` factor).
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

try:
    import gsplat
except ImportError:  # rendering (stage2_refine) needs gsplat; transforms (stage1_gaussians) do not
    gsplat = None

C0 = 0.28209479  # SH DC band normalization factor


def quat_wxyz_to_xyzw(q):
    return np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)


def quat_xyzw_to_wxyz(q):
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def inverse_sigmoid(x, eps=1e-6):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def apply_sim_to_gs(means, scales, rot_wxyz, s, R, t):
    """Apply a similarity ``(s, R, t)`` to a set of Gaussians.

    Positions transform as points; isotropic scales multiply by ``s``; rotations
    are pre-composed with ``R``.
    """
    m_new = s * (means @ R.T) + t
    sc_new = s * scales
    rot_local = Rotation.from_quat(quat_wxyz_to_xyzw(rot_wxyz))
    rot_global = Rotation.from_matrix(R) * rot_local
    return m_new, sc_new, quat_xyzw_to_wxyz(rot_global.as_quat())


class GSParams(nn.Module):
    """A per-frame set of optimizable Gaussians, rendered with gsplat.

    Parameters are stored in their unconstrained form (log scales, logit opacity,
    raw quats) so plain Adam keeps them valid.
    """

    def __init__(self, means, scales, quats_wxyz, opacities, harm_dc):
        super().__init__()
        if gsplat is None:
            raise ImportError("gsplat is required for GSParams.render(); pip install gsplat")
        self.means = nn.Parameter(torch.from_numpy(means).float())
        self.log_scales = nn.Parameter(
            torch.from_numpy(np.log(np.clip(scales, 1e-6, None))).float())
        self.quats = nn.Parameter(torch.from_numpy(quats_wxyz).float())
        self.logit_op = nn.Parameter(torch.from_numpy(inverse_sigmoid(opacities)).float())
        self.harm_dc = nn.Parameter(torch.from_numpy(harm_dc).float())

    def render(self, viewmat, K, H, W):
        scales = torch.exp(self.log_scales)
        opacities = torch.sigmoid(self.logit_op)
        quats = self.quats / (self.quats.norm(dim=-1, keepdim=True) + 1e-8)
        colors = (self.harm_dc * C0 + 0.5).clamp(0, 1)
        r, _, _ = gsplat.rasterization(
            means=self.means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmat.unsqueeze(0), Ks=K.unsqueeze(0),
            width=W, height=H, render_mode="RGB", rasterize_mode="classic",
        )
        return r[0]

    def export(self):
        """Return numpy arrays in the canonical Gaussian storage format."""
        with torch.no_grad():
            q = self.quats / (self.quats.norm(dim=-1, keepdim=True) + 1e-8)
            return dict(
                means=self.means.detach().cpu().numpy().astype(np.float32),
                scales=torch.exp(self.log_scales).detach().cpu().numpy().astype(np.float32),
                rotations=q.detach().cpu().numpy().astype(np.float32),
                opacities=torch.sigmoid(self.logit_op).detach().cpu().numpy().astype(np.float32),
                harmonics_dc=self.harm_dc.detach().cpu().numpy().astype(np.float32),
            )
