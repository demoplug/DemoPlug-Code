"""SE(2)/SE(3) pose representations and interpolation for trajectory augmentation.

A mobile-base pose lives in SE(2), stored as ``(x, y, yaw)``. An end-effector pose
lives in SE(3), stored as a 4x4 homogeneous matrix. The canonical embedding
``iota : SE(2) -> SE(3)`` lifts a base pose into 3D with zero height and a yaw-only
rotation about ``+z``.

Interpolation follows the trajectory interpolation specification: translation is linear and rotation is the
shortest-arc Slerp. This split interpolant stands in for the exact screw-motion
geodesic; the two agree to first order at the perturbation magnitudes used here.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def wrap_angle(a):
    """Wrap an angle into ``(-pi, pi]``."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def se2(x, y, yaw):
    """Pack an SE(2) pose as ``[x, y, yaw]`` with the yaw wrapped."""
    return np.array([float(x), float(y), wrap_angle(float(yaw))])


def se2_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def se2_mul(a, b):
    """Compose two SE(2) poses, ``a . b``."""
    t = a[:2] + se2_rot(a[2]) @ b[:2]
    return se2(t[0], t[1], a[2] + b[2])


def se2_inv(a):
    """Inverse of an SE(2) pose."""
    t = -se2_rot(-a[2]) @ a[:2]
    return se2(t[0], t[1], -a[2])


def iota(a):
    """Canonical embedding SE(2) -> SE(3) (4x4): zero height, yaw-only rotation."""
    M = np.eye(4)
    M[:2, :2] = se2_rot(a[2])
    M[:2, 3] = a[:2]
    return M


def interp_se2(a, b, lam):
    """Split SE(2) interpolation (the trajectory interpolation specification): linear translation, shortest-arc yaw."""
    t = (1.0 - lam) * a[:2] + lam * b[:2]
    yaw = a[2] + lam * wrap_angle(b[2] - a[2])
    return se2(t[0], t[1], yaw)


def interp_se3(A, B, lam):
    """Split SE(3) interpolation (the trajectory interpolation specification): linear translation, quaternion Slerp."""
    out = np.eye(4)
    out[:3, 3] = (1.0 - lam) * A[:3, 3] + lam * B[:3, 3]
    key = Rotation.from_matrix(np.stack([A[:3, :3], B[:3, :3]]))
    out[:3, :3] = Slerp([0.0, 1.0], key)([lam]).as_matrix()[0]
    return out
