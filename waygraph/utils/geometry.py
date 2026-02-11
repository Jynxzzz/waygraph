"""
Geometric Utility Functions
=============================

Common geometric operations used throughout WayGraph: angle computation,
polyline measurements, and coordinate transformations.
"""

from typing import List

import numpy as np
from scipy.ndimage import uniform_filter1d


def angular_diff(a: float, b: float) -> float:
    """Compute minimum angular difference between two angles.

    Args:
        a: First angle in degrees.
        b: Second angle in degrees.

    Returns:
        Minimum angular gap in [0, 180] degrees.
    """
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)


def circular_mean(angles: List[float]) -> float:
    """Compute circular mean of angles in degrees.

    Args:
        angles: List of angles in degrees.

    Returns:
        Circular mean angle in degrees [-180, 180].
    """
    rads = np.radians(angles)
    mean_sin = np.mean(np.sin(rads))
    mean_cos = np.mean(np.cos(rads))
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)))


def normalize_angle(angle_deg: float) -> float:
    """Normalize angle to [-180, 180] degrees.

    Args:
        angle_deg: Angle in degrees (any range).

    Returns:
        Equivalent angle in [-180, 180].
    """
    angle = angle_deg % 360.0
    if angle > 180.0:
        angle -= 360.0
    return angle


def polyline_length(polyline: np.ndarray) -> float:
    """Compute total arc length of a 2D polyline.

    Args:
        polyline: (N, 2) array of points.

    Returns:
        Total length. Returns 0.0 if fewer than 2 points.
    """
    if len(polyline) < 2:
        return 0.0
    diffs = np.diff(polyline, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def polyline_curvature(
    polyline: np.ndarray,
    smoothing_window: int = 5,
) -> np.ndarray:
    """Compute discrete curvature along a 2D polyline.

    Uses the formula: kappa = |dx * d2y - dy * d2x| / (dx^2 + dy^2)^1.5

    Args:
        polyline: (N, 2) array of points (N >= 3).
        smoothing_window: Window size for uniform smoothing. Set to 0
            to disable smoothing.

    Returns:
        Array of curvature values with length N-2. Returns empty array
        if N < 3.
    """
    if len(polyline) < 3:
        return np.array([])

    dx = np.diff(polyline[:, 0])
    dy = np.diff(polyline[:, 1])
    d2x = np.diff(dx)
    d2y = np.diff(dy)

    dx_mid = (dx[:-1] + dx[1:]) / 2.0
    dy_mid = (dy[:-1] + dy[1:]) / 2.0

    denom = (dx_mid**2 + dy_mid**2) ** 1.5
    denom = np.maximum(denom, 1e-10)
    kappa = np.abs(dx_mid * d2y - dy_mid * d2x) / denom

    if smoothing_window > 0 and len(kappa) >= smoothing_window:
        kappa = uniform_filter1d(kappa, smoothing_window)

    return kappa


def rotate_points(
    points: np.ndarray,
    angle_deg: float,
    center: np.ndarray = None,
) -> np.ndarray:
    """Rotate 2D points around a center.

    Args:
        points: (N, 2) array of points.
        angle_deg: Rotation angle in degrees (counterclockwise positive).
        center: (2,) center of rotation. Defaults to origin.

    Returns:
        (N, 2) array of rotated points.
    """
    if center is None:
        center = np.zeros(2)

    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    centered = points - center
    rotated = (R @ centered.T).T
    return rotated + center
