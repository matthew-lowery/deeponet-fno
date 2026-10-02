from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CarData:
    train_points: np.ndarray
    train_targets: np.ndarray
    test_points: np.ndarray
    test_targets: np.ndarray
    grid_min: np.ndarray
    grid_scale: np.ndarray
    pressure_mean: float
    pressure_scale: float
    point_indices: np.ndarray


def load_car(path, ntrain=500, ntest=111, num_points=0):
    """DGPO split: first training cars, last test cars; unweighted point likelihood."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Car data missing: {path}; pass --data-path=/path/to/car.npz")
    with np.load(path) as data:
        points = data["grid"].astype(np.float32)
        pressure = data["press"].astype(np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("grid must have shape (cars, points, 3)")
    if pressure.size != points.shape[0] * points.shape[1]:
        raise ValueError("press must contain one pressure per surface point")
    pressure = pressure.reshape(*points.shape[:2], 1)
    if ntrain < 1 or ntest < 1 or ntrain + ntest > len(points):
        raise ValueError("training and test splits must be nonempty and disjoint")
    if not np.isfinite(points).all() or not np.isfinite(pressure).all():
        raise ValueError("car data contains nonfinite values")
    if not 0 <= num_points <= points.shape[1]:
        raise ValueError(f"num_points must be between 0 and {points.shape[1]}")
    indices = np.arange(points.shape[1])
    if num_points:
        indices = np.linspace(0, points.shape[1] - 1, num_points, dtype=int)
    points, pressure = points[:, indices], pressure[:, indices]
    grid_min = points[:ntrain].min(axis=(0, 1))
    grid_scale = np.maximum(points[:ntrain].max(axis=(0, 1)) - grid_min, 1e-7)
    points = (points - grid_min) / grid_scale
    mean = float(pressure[:ntrain].mean(dtype=np.float64))
    scale = float(pressure[:ntrain].std(dtype=np.float64)) + 1e-7
    return CarData(points[:ntrain], (pressure[:ntrain] - mean) / scale,
                   points[-ntest:], pressure[-ntest:], grid_min, grid_scale,
                   mean, scale, indices)
