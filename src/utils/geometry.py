"""Geometry utilities for coordinate transforms."""
import torch
import numpy as np


def angle_to_rotation_matrix(angle: float) -> np.ndarray:
    """Create 2D rotation matrix from angle (radians).

    Args:
        angle: angle in radians

    Returns:
        (2, 2) rotation matrix
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)


def rotation_matrix_to_angle(R: np.ndarray) -> float:
    """Extract angle from 2D rotation matrix.

    Args:
        R: (2, 2) rotation matrix

    Returns:
        angle in radians
    """
    return np.arctan2(R[1, 0], R[0, 0])


def local_to_world(points: np.ndarray, origin: np.ndarray, angle: float) -> np.ndarray:
    """Transform points from local to world coordinates.

    Args:
        points: (N, 2) points in local coordinates
        origin: (N, 2) origin in world coordinates
        angle: float heading angle in radians

    Returns:
        (N, 2) points in world coordinates
    """
    R = angle_to_rotation_matrix(angle)
    world_points = points @ R.T + origin
    return world_points


def world_to_local(points: np.ndarray, origin: np.ndarray, angle: float) -> np.ndarray:
    """Transform points from world to local coordinates.

    Args:
        points: (N, 2) points in world coordinates
        origin: (N, 2) origin in world coordinates
        angle: float heading angle in radians

    Returns:
        (N, 2) points in local coordinates
    """
    R = angle_to_rotation_matrix(angle)
    local_points = (points - origin) @ R.T
    return local_points


def compute_heading_from_velocity(velocities: np.ndarray) -> np.ndarray:
    """Compute heading angles from velocity vectors.

    Args:
        velocities: (N, 2) array of velocity vectors

    Returns:
        (N,) array of heading angles in radians
    """
    return np.arctan2(velocities[:, 1], velocities[:, 0])


def compute_heading_diff(heading1: np.ndarray, heading2: np.ndarray) -> np.ndarray:
    """Compute shortest angular difference between two heading arrays.

    Args:
        heading1: (N,) first heading angles
        heading2: (N,) second heading angles

    Returns:
        (N,) shortest angular differences
    """
    diff = heading1 - heading2
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return diff


def angular_distance(angle1: float, angle2: float) -> float:
    """Compute shortest angular distance between two angles.

    Args:
        angle1: first angle in radians
        angle2: second angle in radians

    Returns:
        shortest distance in radians
    """
    diff = angle1 - angle2
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return np.abs(diff)


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] using vectorized modulo.

    Args:
        angle: angle in radians

    Returns:
        normalized angle in [-pi, pi]
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_displacement(positions: np.ndarray) -> np.ndarray:
    """Compute displacement vectors from positions.

    Args:
        positions: (N, T, 2) array of positions over time

    Returns:
        (N, T-1, 2) array of displacement vectors
    """
    return positions[:, 1:] - positions[:, :-1]


def compute_velocities(positions: np.ndarray, dt: float = 0.5) -> np.ndarray:
    """Compute velocities from positions.

    Args:
        positions: (N, T, 2) array of positions
        dt: time step in seconds (nuScenes = 0.5s at 2Hz)

    Returns:
        (N, T-1, 2) array of velocities
    """
    displacements = compute_displacement(positions)
    return displacements / dt


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] for torch tensor.

    Args:
        angles: torch.Tensor of any shape

    Returns:
        wrapped angles in [-pi, pi]
    """
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def homogeneous_transform(points: torch.Tensor, translation: torch.Tensor,
                          rotation: torch.Tensor) -> torch.Tensor:
    """Apply homogeneous transformation to points.

    Args:
        points: (..., 2) points
        translation: (2,) translation vector
        rotation: (2, 2) rotation matrix

    Returns:
        Transformed points
    """
    return points @ rotation.T + translation


def torch_rotation_matrix(angle: torch.Tensor) -> torch.Tensor:
    """Create 2D rotation matrix from angle tensor.

    Args:
        angle: (B,) tensor of angles

    Returns:
        (B, 2, 2) rotation matrices
    """
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    R = torch.stack([
        torch.stack([cos_a, -sin_a], dim=-1),
        torch.stack([sin_a, cos_a], dim=-1)
    ], dim=-2)
    return R


def agent_centric_transform(positions: torch.Tensor, origin_idx: int = 0) -> torch.Tensor:
    """Transform positions to agent-centric coordinates.

    Takes world coordinates and centers/rotates so the agent is at origin
    facing +X direction. This makes the model rotation and translation invariant.

    Args:
        positions: (B, T, 2) positions in world frame
        origin_idx: time step to use as origin (default 0 = current time)

    Returns:
        transformed: (B, T, 2) positions in agent-centric frame
        origin: (B, 1, 2) world position at origin
        angle: (B,) heading angles at origin
    """
    origin = positions[:, origin_idx:origin_idx+1]
    if origin_idx > 0:
        heading_vec = positions[:, origin_idx] - positions[:, origin_idx - 1]
    else:
        heading_vec = positions[:, origin_idx + 1] - positions[:, origin_idx]

    angle = torch.atan2(heading_vec[:, 1], heading_vec[:, 0])
    R = torch_rotation_matrix(-angle)

    centered = positions - origin
    transformed = centered @ R.transpose(-2, -1)
    return transformed, origin, angle


def torch_world_to_local(points: torch.Tensor, origin: torch.Tensor,
                          angle: torch.Tensor) -> torch.Tensor:
    """Transform points from world to local coordinates using torch.

    Args:
        points: (B, T, 2) or (B, 2) points in world coordinates
        origin: (B, 1, 2) or (B, 2) origin in world coordinates
        angle: (B,) heading angle at origin

    Returns:
        Transformed points in local coordinates
    """
    R = torch_rotation_matrix(-angle)
    if points.dim() == 2:
        centered = points - origin.squeeze(-2)
        return centered @ R.transpose(-2, -1)
    else:
        centered = points - origin
        return centered @ R.transpose(-2, -1)


def torch_local_to_world(points: torch.Tensor, origin: torch.Tensor,
                          angle: torch.Tensor) -> torch.Tensor:
    """Transform points from local to world coordinates using torch.

    Args:
        points: (B, T, 2) or (B, 2) points in local coordinates
        origin: (B, 1, 2) or (B, 2) origin in world coordinates
        angle: (B,) heading angle at origin

    Returns:
        Transformed points in world coordinates
    """
    R = torch_rotation_matrix(angle)
    if points.dim() == 2:
        rotated = points @ R.transpose(-2, -1)
        return rotated + origin.squeeze(-2)
    else:
        rotated = points @ R.transpose(-2, -1)
        return rotated + origin
