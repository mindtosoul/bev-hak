"""Multi-modal trajectory sampling and diversity strategies."""
import torch
import numpy as np
from typing import Tuple


def sample_cvae(model, batch: dict, num_samples: int = 30,
                device: torch.device = None) -> torch.Tensor:
    """Generate multiple trajectory samples from CVAE model.

    Args:
        model: CVAE trajectory model
        batch: dict with 'history' tensor (B, hist_len, 2)
        num_samples: number of trajectories to sample
        device: target device

    Returns:
        (B, num_samples, future_len, 2) predicted trajectories
    """
    if device is None:
        device = batch['history'].device

    B = batch['history'].shape[0]
    future_len = batch['future'].shape[1] if 'future' in batch else 6

    all_trajectories = []

    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(B, model.latent_dim, device=device)
            traj = model.decode(batch['history'], z)
            all_trajectories.append(traj)

    return torch.stack(all_trajectories, dim=1)


def farthest_point_sampling(points: torch.Tensor, k: int) -> torch.Tensor:
    """GPU-native Farthest Point Sampling for diverse trajectory selection.

    Faster than KMeans on GPU. Selects k points that are maximally far apart.
    This ensures the selected trajectories represent fundamentally different futures.

    Args:
        points: (B, N, D) points (e.g., endpoints)
        k: number of points to select

    Returns:
        (B, k) indices of selected points
    """
    B, N, D = points.shape
    device = points.device

    selected_indices = torch.zeros(B, k, dtype=torch.long, device=device)
    distances = torch.full((B, N), float('inf'), device=device)

    batch_indices = torch.arange(B, device=device)

    idx0 = torch.randint(N, (B,), device=device)
    selected_indices[:, 0] = idx0

    points_expanded = points.view(B, N, 1, D)
    selected_expanded = points[batch_indices, idx0].view(B, 1, D)
    dist_to_selected = torch.norm(points_expanded - selected_expanded, dim=-1)
    distances = torch.minimum(distances, dist_to_selected)

    for i in range(1, k):
        _, idx_i = distances.max(dim=-1)
        selected_indices[:, i] = idx_i

        selected_expanded = points[batch_indices, idx_i].view(B, 1, D)
        dist_to_selected = torch.norm(points_expanded - selected_expanded, dim=-1)
        distances = torch.minimum(distances, dist_to_selected)

    return selected_indices


def cluster_diverse_trajectories(trajectories: torch.Tensor,
                                k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select diverse trajectories using FPS (GPU-native) on endpoints.

    Uses Farthest Point Sampling instead of KMeans for:
    - GPU-native computation (no CPU transfer)
    - Faster execution
    - Better diversity coverage

    Args:
        trajectories: (B, N, T, 2) all N candidate trajectories
        k: number of diverse trajectories to select

    Returns:
        (B, k, T, 2) selected diverse trajectories, (B, k) indices
    """
    B, N, T, _ = trajectories.shape

    if N <= k:
        indices = torch.arange(N, device=trajectories.device).unsqueeze(0).expand(B, N)
        if N < k:
            pad_indices = torch.zeros(B, k - N, dtype=torch.long, device=trajectories.device)
            indices = torch.cat([indices, pad_indices], dim=1)
        batch_idx = torch.arange(B, device=trajectories.device).unsqueeze(1).expand(B, k)
        selected = trajectories[batch_idx, indices]
        return selected, indices

    endpoints = trajectories[:, :, -1, :]

    selected_indices = farthest_point_sampling(endpoints, k)

    batch_indices = torch.arange(B, device=trajectories.device).unsqueeze(1).expand(B, k)
    selected = trajectories[batch_indices, selected_indices]

    return selected, selected_indices


def select_top_k_by_score(trajectories: torch.Tensor, scores: torch.Tensor,
                          k: int = 3, higher_is_better: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top-k trajectories by score.

    Args:
        trajectories: (B, N, T, 2)
        scores: (B, N) scores (for softmax outputs: higher is better)
        k: number to select
        higher_is_better: if True, use largest=True for topk

    Returns:
        (B, k, T, 2), (B, k) indices
    """
    B, N, T, _ = trajectories.shape
    _, topk_idx = scores.topk(k, dim=-1, largest=higher_is_better)

    batch_indices = torch.arange(B, device=trajectories.device).unsqueeze(1).expand(B, k)
    selected = trajectories[batch_indices, topk_idx]
    return selected, topk_idx


def best_of_many(trajectories: torch.Tensor, targets: torch.Tensor,
                  k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select best k trajectories based on ADE against ground truth.

    Args:
        trajectories: (B, N, T, 2)
        targets: (B, T, 2)
        k: number to select

    Returns:
        (B, k, T, 2) selected trajectories, (B, k) indices
    """
    B, N, T, _ = trajectories.shape
    targets_exp = targets.unsqueeze(1).expand(B, N, T, 2)
    ade = torch.norm(trajectories - targets_exp, dim=-1).mean(dim=-1)
    _, best_idx = ade.topk(k, dim=-1, largest=False)
    batch_idx = torch.arange(B, device=trajectories.device).unsqueeze(1).expand(B, k)
    return trajectories[batch_idx, best_idx], best_idx


def trajectory_diversity(trajectories: torch.Tensor) -> float:
    """Compute diversity metric (mean pairwise distance at endpoints).

    Args:
        trajectories: (B, N, T, 2)

    Returns:
        mean pairwise endpoint distance
    """
    B, N, T, _ = trajectories.shape
    endpoints = trajectories[:, :, -1, :]
    pairwise_dist = torch.norm(endpoints.unsqueeze(2) - endpoints.unsqueeze(1), dim=-1)
    mask = torch.triu(torch.ones(N, N, device=trajectories.device), diagonal=1).bool()
    return pairwise_dist[mask].mean().item()


def goal_conditioned_selection(predictions: torch.Tensor, goal_scores: torch.Tensor,
                                k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top-k goal-conditioned trajectories by score.

    Goal scores from softmax are probabilities (higher is better).

    Args:
        predictions: (B, K, T, 2) trajectory predictions
        goal_scores: (B, K) softmax probabilities for each goal/mode

    Returns:
        selected trajectories and indices
    """
    return select_top_k_by_score(predictions, goal_scores, k, higher_is_better=True)


def diverse_topk_selection(trajectories: torch.Tensor, scores: torch.Tensor,
                           k: int = 3, diversity_weight: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top-k trajectories balancing score and diversity.

    Combines score-based selection with FPS diversity for Kaggle leaderboard.
    Higher diversity_weight = more diverse predictions (may sacrifice score).

    Args:
        trajectories: (B, N, T, 2) all candidate trajectories
        scores: (B, N) scores/probabilities (higher is better)
        k: number to select
        diversity_weight: weight for diversity (0-1), 0 = score only

    Returns:
        (B, k, T, 2), (B, k) indices
    """
    if diversity_weight == 0:
        return select_top_k_by_score(trajectories, scores, k, higher_is_better=True)

    score_selected, score_idx = select_top_k_by_score(trajectories, scores, k, higher_is_better=True)
    diverse_selected, diverse_idx = cluster_diverse_trajectories(trajectories, k)

    alpha = 1.0 - diversity_weight
    selected = alpha * score_selected + diversity_weight * diverse_selected

    combined_idx = torch.where(score_idx[:, :1] != 0, score_idx, diverse_idx)
    return selected, combined_idx
