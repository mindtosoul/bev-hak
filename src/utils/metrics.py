"""Metrics for trajectory prediction: ADE, FDE, minADE@K, minFDE@K."""
import torch
import numpy as np
from typing import Tuple, Optional


def ade_loss(predictions: torch.Tensor, targets: torch.Tensor,
             reduction: str = 'mean') -> torch.Tensor:
    """Average Displacement Error.

    Args:
        predictions: (B, K, T, 2) predicted trajectories (K modes, T timesteps)
        targets: (B, T, 2) ground truth future trajectories
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar ADE loss
    """
    B, K, T, _ = predictions.shape
    targets_expanded = targets.unsqueeze(1).expand(B, K, T, 2)
    errors = torch.norm(predictions - targets_expanded, dim=-1)
    ade = errors.mean(dim=-1)
    if reduction == 'mean':
        return ade.mean()
    elif reduction == 'sum':
        return ade.sum()
    return ade


def fde_loss(predictions: torch.Tensor, targets: torch.Tensor,
             reduction: str = 'mean') -> torch.Tensor:
    """Final Displacement Error.

    Args:
        predictions: (B, K, T, 2) predicted trajectories
        targets: (B, T, 2) ground truth future trajectories
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Scalar FDE loss
    """
    B, K, T, _ = predictions.shape
    targets_expanded = targets.unsqueeze(1).expand(B, K, T, 2)
    final_pred = predictions[:, :, -1, :]
    final_target = targets_expanded[:, :, -1, :]
    errors = torch.norm(final_pred - final_target, dim=-1)
    if reduction == 'mean':
        return errors.mean()
    elif reduction == 'sum':
        return errors.sum()
    return errors


def min_ade_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 3,
              scores: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute minADE@K (minimum ADE among top-K modes).

    If scores are provided, sorts predictions by score before taking top-K.
    Otherwise, assumes first K predictions are the best.

    Args:
        predictions: (B, K_full, T, 2) predicted trajectories
        targets: (B, T, 2) ground truth future trajectories
        k: number of top modes to consider
        scores: (B, K_full) optional scores/probabilities for sorting

    Returns:
        (minADE, best_indices) where minADE is (B,) and best_indices is (B,)
    """
    B, K_full, T, _ = predictions.shape
    K = min(k, K_full)

    if scores is not None:
        _, topk_idx = scores.topk(K, dim=-1, largest=True)
        batch_idx = torch.arange(B, device=predictions.device).unsqueeze(1).expand(B, K)
        preds_k = predictions[batch_idx, topk_idx]
    else:
        preds_k = predictions[:, :K]

    targets_expanded = targets.unsqueeze(1).expand(B, K, T, 2)
    errors = torch.norm(preds_k - targets_expanded, dim=-1)
    ade = errors.mean(dim=-1)
    min_ade, best_idx = ade.min(dim=-1)
    return min_ade, best_idx


def min_fde_k(predictions: torch.Tensor, targets: torch.Tensor, k: int = 3,
               scores: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute minFDE@K (minimum FDE among top-K modes).

    Args:
        predictions: (B, K_full, T, 2) predicted trajectories
        targets: (B, T, 2) ground truth future trajectories
        k: number of top modes to consider
        scores: (B, K_full) optional scores/probabilities for sorting

    Returns:
        (minFDE, best_indices)
    """
    B, K_full, T, _ = predictions.shape
    K = min(k, K_full)

    if scores is not None:
        _, topk_idx = scores.topk(K, dim=-1, largest=True)
        batch_idx = torch.arange(B, device=predictions.device).unsqueeze(1).expand(B, K)
        preds_k = predictions[batch_idx, topk_idx]
    else:
        preds_k = predictions[:, :K]

    final_pred = preds_k[:, :, -1, :]
    final_target = targets.unsqueeze(1).expand(B, K, 2)
    errors = torch.norm(final_pred - final_target, dim=-1)
    min_fde, best_idx = errors.min(dim=-1)
    return min_fde, best_idx


def weighted_ade_k(predictions: torch.Tensor, targets: torch.Tensor,
                   scores: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Compute probability-weighted ADE@K (pADE@K).

    Weights each mode's ADE by its probability/score.
    This is commonly used in Kaggle leaderboards for multi-modal prediction.

    Args:
        predictions: (B, K, T, 2) predicted trajectories
        targets: (B, T, 2) ground truth future trajectories
        scores: (B, K) probabilities/logits for each mode (will be softmax'd)

    Returns:
        Weighted ADE (scalar)
    """
    B, K, T, _ = predictions.shape

    probs = torch.softmax(scores, dim=-1)

    targets_expanded = targets.unsqueeze(1).expand(B, K, T, 2)
    errors = torch.norm(predictions - targets_expanded, dim=-1)
    ade_per_mode = errors.mean(dim=-1)

    weighted_ade = (ade_per_mode * probs).sum(dim=-1)
    return weighted_ade.mean()


def compute_ade_fde_single(pred_traj: np.ndarray, gt_traj: np.ndarray) -> Tuple[float, float]:
    """Compute ADE and FDE for a single trajectory pair.

    Args:
        pred_traj: (T, 2) predicted trajectory
        gt_traj: (T, 2) ground truth trajectory

    Returns:
        (ADE, FDE)
    """
    ade = np.mean(np.linalg.norm(pred_traj - gt_traj, axis=-1))
    fde = np.linalg.norm(pred_traj[-1] - gt_traj[-1])
    return ade, fde


def miss_rate(predictions: torch.Tensor, targets: torch.Tensor,
              threshold: float = 2.0, scores: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute miss rate (percentage of predictions beyond threshold at final step).

    Args:
        predictions: (B, K, T, 2)
        targets: (B, T, 2)
        threshold: distance threshold in meters
        scores: (B, K) optional scores for sorting

    Returns:
        miss rate (scalar)
    """
    B, K, T, _ = predictions.shape

    if scores is not None:
        _, topk_idx = scores.topk(1, dim=-1, largest=True)
        batch_idx = torch.arange(B, device=predictions.device)
        final_pred = predictions[batch_idx, topk_idx.squeeze(-1), -1, :]
    else:
        final_pred = predictions[:, 0, -1, :]

    final_target = targets[:, -1, :]
    errors = torch.norm(final_pred - final_target, dim=-1)
    return (errors > threshold).float().mean()


class TrajectoryMetrics:
    """Track and compute trajectory prediction metrics."""

    def __init__(self, k_modes: int = 3):
        self.k_modes = k_modes
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.min_ade_scores = []
        self.min_fde_scores = []
        self.miss_rates = []
        self.weighted_ade_scores = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               scores: Optional[torch.Tensor] = None):
        """Update metrics with batch predictions.

        Args:
            predictions: (B, K, T, 2) predicted trajectories
            targets: (B, T, 2) ground truth trajectories
            scores: (B, K) optional scores/probabilities
        """
        B, K, T, _ = predictions.shape

        if scores is not None:
            _, topk_idx = scores.topk(self.k_modes, dim=-1, largest=True)
            batch_idx = torch.arange(B, device=predictions.device).unsqueeze(1).expand(B, self.k_modes)
            preds_k = predictions[batch_idx, topk_idx]
        else:
            preds_k = predictions[:, :self.k_modes]

        targets_exp = targets.unsqueeze(1).expand(B, self.k_modes, T, 2)
        ade_per_mode = torch.norm(preds_k - targets_exp, dim=-1).mean(dim=-1)
        fde_per_mode = torch.norm(preds_k[:, :, -1, :] - targets_exp[:, :, -1, :], dim=-1)

        min_ade, _ = ade_per_mode.min(dim=-1)
        min_fde, _ = fde_per_mode.min(dim=-1)

        self.min_ade_scores.append(min_ade.detach().cpu().numpy())
        self.min_fde_scores.append(min_fde.detach().cpu().numpy())

        final_errors = fde_per_mode.min(dim=-1)[0]
        self.miss_rates.append((final_errors > 2.0).float().mean().item())

        if scores is not None:
            w_ade = weighted_ade_k(predictions, targets, scores, k=self.k_modes)
            self.weighted_ade_scores.append(w_ade.detach().cpu().numpy())

    def compute(self) -> dict:
        """Compute final metrics."""
        all_min_ade = np.concatenate(self.min_ade_scores)
        all_min_fde = np.concatenate(self.min_fde_scores)

        metrics = {
            'minADE@{}'.format(self.k_modes): float(np.mean(all_min_ade)),
            'minFDE@{}'.format(self.k_modes): float(np.mean(all_min_fde)),
            'miss_rate@2m': float(np.mean(self.miss_rates)),
            'ADE_std': float(np.std(all_min_ade)),
            'FDE_std': float(np.std(all_min_fde)),
        }

        if self.weighted_ade_scores:
            all_w_ade = np.concatenate(self.weighted_ade_scores)
            metrics['pADE@{}'.format(self.k_modes)] = float(np.mean(all_w_ade))

        return metrics

    def __repr__(self):
        metrics = self.compute()
        return ', '.join(['{}: {:.4f}'.format(k, v) for k, v in metrics.items()])


def batched_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                    k: int = 3, scores: Optional[torch.Tensor] = None) -> dict:
    """Compute all metrics for a batch.

    Args:
        predictions: (B, K, T, 2)
        targets: (B, T, 2)
        k: number of modes for min metrics
        scores: (B, K) optional scores/probabilities

    Returns:
        dict of metrics
    """
    B, K_full, T, _ = predictions.shape
    K = min(k, K_full)

    if scores is not None:
        _, topk_idx = scores.topk(K, dim=-1, largest=True)
        batch_idx = torch.arange(B, device=predictions.device).unsqueeze(1).expand(B, K)
        preds_k = predictions[batch_idx, topk_idx]
    else:
        preds_k = predictions[:, :K]

    targets_exp = targets.unsqueeze(1).expand(B, K, T, 2)

    ade_per_mode = torch.norm(preds_k - targets_exp, dim=-1).mean(dim=-1)
    fde_per_mode = torch.norm(preds_k[:, :, -1, :] - targets_exp[:, :, -1, :], dim=-1)

    min_ade, best_ade_idx = ade_per_mode.min(dim=-1)
    min_fde, best_fde_idx = fde_per_mode.min(dim=-1)

    metrics = {
        'minADE@{}'.format(k): min_ade.mean().item(),
        'minFDE@{}'.format(k): min_fde.mean().item(),
        'miss_rate@2m': (min_fde > 2.0).float().mean().item(),
        'best_mode_rate': (best_ade_idx == 0).float().mean().item(),
    }

    if scores is not None:
        metrics['pADE@{}'.format(k)] = weighted_ade_k(predictions, targets, scores, k).item()

    return metrics
