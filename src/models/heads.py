"""Prediction heads for trajectory models."""
import torch
import torch.nn as nn
from typing import Tuple


class CVAELatentHead(nn.Module):
    """CVAE latent space head for multi-modality."""

    def __init__(self, encoder_dim: int, latent_dim: int):
        super().__init__()
        self.mu_layer = nn.Linear(encoder_dim, latent_dim)
        self.logvar_layer = nn.Linear(encoder_dim, latent_dim)

    def forward(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute latent distribution parameters.

        Args:
            encoder_out: (B, encoder_dim) output from encoder

        Returns:
            (mu, logvar) of latent distribution
        """
        mu = self.mu_layer(encoder_out)
        logvar = self.logvar_layer(encoder_out)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std) * std + mu
        return z


class MixtureOfGaussiansHead(nn.Module):
    """Mixture of Gaussians head for multi-modal prediction.

    Properly samples from a mixture by:
    1. Picking a mode based on weights (categorical distribution)
    2. Sampling from that specific Gaussian
    """

    def __init__(self, input_dim: int, k_modes: int = 3, output_dim: int = 2):
        super().__init__()
        self.k_modes = k_modes
        self.output_dim = output_dim

        self.mean_layer = nn.Linear(input_dim, k_modes * output_dim)
        self.logvar_layer = nn.Linear(input_dim, k_modes * output_dim)
        self.weight_layer = nn.Linear(input_dim, k_modes)

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict mixture of Gaussians.

        Args:
            context: (B, input_dim) encoded context

        Returns:
            means: (B, k_modes, output_dim)
            logvars: (B, k_modes, output_dim)
            weights: (B, k_modes)
        """
        means = self.mean_layer(context).view(-1, self.k_modes, self.output_dim)
        logvars = torch.clamp(self.logvar_layer(context), min=-10, max=10).view(-1, self.k_modes, self.output_dim)
        weights = torch.softmax(self.weight_layer(context), dim=-1)

        return means, logvars, weights

    def sample(self, means: torch.Tensor, logvars: torch.Tensor,
               weights: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Sample from mixture of Gaussians correctly.

        Args:
            means: (B, k_modes, output_dim)
            logvars: (B, k_modes, output_dim)
            weights: (B, k_modes)
            num_samples: number of samples per batch element

        Returns:
            (B * num_samples, k_modes, output_dim) sampled trajectories
            OR (B, num_samples, k_modes, output_dim) if keepdims=True
        """
        B, k_modes, d = means.shape

        mode_indices = torch.multinomial(weights, num_samples, replacement=True)

        batch_indices = torch.arange(B, device=means.device).unsqueeze(1).expand(B, num_samples)
        flat_batch_indices = batch_indices.reshape(-1)
        flat_mode_indices = mode_indices.reshape(-1)

        selected_means = means[flat_batch_indices, flat_mode_indices]
        selected_logvars = logvars[flat_batch_indices, flat_mode_indices]

        stds = torch.exp(0.5 * selected_logvars).clamp(min=1e-5)
        noise = torch.randn_like(selected_means)
        samples = selected_means + noise * stds

        return samples.view(B, num_samples, k_modes, d)


class GoalConditionedHead(nn.Module):
    """Goal-conditioned head for endpoint prediction.

    Predicts K goal candidates with learned importance scores.
    Uses output_dim dynamically instead of hardcoded 2.
    """

    def __init__(self, input_dim: int, k_goals: int = 8, output_dim: int = 2):
        super().__init__()
        self.k_goals = k_goals
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, k_goals * (output_dim + 1))
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict goal candidates with scores.

        Args:
            context: (B, input_dim)

        Returns:
            goals: (B, k_goals, output_dim)
            scores: (B, k_goals)
        """
        B = context.size(0)
        output = self.fc(context)

        goal_chunk_size = self.k_goals * self.output_dim
        goals = output[:, :goal_chunk_size].view(B, self.k_goals, self.output_dim)
        raw_scores = output[:, goal_chunk_size:]
        scores = torch.softmax(raw_scores, dim=-1)

        return goals, scores


class GoalHead(nn.Module):
    """Goal-conditioned head for endpoint prediction.

    Alias for GoalConditionedHead for backwards compatibility.
    """

    def __init__(self, input_dim: int, k_candidates: int = 8, output_dim: int = 2):
        super().__init__()
        self.k_candidates = k_candidates
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, k_candidates * (output_dim + 1))
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict goal candidates with scores.

        Args:
            context: (B, input_dim)

        Returns:
            goals: (B, k_candidates, output_dim)
            scores: (B, k_candidates)
        """
        B = context.size(0)
        output = self.fc(context)

        goal_chunk_size = self.k_candidates * self.output_dim
        goals = output[:, :goal_chunk_size].view(B, self.k_candidates, self.output_dim)
        raw_scores = output[:, goal_chunk_size:]
        scores = torch.softmax(raw_scores, dim=-1)

        return goals, scores
