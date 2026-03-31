"""Social-GRU CVAE model for multi-modal trajectory prediction.

Lightweight model optimized for RTX 3050 4GB.
Architecture:
- Encoder: 2-layer GRU on target history + vectorized neighbor GRU + Multi-Head Attention
- Recognition Network: encodes [history + future] → μ, σ
- Prior Network: encodes [history only] → μ_prior, σ_prior
- Latent: CVAE with KL divergence between Recognition and Prior
- Decoder: 1-layer GRU conditioned on z to predict future displacements

Key fixes from v1:
1. Vectorized neighbor encoding (no Python loops)
2. Recognition + Prior CVAE to prevent mode collapse
3. Multi-Head Attention instead of grid pooling for continuous social interaction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class MultiHeadSocialAttention(nn.Module):
    """Multi-Head Social Attention for continuous neighbor interactions.

    Replaces grid-based pooling with continuous attention over neighbors.
    Each head attends to different aspects of neighbor behavior.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ego_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, ego_features: torch.Tensor,
                neighbor_features: torch.Tensor,
                neighbor_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head social attention.

        Args:
            ego_features: (B, hidden_dim) ego agent features
            neighbor_features: (B, N, hidden_dim) neighbor features
            neighbor_mask: (B, N) True=valid neighbor

        Returns:
            output: (B, hidden_dim) attended features
            attn_weights: (B, N) attention weights
        """
        B, N, _ = neighbor_features.shape

        ego_q = self.ego_proj(ego_features).unsqueeze(1)
        neighbor_k = self.neighbor_proj(neighbor_features)
        neighbor_v = self.value_proj(neighbor_features)

        ego_q = ego_q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        neighbor_k = neighbor_k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        neighbor_v = neighbor_v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(ego_q, neighbor_k.transpose(-2, -1)) * self.scale

        if neighbor_mask is not None:
            mask_expanded = neighbor_mask.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, 1, N)
            attn_scores = attn_scores.masked_fill(~mask_expanded, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)

        attn_output = torch.matmul(attn_weights, neighbor_v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, self.hidden_dim)
        output = self.out_proj(attn_output).squeeze(1)

        attn_weights_avg = attn_weights.mean(dim=1).squeeze(-2)

        return output, attn_weights_avg


class VectorizedNeighborEncoder(nn.Module):
    """Vectorized neighbor encoder - no Python loops.

    Processes all neighbors in a single GPU operation.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.neighbor_rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

    def forward(self, neighbors: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        """Vectorized neighbor encoding.

        Args:
            neighbors: (B, max_neighbors, past_steps, input_dim)
            neighbor_mask: (B, max_neighbors)

        Returns:
            (B, max_neighbors, hidden_dim) encoded neighbors
        """
        B, max_n, steps, _ = neighbors.shape

        neighbors_flat = neighbors.view(B * max_n, steps, -1)
        mask_flat = neighbor_mask.view(B * max_n)

        outputs, _ = self.neighbor_rnn(neighbors_flat)

        final_hidden = outputs[:, -1, :].view(B, max_n, self.hidden_dim)

        mask_expanded = neighbor_mask.unsqueeze(-1).float()
        encoded = final_hidden * mask_expanded

        zero_mask = ~neighbor_mask
        if zero_mask.any():
            encoded = encoded.masked_fill(zero_mask.unsqueeze(-1), 0.0)

        return encoded


class MapEncoder(nn.Module):
    """CNN Encoder for semantic map patches.

    Encodes 3-channel semantic map (road, crossing, walkway) into a feature vector.
    """

    def __init__(self, map_channels: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(map_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.map_proj = nn.Linear(256 * 16, hidden_dim)
        self.map_norm = nn.LayerNorm(hidden_dim)

    def forward(self, map_patch: torch.Tensor) -> torch.Tensor:
        """Encode map patch.

        Args:
            map_patch: (B, 3, 64, 64) semantic map patch

        Returns:
            (B, hidden_dim) encoded map features (normalized to match GRU hidden_dim)
        """
        x = torch.relu(self.bn1(self.conv1(map_patch)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.map_proj(x)
        x = self.map_norm(x)
        return x


class SocialGRUCVAE(nn.Module):
    """Social-GRU CVAE with Recognition-Prior architecture for multi-modal prediction.

    Prevents mode collapse through:
    1. Recognition network: encodes [history + social + FUTURE] to get true posterior
    2. Prior network: encodes [history + social] only → what we have at inference
    3. KL divergence loss: forces prior to match recognition
       - This structures the latent space so sampling z from prior produces diverse modes

    Architecture:
    - Encoder: 2-layer GRU on target history (shared)
    - FutureEncoder: 1-layer GRU on future trajectories (recognition only)
    - MapEncoder: CNN on semantic map patches
    - Recognition Network: [history + social + map + future] → μ_r, σ_r
    - Prior Network: [history + social + map] → μ_p, σ_p
    - Latent: Reparameterize from recognition during training, prior during inference
    - Decoder: 1-layer GRU conditioned on z
    """

    def __init__(self,
                 input_dim: int = 2,
                 hidden_dim: int = 128,
                 latent_dim: int = 16,
                 output_dim: int = 2,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 1,
                 dropout: float = 0.1,
                 social_num_heads: int = 4,
                 max_neighbors: int = 15,
                 future_steps: int = 6,
                 use_map: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_neighbors = max_neighbors
        self.future_steps = future_steps
        self.use_map = use_map

        self.encoder_rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_encoder_layers,
            dropout=dropout if num_encoder_layers > 1 else 0,
            batch_first=True
        )

        self.neighbor_encoder = VectorizedNeighborEncoder(
            input_dim, hidden_dim, num_encoder_layers, dropout
        )

        self.social_attention = MultiHeadSocialAttention(
            hidden_dim, num_heads=social_num_heads, dropout=dropout
        )

        self.map_encoder = MapEncoder(map_channels=3, hidden_dim=hidden_dim) if use_map else None

        self.future_encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=1,
            dropout=0,
            batch_first=True
        )

        map_feat_dim = hidden_dim if use_map else 0
        recognition_input_dim = hidden_dim + hidden_dim + hidden_dim + map_feat_dim
        prior_input_dim = hidden_dim + hidden_dim + map_feat_dim

        self.recognition_mu = nn.Linear(recognition_input_dim, latent_dim)
        self.recognition_logvar = nn.Linear(recognition_input_dim, latent_dim)

        self.prior_mu = nn.Linear(prior_input_dim, latent_dim)
        self.prior_logvar = nn.Linear(prior_input_dim, latent_dim)

        self.decoder_rnn = nn.GRU(
            input_dim + latent_dim,
            hidden_dim,
            num_decoder_layers,
            dropout=dropout if num_decoder_layers > 1 else 0,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def encode_history(self, history: torch.Tensor) -> torch.Tensor:
        """Encode agent history.

        Args:
            history: (B, past_steps, input_dim)

        Returns:
            (B, hidden_dim) encoded history
        """
        _, history_enc = self.encoder_rnn(history)
        return history_enc[-1]

    def encode_social(self, ego_features: torch.Tensor,
                     neighbors: torch.Tensor,
                     neighbor_mask: torch.Tensor) -> torch.Tensor:
        """Encode social context with multi-head attention.

        Args:
            ego_features: (B, hidden_dim)
            neighbors: (B, max_neighbors, past_steps, input_dim)
            neighbor_mask: (B, max_neighbors)

        Returns:
            (B, hidden_dim) social context
        """
        neighbor_enc = self.neighbor_encoder(neighbors, neighbor_mask)
        social_features, _ = self.social_attention(ego_features, neighbor_enc, neighbor_mask)
        return social_features

    def encode_future(self, future: torch.Tensor) -> torch.Tensor:
        """Encode future trajectory (for recognition network only).

        Args:
            future: (B, future_steps, input_dim)

        Returns:
            (B, hidden_dim) encoded future
        """
        _, future_enc = self.future_encoder(future)
        return future_enc[-1]

    def recognition_network(self, history_enc: torch.Tensor,
                          social_context: torch.Tensor,
                          future_enc: torch.Tensor,
                          map_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recognition network: encodes [history + social + map + future] → posterior μ, σ.

        During training - sees both history AND future ground truth.
        This is what allows the model to learn that specific z values correspond
        to specific future outcomes (left turn vs right turn).
        """
        if self.use_map and map_features is not None:
            combined = torch.cat([history_enc, social_context, map_features, future_enc], dim=-1)
        else:
            combined = torch.cat([history_enc, social_context, future_enc], dim=-1)
        mu = self.recognition_mu(combined)
        logvar = self.recognition_logvar(combined)
        return mu, logvar

    def prior_network(self, history_enc: torch.Tensor,
                     social_context: torch.Tensor,
                     map_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prior network: encodes [history + social + map] only → prior μ, σ.

        This is what we have at inference time - we can only condition on history.
        The KL loss forces this to match the recognition network.
        """
        if self.use_map and map_features is not None:
            combined = torch.cat([history_enc, social_context, map_features], dim=-1)
        else:
            combined = torch.cat([history_enc, social_context], dim=-1)
        mu = self.prior_mu(combined)
        logvar = self.prior_logvar(combined)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for CVAE."""
        std = torch.exp(0.5 * logvar)
        z = torch.randn_like(std) * std + mu
        return z

    def decode(self, history: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent and history to future trajectory.

        Args:
            history: (B, past_steps, input_dim) target agent history
            z: (B, latent_dim) latent sample

        Returns:
            (B, future_steps, output_dim) predicted future displacements
        """
        B, past_steps, _ = history.shape

        decoder_input = torch.cat([history, z.unsqueeze(1).expand(-1, past_steps, -1)], dim=-1)
        _, hidden = self.decoder_rnn(decoder_input)

        predictions = []
        curr_pos = history[:, -1:]

        for t in range(self.future_steps):
            input_t = torch.cat([curr_pos.squeeze(1), z], dim=-1).unsqueeze(1)
            output, hidden = self.decoder_rnn(input_t, hidden)
            pred_t = self.fc_out(output)
            predictions.append(pred_t)
            curr_pos = pred_t

        return torch.cat(predictions, dim=1)

    def decode_wta(self, history: torch.Tensor, z_samples: torch.Tensor) -> torch.Tensor:
        """Decode K trajectory samples for WTA loss.

        Args:
            history: (B, past_steps, input_dim)
            z_samples: (B, K, latent_dim) K latent samples

        Returns:
            (B, K, future_steps, output_dim) K predicted trajectories
        """
        B, K, latent_dim = z_samples.shape
        _, past_steps, _ = history.shape

        all_trajs = []
        for k in range(K):
            z_k = z_samples[:, k]
            traj_k = self.decode(history, z_k)
            all_trajs.append(traj_k)

        return torch.stack(all_trajs, dim=1)

    def forward(self, batch: Dict[str, torch.Tensor],
                mode: str = 'train') -> Dict[str, torch.Tensor]:
        """Forward pass with WTA multi-sample training.

        Args:
            batch: dict with keys 'history', 'neighbors', 'neighbor_mask', 'future_rel', 'map'
            mode: 'train' or 'infer'

        Returns:
            dict with predictions, loss, and metrics
        """
        history = batch['history']
        neighbors = batch['neighbors']
        neighbor_mask = batch['neighbor_mask']
        future = batch['future_rel']
        map_patch = batch.get('map', None)

        map_features = None
        if self.use_map and map_patch is not None:
            map_features = self.map_encoder(map_patch)

        history_enc = self.encode_history(history)
        social_context = self.encode_social(history_enc, neighbors, neighbor_mask)

        if mode == 'train':
            future_enc = self.encode_future(future)
            rec_mu, rec_logvar = self.recognition_network(history_enc, social_context, future_enc, map_features)
        else:
            rec_mu, rec_logvar = None, None

        prior_mu, prior_logvar = self.prior_network(history_enc, social_context, map_features)

        if mode == 'train':
            K_wta = 3
            z_samples = []
            for _ in range(K_wta):
                z_k = self.reparameterize(rec_mu, rec_logvar)
                z_samples.append(z_k)
            z_samples = torch.stack(z_samples, dim=1)

            predictions = self.decode_wta(history, z_samples)
            predictions_winner = predictions
        else:
            z = self.reparameterize(prior_mu, prior_logvar)
            predictions = self.decode(history, z)
            predictions_winner = predictions

        result = {
            'predictions': predictions,
            'predictions_winner': predictions_winner,
            'z': z if mode == 'infer' else z_samples,
            'rec_mu': rec_mu,
            'rec_logvar': rec_logvar,
            'prior_mu': prior_mu,
            'prior_logvar': prior_logvar,
        }

        if mode == 'train' and future is not None:
            B, K, T, _ = predictions.shape

            future_exp = future.unsqueeze(1).expand(B, K, T, 2)
            loss_per_mode = F.smooth_l1_loss(predictions, future_exp, reduction='none').mean(dim=-1).mean(dim=-1)

            winner_idx = loss_per_mode.argmin(dim=-1)

            batch_idx = torch.arange(B, device=predictions.device)
            winner_trajs = predictions[batch_idx, winner_idx]

            loss_traj = F.smooth_l1_loss(winner_trajs, future)

            rec_logvar_clamped = rec_logvar.clamp(min=-10, max=10)
            prior_logvar_clamped = prior_logvar.clamp(min=-10, max=10)

            kl_rec_prior = -0.5 * torch.mean(
                1 + rec_logvar_clamped - prior_logvar_clamped -
                (rec_mu - prior_mu).pow(2) / prior_logvar_clamped.exp() -
                rec_logvar_clamped.exp() / prior_logvar_clamped.exp()
            )

            kl_loss = kl_rec_prior

            loss = loss_traj + 0.5 * kl_loss

            result['loss'] = loss
            result['loss_traj'] = loss_traj
            result['loss_kl'] = kl_loss
            result['winner_idx'] = winner_idx

        return result

    def inference(self, history: torch.Tensor, neighbors: torch.Tensor,
                  neighbor_mask: torch.Tensor,
                  num_candidates: int = 30,
                  map_patch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generate multiple trajectory samples using prior network.

        Args:
            history: (B, past_steps, 2)
            neighbors: (B, max_neighbors, past_steps, 2)
            neighbor_mask: (B, max_neighbors)
            num_candidates: number of trajectories to sample
            map_patch: (B, 3, 64, 64) optional semantic map patch

        Returns:
            (B, num_candidates, future_steps, 2) predicted trajectories
        """
        B = history.shape[0]

        map_features = None
        if self.use_map and map_patch is not None:
            map_features = self.map_encoder(map_patch)

        history_enc = self.encode_history(history)
        social_context = self.encode_social(history_enc, neighbors, neighbor_mask)
        prior_mu, prior_logvar = self.prior_network(history_enc, social_context, map_features)

        all_predictions = []
        for _ in range(num_candidates):
            z = self.reparameterize(prior_mu, prior_logvar)
            pred = self.decode(history, z)
            all_predictions.append(pred)

        return torch.stack(all_predictions, dim=1)
