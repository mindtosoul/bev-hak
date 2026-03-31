"""Inference script for trajectory prediction with multi-modal output."""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, Config
from src.utils.sampling import cluster_diverse_trajectories
from src.data.nuscenes_dataset import NuScenesTrajectoryDataset, nuscenes_collate_fn
from src.models.social_gru import SocialGRUCVAE


def build_model(config: Config, device: torch.device):
    """Build model based on configuration."""
    model = SocialGRUCVAE(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        latent_dim=config.model.latent_dim,
        output_dim=config.model.output_dim,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        dropout=config.model.dropout,
        social_num_heads=4,
        max_neighbors=config.data.max_neighbors,
        future_steps=config.data.future_steps,
        use_map=getattr(config.model, 'use_map', True),
    )
    return model.to(device)


def local_to_world_batch(local_traj: torch.Tensor, origin: torch.Tensor,
                         angle: torch.Tensor) -> torch.Tensor:
    """Convert local trajectories to world coordinates (GPU-optimized).

    Args:
        local_traj: (B, K, T, 2) or (B, T, 2) trajectories in local coords
        origin: (B, 2) origin positions in world coords
        angle: (B,) heading angles

    Returns:
        Trajectories in world coordinates
    """
    if local_traj.dim() == 3:
        local_traj = local_traj.unsqueeze(1)

    B, K, T, _ = local_traj.shape
    angle_expanded = angle.unsqueeze(-1).unsqueeze(-1)
    cos_a = torch.cos(angle_expanded)
    sin_a = torch.sin(angle_expanded)

    R = torch.stack([
        torch.stack([cos_a, -sin_a], dim=-1),
        torch.stack([sin_a, cos_a], dim=-1)
    ], dim=-3)

    local_flat = local_traj.view(B * K, T, 2)
    R_expanded = R.unsqueeze(1).expand(B, K, 2, 2).reshape(B * K, 2, 2)
    origin_expanded = origin.unsqueeze(1).expand(B, K, 2).reshape(B * K, 2)

    world_flat = local_flat @ R_expanded.transpose(-2, -1) + origin_expanded
    world_traj = world_flat.view(B, K, T, 2)

    return world_traj


def infer(model: nn.Module, dataloader: DataLoader, device: torch.device,
          config: Config, num_candidates: int = 30, top_k: int = 3,
          save_endpoints_only: bool = False) -> list:
    """Run inference with multi-modal trajectory generation.

    Args:
        model: Trained model
        dataloader: Data loader
        device: compute device
        config: configuration
        num_candidates: number of trajectories to sample
        top_k: top-k trajectories to output
        save_endpoints_only: if True, only save final positions (for smaller JSON)

    Returns:
        list of predictions with world coordinates
    """
    model.eval()

    predictions_output = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inferencing"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            predictions_local = model.inference(
                batch['history'],
                batch['neighbors'],
                batch['neighbor_mask'],
                num_candidates=num_candidates,
                map_patch=batch.get('map')
            )

            predictions_local_selected, _ = cluster_diverse_trajectories(
                predictions_local, k=top_k
            )

            predictions_world = local_to_world_batch(
                predictions_local_selected,
                batch['origin'],
                batch['angle']
            )

            B, K, T, _ = predictions_world.shape

            for b in range(B):
                if save_endpoints_only:
                    pred_data = predictions_world[b, :, -1, :].cpu().numpy().tolist()
                else:
                    pred_data = predictions_world[b].cpu().numpy().tolist()

                prediction_entry = {
                    'instance_token': batch['instance_tokens'][b],
                    'agent_type': batch['agent_types'][b],
                    'location': batch.get('location', 'unknown'),
                    'predictions_top{}'.format(K): pred_data,
                    'origin': batch['origin'][b].cpu().numpy().tolist(),
                    'angle': batch['angle'][b].item(),
                }
                predictions_output.append(prediction_entry)

    return predictions_output


def main(config_path: str, checkpoint_path: str, output_path: str = "predictions.json",
         num_candidates: int = 30, top_k: int = 3, save_endpoints_only: bool = False):
    """Main inference function."""
    config = load_config(config_path)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    dataset = NuScenesTrajectoryDataset(
        root=config.data.nuscenes_root,
        version=config.data.version,
        past_steps=config.data.past_steps,
        future_steps=config.data.future_steps,
        neighbor_radius=config.data.neighbor_radius,
        max_neighbors=config.data.max_neighbors,
        agents_of_interest=config.data.agents_of_interest,
        split='val',
        cache_path=os.path.join('checkpoints', 'val_cache.json'),
    )

    print(f"Inference samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=nuscenes_collate_fn,
    )

    model = build_model(config, device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    predictions = infer(model, dataloader, device, config,
                       num_candidates=num_candidates, top_k=top_k,
                       save_endpoints_only=save_endpoints_only)

    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved {len(predictions)} predictions to {output_path}")

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on trajectory prediction model')
    parser.add_argument('--config', type=str, default='configs/baseline_light.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output path for predictions')
    parser.add_argument('--num-candidates', type=int, default=30,
                       help='Number of trajectory candidates to generate')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top trajectories to output')
    parser.add_argument('--endpoints-only', action='store_true',
                       help='Save only final positions (smaller file size)')
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.output, args.num_candidates, args.top_k,
         args.endpoints_only)
