"""Evaluation script for trajectory prediction models."""
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
from src.utils.metrics import batched_metrics, TrajectoryMetrics
from src.utils.sampling import cluster_diverse_trajectories, sample_cvae
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


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device,
           config: Config, num_candidates: int = 30, top_k: int = 3,
           use_clustering: bool = False) -> dict:
    """Evaluate model with multi-modal predictions.

    Args:
        model: Trained model
        dataloader: Validation dataloader
        device: compute device
        config: configuration
        num_candidates: number of trajectories to sample
        top_k: top-k trajectories to select
        use_clustering: whether to use clustering for selection

    Returns:
        dict of evaluation metrics
    """
    model.eval()
    metrics_tracker = TrajectoryMetrics(k_modes=top_k)

    all_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            predictions = model.inference(
                batch['history'],
                batch['neighbors'],
                batch['neighbor_mask'],
                num_candidates=num_candidates,
                map_patch=batch.get('map')
            )

            targets = batch['future_rel']

            if use_clustering:
                B, N, T, _ = predictions.shape
                if N >= top_k:
                    predictions_selected, _ = cluster_diverse_trajectories(predictions, k=top_k)
                else:
                    predictions_selected = predictions[:, :N]
            else:
                B, N, T, _ = predictions.shape
                predictions_selected = predictions[:, :min(top_k, N)]

            metrics_tracker.update(predictions_selected, targets)

            if batch_idx < 10:
                for b in range(min(3, predictions_selected.shape[0])):
                    result = {
                        'batch_idx': batch_idx,
                        'sample_idx': b,
                        'predictions': predictions_selected[b].cpu().numpy(),
                        'target': targets[b].cpu().numpy(),
                    }
                    all_results.append(result)

    final_metrics = metrics_tracker.compute()

    print("\n=== Evaluation Results ===")
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}")

    return final_metrics, all_results


def main(config_path: str, checkpoint_path: str, output_dir: str = "results"):
    """Main evaluation function."""
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
        cache_path=os.path.join(output_dir, 'eval_cache.json'),
    )

    print(f"Evaluation samples: {len(dataset)}")

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

    os.makedirs(output_dir, exist_ok=True)

    metrics, results = evaluate(
        model, dataloader, device, config,
        num_candidates=config.inference.num_candidates,
        top_k=config.inference.top_k,
        use_clustering=False
    )

    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'num_samples': len(dataset),
        }, f, indent=2)
    print(f"Saved metrics to {results_path}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trajectory prediction model')
    parser.add_argument('--config', type=str, default='configs/baseline_light.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    args = parser.parse_args()

    main(args.config, args.checkpoint, args.output)
