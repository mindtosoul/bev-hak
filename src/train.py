"""Training script for trajectory prediction models."""
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, Config
from src.utils.metrics import batched_metrics, TrajectoryMetrics
from src.utils.sampling import cluster_diverse_trajectories, best_of_many
from src.data.nuscenes_dataset import NuScenesTrajectoryDataset, nuscenes_collate_fn
from src.models.social_gru import SocialGRUCVAE


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    model = model.to(device)

    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")

    return model


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
               device: torch.device, config: Config, epoch: int,
               scaler: torch.cuda.amp.GradScaler = None) -> dict:
    """Train for one epoch with optional AMP."""
    model.train()
    total_loss = 0
    total_traj_loss = 0
    total_kl_loss = 0
    num_batches = 0

    kl_beta = min(1.0, epoch / config.training.kl_warmup_epochs) * config.training.kl_beta

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(batch, mode='train')
                loss = output['loss_traj'] + kl_beta * output.get('loss_kl', torch.tensor(0.0, device=device))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(batch, mode='train')
            loss = output['loss_traj'] + kl_beta * output.get('loss_kl', torch.tensor(0.0, device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        traj_loss = output.get('loss_traj', loss)
        kl_loss = output.get('loss_kl', torch.tensor(0.0))

        total_loss += loss.item()
        total_traj_loss += traj_loss.item()
        total_kl_loss += kl_loss.item()
        num_batches += 1

        if batch_idx % config.training.log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'traj': f'{traj_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'beta': f'{kl_beta:.2f}',
            })

    return {
        'loss': total_loss / num_batches,
        'loss_traj': total_traj_loss / num_batches,
        'loss_kl': total_kl_loss / num_batches,
    }


def validate(model: nn.Module, dataloader: DataLoader, device: torch.device,
            config: Config, k_modes: int = 3) -> dict:
    """Validate model."""
    model.eval()
    metrics_tracker = TrajectoryMetrics(k_modes=k_modes)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            output = model(batch, mode='train')

            predictions = output['predictions']
            targets = batch['future_rel']

            if predictions.dim() == 3:
                predictions = predictions.unsqueeze(1)

            if targets.dim() == 2:
                targets = targets.unsqueeze(1)

            actual_k = min(predictions.shape[1], k_modes)
            metrics_tracker.update(predictions[:, :actual_k], targets)

    metrics = metrics_tracker.compute()
    return metrics


def train(config_path: str):
    """Main training function."""
    config = load_config(config_path)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Model type: {config.model.model_type}")
    print(f"Config: {config}")

    set_seed(config.seed)

    os.makedirs(config.training.save_dir, exist_ok=True)

    dataset = NuScenesTrajectoryDataset(
        root=config.data.nuscenes_root,
        version=config.data.version,
        past_steps=config.data.past_steps,
        future_steps=config.data.future_steps,
        neighbor_radius=config.data.neighbor_radius,
        max_neighbors=config.data.max_neighbors,
        agents_of_interest=config.data.agents_of_interest,
        split='train',
        cache_path=os.path.join(config.training.save_dir, 'train_cache.json'),
    )

    val_dataset = NuScenesTrajectoryDataset(
        root=config.data.nuscenes_root,
        version=config.data.version,
        past_steps=config.data.past_steps,
        future_steps=config.data.future_steps,
        neighbor_radius=config.data.neighbor_radius,
        max_neighbors=config.data.max_neighbors,
        agents_of_interest=config.data.agents_of_interest,
        split='val',
        cache_path=os.path.join(config.training.save_dir, 'val_cache.json'),
    )

    train_subset_size = int(len(dataset) * config.data.train_fraction)
    if train_subset_size < len(dataset):
        indices = list(range(train_subset_size))
        dataset = Subset(dataset, indices)

    print(f"Training samples: {len(dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    persistent_workers = config.training.num_workers > 0
    train_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if config.training.num_workers > 0 else None,
        collate_fn=nuscenes_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if config.training.num_workers > 0 else None,
        collate_fn=nuscenes_collate_fn,
    )

    model = build_model(config, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs,
        eta_min=config.training.learning_rate * 0.01,
    )

    scaler = torch.cuda.amp.GradScaler() if config.training.use_amp else None

    best_metric = float('inf')
    patience_counter = 0

    for epoch in range(1, config.training.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, config, epoch, scaler)

        print(f"Epoch {epoch} Train - Loss: {train_metrics['loss']:.4f}, "
              f"Traj: {train_metrics['loss_traj']:.4f}, KL: {train_metrics['loss_kl']:.4f}")

        if epoch % 10 == 0 or epoch == config.training.epochs:
            val_metrics = validate(model, val_loader, device, config)
            print(f"Epoch {epoch} Val - minADE@3: {val_metrics['minADE@3']:.4f}, "
                  f"minFDE@3: {val_metrics['minFDE@3']:.4f}")

            if val_metrics['minADE@3'] < best_metric:
                best_metric = val_metrics['minADE@3']
                patience_counter = 0
                ckpt_path = os.path.join(config.training.save_dir, 'best_model.pt')

                save_state = model.state_dict()
                if isinstance(model, nn.DataParallel):
                    save_state = model.module.state_dict()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'config': config,
                }, ckpt_path)
                print(f"Saved best model to {ckpt_path}")
            else:
                patience_counter += 1

            if patience_counter >= config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        scheduler.step()

    print("Training complete!")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train trajectory prediction model')
    parser.add_argument('--config', type=str, default='configs/baseline_light.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    train(args.config)
