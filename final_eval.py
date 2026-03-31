"""Final evaluation script on v1.0-mini val set."""
import torch
from src.data.nuscenes_dataset import NuScenesTrajectoryDataset, nuscenes_collate_fn
from src.models.social_gru import SocialGRUCVAE
from src.utils.metrics import TrajectoryMetrics
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

model = SocialGRUCVAE(
    input_dim=2, hidden_dim=128, latent_dim=16, output_dim=2,
    num_encoder_layers=2, num_decoder_layers=1, dropout=0.1, social_num_heads=4,
    max_neighbors=10, future_steps=6, use_map=True
).to(device)
model.load_state_dict(torch.load('checkpoints_gru/best_model.pt', map_location=device, weights_only=False)['model_state_dict'])
model.eval()
print('Model loaded!')

ds = NuScenesTrajectoryDataset(
    root='nuscenes',
    version='v1.0-mini',
    past_steps=4, future_steps=6, neighbor_radius=10.0, max_neighbors=10,
    split='val'
)
print(f'Val samples: {len(ds)}')

loader = DataLoader(ds, batch_size=48, shuffle=False, num_workers=0, collate_fn=nuscenes_collate_fn)
metrics_tracker = TrajectoryMetrics(k_modes=3)

with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model(batch, mode='train')
        preds = output['predictions']
        targets = batch['future_rel']
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)
        if targets.dim() == 2:
            targets = targets.unsqueeze(1)
        metrics_tracker.update(preds[:, :3], targets)
        if batch_idx % 20 == 0:
            print(f'Batch {batch_idx}/{len(loader)}')

metrics = metrics_tracker.compute()
print('\n=== FINAL EVALUATION RESULTS ===')
print(f'minADE@3:  {metrics["minADE@3"]:.4f}')
print(f'minFDE@3:  {metrics["minFDE@3"]:.4f}')
print(f'Miss Rate@2m: {metrics["miss_rate@2m"]:.4f}')
