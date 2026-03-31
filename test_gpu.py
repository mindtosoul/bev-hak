import torch
from src.models.social_gru import SocialGRUCVAE

device = torch.device('cuda')
print(f"CUDA available: {torch.cuda.is_available()}")

model = SocialGRUCVAE(use_map=True).to(device)
print(f"Model device: {next(model.parameters()).device}")

batch = {
    'history': torch.randn(4, 4, 2).to(device),
    'neighbors': torch.randn(4, 10, 4, 2).to(device),
    'neighbor_mask': torch.ones(4, 10, dtype=torch.bool).to(device),
    'future_rel': torch.randn(4, 6, 2).to(device),
    'map': torch.randn(4, 3, 64, 64).to(device),
}

out = model(batch, mode='train')
print(f"Output device: {out['predictions'].device}")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
print("GPU is working!")
