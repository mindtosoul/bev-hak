import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift
from scipy.ndimage import rotate
from src.data.nuscenes_dataset import NuScenesTrajectoryDataset
from src.models.social_gru import SocialGRUCVAE


def apply_2d_rotation(coords, angle_deg):
    """Apply 2D rotation matrix to coordinates (N, 2) or (2,)"""
    angle_rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a],
                                [sin_a, cos_a]])
    if coords.ndim == 1:
        return np.dot(rotation_matrix, coords)
    return np.dot(coords, rotation_matrix.T)


def rotate_map_patch(map_patch, angle_deg):
    """Rotate a 3-channel map patch image to match trajectory rotation.

    The map is centered at agent position (32,32 for 64x64). We rotate using
    the SAME angle and direction as the trajectory rotation to keep them aligned.

    Args:
        map_patch: (3, H, W) tensor or numpy array
        angle_deg: rotation angle in degrees (same as applied to trajectory)

    Returns:
        (3, H, W) rotated map patch
    """
    map_np = map_patch.cpu().numpy() if isinstance(map_patch, torch.Tensor) else map_patch

    if map_np.shape[0] == 3:
        map_np = np.transpose(map_np, (1, 2, 0))

    rotated = rotate(map_np, angle=angle_deg, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0)

    rotated = np.transpose(rotated, (2, 0, 1))

    return torch.from_numpy(rotated).float()


def shift_map_patch(map_patch, shift_px):
    """Shift a 3-channel map patch image by specified pixels per channel."""
    map_np = map_patch.cpu().numpy() if isinstance(map_patch, torch.Tensor) else map_patch
    if map_np.shape[0] == 3:
        map_np = np.transpose(map_np, (1, 2, 0))
    shifted = shift(map_np, shift=(shift_px[0], shift_px[1], 0), order=1, mode='constant', cval=0)
    shifted = np.transpose(shifted, (2, 0, 1))
    return torch.from_numpy(shifted).float()


def stress_test():
    """Generalization Stress Test: Translate trajectories by 5cm to test spatial invariance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = NuScenesTrajectoryDataset(
        root='c:/Users/Aditya/Desktop/bev_hak/nuscenes',
        version='v1.0-mini',
        split='val',
        past_steps=4,
        future_steps=6,
        neighbor_radius=10.0,
        max_neighbors=10,
    )

    model = SocialGRUCVAE(
        input_dim=2,
        hidden_dim=128,
        latent_dim=16,
        output_dim=2,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.1,
        social_num_heads=4,
        max_neighbors=10,
        future_steps=6,
        use_map=True,
    ).to(device)

    model.load_state_dict(torch.load('checkpoints_gru/best_model.pt', map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    sample_idx = 61
    translation = np.array([0.20, 0.20])

    print(f"\n=== Generalization Stress Test ===")
    print(f"Sample Index: {sample_idx}")
    print(f"Translation: {translation} (20cm in x and y)")

    data = ds[sample_idx]

    hist = data['history'].cpu().numpy()
    neigh = data['neighbors'].cpu().numpy()
    mask = data['neighbor_mask'].cpu().numpy()
    future_rel = data['future_rel'].cpu().numpy()
    map_patch = data['map'].unsqueeze(0).to(device)

    pixels_per_meter = 64 / 20.0
    shift_px = translation * pixels_per_meter
    print(f"Map shift in pixels: {shift_px}")

    shifted_hist = hist + translation
    shifted_future_rel = future_rel.copy()
    shifted_neigh = neigh.copy()
    for i in range(neigh.shape[0]):
        if mask[i] > 0:
            shifted_neigh[i] = neigh[i] + translation

    shifted_map = shift_map_patch(map_patch.squeeze(0), shift_px).unsqueeze(0).to(device)

    hist_tensor = torch.from_numpy(shifted_hist).unsqueeze(0).float().to(device)
    neigh_tensor = torch.from_numpy(shifted_neigh).unsqueeze(0).float().to(device)
    mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0).to(device)

    with torch.no_grad():
        preds_shifted = model.inference(hist_tensor, neigh_tensor, mask_tensor, num_candidates=10, map_patch=shifted_map)
        preds_shifted = preds_shifted.squeeze(0).cpu().numpy()

        orig_hist_tensor = torch.from_numpy(hist).unsqueeze(0).float().to(device)
        orig_neigh_tensor = torch.from_numpy(neigh).unsqueeze(0).float().to(device)
        orig_mask_tensor = torch.from_numpy(mask).bool().unsqueeze(0).to(device)
        preds_original = model.inference(orig_hist_tensor, orig_neigh_tensor, orig_mask_tensor, num_candidates=10, map_patch=map_patch)
        preds_original = preds_original.squeeze(0).cpu().numpy()

    print("\n=== DIAGNOSTIC: Prediction Magnitudes ===")
    print(f"Original predictions - shape: {preds_original.shape}")
    print(f"  Mean per step: {np.mean(np.abs(preds_original), axis=(0,2))}")
    print(f"  Total trajectory length: {np.mean(np.linalg.norm(preds_original, axis=2), axis=0)}")

    print(f"\nShifted predictions - shape: {preds_shifted.shape}")
    print(f"  Mean per step: {np.mean(np.abs(preds_shifted), axis=(0,2))}")
    print(f"  Total trajectory length: {np.mean(np.linalg.norm(preds_shifted, axis=2), axis=0)}")

    print(f"\nOriginal ground truth total length: {np.linalg.norm(future_rel[-1]) if future_rel.ndim > 1 else np.linalg.norm(future_rel)}")
    print(f"Shifted ground truth same as original: {np.linalg.norm(future_rel[-1]) if future_rel.ndim > 1 else np.linalg.norm(future_rel)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    anchor_orig = hist[-1]
    anchor_shifted = shifted_hist[-1]

    for plot_idx, (ax, title, h, anchor, gt_rel, preds, neigh_data) in enumerate([
        (axes[0], f"Original Trajectory", hist, anchor_orig, future_rel, preds_original, neigh),
        (axes[1], f"Shifted (+5cm) - Stress Test", shifted_hist, anchor_shifted, shifted_future_rel, preds_shifted, shifted_neigh)
    ]):
        ax.clear()

        for n_idx in range(neigh_data.shape[0]):
            if mask[n_idx] > 0:
                ax.plot(neigh_data[n_idx, :, 0], neigh_data[n_idx, :, 1], 'o-',
                        color='grey', markersize=2, alpha=0.3,
                        label='Neighbors' if plot_idx == 0 and n_idx == 0 else "")

        ax.plot(h[:, 0], h[:, 1], 'ro-', label='History (Past)', markersize=4, alpha=0.6)

        gt_abs = np.vstack([anchor, anchor + gt_rel])
        ax.plot(gt_abs[:, 0], gt_abs[:, 1], 'go-', linewidth=3, label='Ground Truth')

        for k in range(preds.shape[0]):
            p_rel = preds[k]
            p_abs = np.vstack([anchor, anchor + p_rel])
            ax.plot(p_abs[:, 0], p_abs[:, 1], 'b--', alpha=0.4,
                    label='CVAE Predictions' if plot_idx == 0 and k == 0 else "")

        ax.set_title(title, fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xlim(anchor[0] - 10, anchor[0] + 10)
        ax.set_ylim(anchor[1] - 10, anchor[1] + 10)
        if plot_idx == 0:
            ax.legend(loc='upper left', fontsize=8)

    plt.suptitle("Generalization Stress Test: Translation Invariance Check", fontsize=16)
    plt.tight_layout()
    plt.show()

    print(f"\n=== Analysis ===")
    print(f"If blue predictions match green ground truth for both original and shifted:")
    print(f"  -> Model is TRANSLATION INVARIANT (good generalization)")
    print(f"\nIf predictions change significantly after 5cm shift:")
    print(f"  -> Model may have position leakage")


def visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = NuScenesTrajectoryDataset(
        root='c:/Users/Aditya/Desktop/bev_hak/nuscenes',
        version='v1.0-mini',
        split='val',
        past_steps=4,
        future_steps=6,
        neighbor_radius=10.0,
        max_neighbors=10,
    )

    model = SocialGRUCVAE(
        input_dim=2,
        hidden_dim=128,
        latent_dim=16,
        output_dim=2,
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.1,
        social_num_heads=4,
        max_neighbors=10,
        future_steps=6,
        use_map=True,
    ).to(device)

    model.load_state_dict(torch.load('checkpoints_gru/best_model.pt', map_location=device, weights_only=False)['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    indices = [1094, 1100, 61]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            data = ds[idx]
            hist = data['history'].unsqueeze(0).to(device)
            neigh = data['neighbors'].unsqueeze(0).to(device)
            mask = data['neighbor_mask'].unsqueeze(0).to(device)
            map_patch = data['map'].unsqueeze(0).to(device)

            preds = model.inference(hist, neigh, mask, num_candidates=10, map_patch=map_patch)
            preds = preds.squeeze(0).cpu().numpy()

            ax = axes[i]

            n_data = neigh.squeeze(0).cpu().numpy()
            n_mask = mask.squeeze(0).cpu().numpy()

            for n_idx in range(n_data.shape[0]):
                if n_mask[n_idx] > 0:
                    n_path = n_data[n_idx]
                    ax.plot(n_path[:, 0], n_path[:, 1], 'o-', color='grey',
                            markersize=2, alpha=0.3, label='Neighbors' if i == 0 and n_idx == 0 else "")

            h = data['history'].cpu().numpy()
            ax.plot(h[:, 0], h[:, 1], 'ro-', label='History (Past)', markersize=4, alpha=0.6)

            anchor = h[-1]
            gt_rel = data['future_rel'].cpu().numpy()
            gt_abs_path = np.vstack([anchor, anchor + gt_rel])
            ax.plot(gt_abs_path[:, 0], gt_abs_path[:, 1], 'go-', linewidth=3, label='Ground Truth')

            for k in range(preds.shape[0]):
                p_rel = preds[k]
                p_abs_path = np.vstack([anchor, anchor + p_rel])
                ax.plot(p_abs_path[:, 0], p_abs_path[:, 1], 'b--', alpha=0.4,
                        label='CVAE Predictions' if k == 0 else "")

            ax.set_title(f"Scenario {idx}", fontsize=12)
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
            if i == 0:
                ax.legend()

    plt.suptitle("Trajectory Prediction: Multi-modal Intent (CVAE) - Social GRU", fontsize=16)
    plt.tight_layout()
    plt.savefig('visualization_3_cases.png', dpi=150, bbox_inches='tight')
    print("Saved to visualization_3_cases.png")
    plt.close()


if __name__ == "__main__":
    visualize()