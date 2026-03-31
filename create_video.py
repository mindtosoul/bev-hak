import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import Counter
from src.data.nuscenes_dataset import NuScenesTrajectoryDataset
from src.models.social_gru import SocialGRUCVAE


def create_multi_pedestrian_movie(num_pedestrians=3, output_filename="multi_pedestrian_tracking.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = NuScenesTrajectoryDataset(
        root='nuscenes',
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

    checkpoint = torch.load('checkpoints_gru/best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_instances = [ds[i]['instance_token'] for i in range(len(ds))]
    instance_counts = Counter(all_instances)

    pedestrians = []
    for token, count in instance_counts.most_common(50):
        sequence_indices = [i for i, t in enumerate(all_instances) if t == token]
        if len(sequence_indices) >= 20:
            pedestrians.append({
                'token': token,
                'indices': sequence_indices,
                'agent_type': ds[sequence_indices[0]]['agent_type']
            })
            if len(pedestrians) >= num_pedestrians:
                break

    if len(pedestrians) < num_pedestrians:
        print(f"Warning: Only found {len(pedestrians)} pedestrians with enough frames")
        num_pedestrians = len(pedestrians)

    print(f"Tracking {num_pedestrians} pedestrians:")
    for p in pedestrians:
        print(f"  - {p['token'][:8]}... ({p['agent_type']}) - {len(p['indices'])} frames")

    min_frames = min(len(p['indices']) for p in pedestrians)
    print(f"Each pedestrian has {min_frames} frames. Using {min_frames} frames for animation.")

    fig, axes = plt.subplots(1, num_pedestrians, figsize=(6 * num_pedestrians, 6))
    if num_pedestrians == 1:
        axes = [axes]

    def update(frame_idx):
        for ax_idx, ped in enumerate(pedestrians):
            ax = axes[ax_idx]
            ax.clear()

            if frame_idx >= len(ped['indices']):
                continue

            ds_idx = ped['indices'][frame_idx]
            data = ds[ds_idx]

            hist = data['history'].unsqueeze(0).to(device)
            neigh = data['neighbors'].unsqueeze(0).to(device)
            mask = data['neighbor_mask'].unsqueeze(0).to(device)
            map_patch = data['map'].unsqueeze(0).to(device)

            with torch.no_grad():
                preds = model.inference(hist, neigh, mask, num_candidates=5, map_patch=map_patch)
                preds = preds.squeeze(0).cpu().numpy()

            h = data['history'].cpu().numpy()
            anchor = h[-1]
            gt_rel = data['future_rel'].cpu().numpy()

            n_data = data['neighbors'].cpu().numpy()
            n_mask = data['neighbor_mask'].cpu().numpy()
            for n_idx in range(len(n_mask)):
                if n_mask[n_idx] > 0:
                    ax.plot(n_data[n_idx, :, 0], n_data[n_idx, :, 1], color='grey', alpha=0.3)

            ax.plot(h[:, 0], h[:, 1], 'ro-', label='Past', markersize=4)

            gt_path = np.vstack([anchor, anchor + gt_rel])
            ax.plot(gt_path[:, 0], gt_path[:, 1], 'go-', linewidth=3, label='Actual')

            for k in range(preds.shape[0]):
                p_path = np.vstack([anchor, anchor + preds[k]])
                ax.plot(p_path[:, 0], p_path[:, 1], 'b--', alpha=0.5, label='Pred' if k == 0 else "")

            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.set_xlim(anchor[0] - 10, anchor[0] + 10)
            ax.set_ylim(anchor[1] - 10, anchor[1] + 10)
            ax.set_title(f"{ped['agent_type'].capitalize()}\nFrame {frame_idx + 1}/{min_frames}")
            ax.legend(loc='upper left', fontsize=8)

        plt.suptitle(f"Multi-Pedestrian Tracking | Frame {frame_idx + 1}/{min_frames}", fontsize=14)
        plt.tight_layout()

    print(f"Creating animation with {min_frames} frames (looping)...")
    ani = FuncAnimation(fig, update, frames=min_frames, repeat=True)

    try:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=4, bitrate=1800)
        ani.save(output_filename, writer=writer)
        print(f"Video saved as {output_filename}")
    except Exception as e:
        print(f"FFMpeg not available ({e}), trying Pillow writer for GIF...")
        try:
            ani.save(output_filename.replace('.mp4', '.gif'), writer='pillow', fps=4)
            print(f"GIF saved as {output_filename.replace('.mp4', '.gif')}")
        except Exception as e2:
            print(f"Pillow also failed: {e2}")
            print("Showing animation instead (will loop)...")
            plt.show()


if __name__ == "__main__":
    create_multi_pedestrian_movie(num_pedestrians=3)