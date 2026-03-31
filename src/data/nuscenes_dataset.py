"""nuScenes dataset loader for trajectory prediction.

Extracts pedestrian and cyclist trajectories with social context and map information.
Uses the official nuscenes-devkit API and NuScenes Map API.
"""
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap

from ..utils.geometry import world_to_local


PEDESTRIAN_CATEGORIES = ['pedestrian.adult', 'pedestrian.child', 'pedestrian.construction_worker']
BICYCLE_CATEGORIES = ['bicycle.rider', 'cyclist.rider']
ALL_AGENTS = PEDESTRIAN_CATEGORIES + BICYCLE_CATEGORIES

NUSCENES_FREQ = 2
DT = 1.0 / NUSCENES_FREQ

MAP_LAYERS = ['road_segment', 'ped_crossing', 'walkway', 'bus_stop', '停']
MAP_PATCH_SIZE = 20.0


class NuScenesTrajectoryDataset(Dataset):
    """Dataset for loading trajectory sequences from nuScenes.

    Extracts agent-centric trajectory sequences with:
    - Past: 2 seconds (4 frames at 2Hz)
    - Future: 3 seconds (6 frames at 2Hz)
    - Social context: neighbors within radius
    """

    def __init__(self,
                 root: str,
                 version: str = "v1.0-mini",
                 past_steps: int = 4,
                 future_steps: int = 6,
                 neighbor_radius: float = 10.0,
                 max_neighbors: int = 15,
                 agents_of_interest: List[str] = None,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 cache_path: Optional[str] = None):
        """Initialize dataset.

        Args:
            root: Path to nuScenes dataset root
            version: Dataset version (v1.0-mini or v1.0)
            past_steps: Number of past timesteps (4 for 2s at 2Hz)
            future_steps: Number of future timesteps (6 for 3s at 2Hz)
            neighbor_radius: Radius to search for neighbors (meters)
            max_neighbors: Maximum number of neighbors to store
            agents_of_interest: List of category names to filter
            split: 'train' or 'val'
            transform: Optional transform to apply
            cache_path: Optional path to precomputed sample cache
        """
        self.root = Path(root)
        self.version = version
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.total_steps = past_steps + future_steps
        self.neighbor_radius = neighbor_radius
        self.max_neighbors = max_neighbors
        self.agents_of_interest = agents_of_interest or ALL_AGENTS
        self.split = split
        self.transform = transform

        data_path = self.root / version
        if not data_path.exists():
            raise FileNotFoundError(f"nuScenes data not found at {data_path}")

        self.nusc = NuScenes(version=version, dataroot=str(self.root), verbose=False)

        self.maps = {}
        self._load_maps()

        self._build_scene_index()
        self._build_valid_samples(cache_path)

        print(f"Loaded {len(self.valid_samples)} valid samples for {split}")

    def _build_scene_index(self):
        """Build index mapping scenes to samples and annotations."""
        self.scene_to_samples = {}
        self.sample_to_anns = {}

        for sample in self.nusc.sample:
            scene_token = sample['scene_token']
            if scene_token not in self.scene_to_samples:
                self.scene_to_samples[scene_token] = []
            self.scene_to_samples[scene_token].append(sample['token'])
            self.sample_to_anns[sample['token']] = sample['anns']

        for scene_tokens in self.scene_to_samples.values():
            scene_tokens.sort(key=lambda t: self.nusc.get('sample', t)['timestamp'])

    def _load_maps(self):
        """Load NuScenes maps for all locations."""
        locations = ['singapore-onenorth', 'boston-seaport', 'singapore-hollandvillage', 'singapore-queenstown']
        for loc in locations:
            try:
                self.maps[loc] = NuScenesMap(dataroot=str(self.root), map_name=loc)
                print(f"Loaded map for {loc}")
            except Exception as e:
                print(f"Could not load map for {loc}: {e}")

    def _get_map_patch(self, pose: np.ndarray, heading: float, location: str) -> torch.Tensor:
        """Get a map patch centered on the agent, rotated to match heading.

        Args:
            pose: (2,) world position [x, y]
            heading: heading angle in radians
            location: scene location name

        Returns:
            (3, H, W) tensor with road, crossing, walkway layers
        """
        if location not in self.maps:
            return torch.zeros(3, 64, 64)

        nusc_map = self.maps[location]

        patch_box = [pose[0], pose[1], MAP_PATCH_SIZE, MAP_PATCH_SIZE]
        patch_angle = np.degrees(heading)

        try:
            layer_names = ['road_segment', 'ped_crossing', 'walkway']
            map_mask = nusc_map.get_map_mask(
                patch_box=patch_box,
                patch_angle=patch_angle,
                layer_names=layer_names,
                canvas_size=(64, 64)
            )

            if map_mask is None or len(map_mask) == 0:
                return torch.zeros(3, 64, 64)

            if isinstance(map_mask, tuple):
                map_mask = map_mask[0]

            if map_mask.ndim == 3:
                map_tensor = torch.from_numpy(map_mask).float()
            else:
                map_tensor = torch.zeros(3, 64, 64)

            if map_tensor.shape[1:] != (64, 64):
                map_tensor = torch.nn.functional.interpolate(
                    map_tensor.unsqueeze(0),
                    size=(64, 64),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            return map_tensor

        except Exception as e:
            return torch.zeros(3, 64, 64)

    def _build_valid_samples(self, cache_path: Optional[str]):
        """Build list of valid sample entries with full history/future."""
        if cache_path and os.path.exists(cache_path):
            self._load_cache(cache_path)
            return

        self.valid_samples = []
        split_ratio = 0.8 if self.split == 'train' else 0.2

        for scene_token, sample_tokens in self.scene_to_samples.items():
            n_samples = len(sample_tokens)
            split_idx = int(n_samples * split_ratio)

            if n_samples < self.past_steps + self.future_steps + 1:
                continue

            if self.split == 'train':
                indices = range(self.past_steps, split_idx - self.future_steps)
            else:
                indices = range(split_idx, n_samples - self.future_steps)

            for sample_idx in indices:
                if sample_idx < self.past_steps or sample_idx + self.future_steps >= n_samples:
                    continue

                sample_token = sample_tokens[sample_idx]
                ann_tokens = self.sample_to_anns[sample_token]

                for ann_token in ann_tokens:
                    ann = self.nusc.get('sample_annotation', ann_token)
                    category_name = ann['category_name']

                    if not any(cat in category_name.lower() for cat in self.agents_of_interest):
                        continue

                    entry = {
                        'sample_token': sample_token,
                        'ann_token': ann_token,
                        'scene_token': scene_token,
                        'category_name': category_name,
                        'past_sample_tokens': sample_tokens[sample_idx - self.past_steps:sample_idx],
                        'future_sample_tokens': sample_tokens[sample_idx + 1:sample_idx + self.future_steps + 1],
                    }
                    self.valid_samples.append(entry)

        if cache_path:
            self._save_cache(cache_path)

    def _load_cache(self, cache_path: str):
        """Load cached valid samples."""
        import json
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        self.valid_samples = cache_data['valid_samples']
        print(f"Loaded {len(self.valid_samples)} cached valid samples")

    def _save_cache(self, cache_path: str):
        """Save valid samples to cache."""
        import json
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({'valid_samples': self.valid_samples}, f)
        print(f"Cached {len(self.valid_samples)} samples to {cache_path}")

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single trajectory sample.

        Returns:
            dict with:
            - history: (past_steps, 2) past positions in agent-centric coords
            - future: (future_steps, 2) future positions in world coords
            - future_rel: (future_steps, 2) future positions in agent-centric coords
            - origin: (2,) reference agent position at t0
            - angle: (1,) reference agent heading angle
            - neighbors: (max_neighbors, past_steps, 2) neighbor trajectories
            - neighbor_mask: (max_neighbors,) True for valid neighbors
            - agent_type: 'pedestrian', 'cyclist', or 'motorcycle'
            - map: (3, 64, 64) semantic map patch [road, crossing, walkway]
        """
        entry = self.valid_samples[idx]

        sample_token = entry['sample_token']
        ann_token = entry['ann_token']
        category_name = entry['category_name']

        ann = self.nusc.get('sample_annotation', ann_token)
        instance_token = ann['instance_token']

        ref_pos = np.array(ann['translation'][:2])

        heading_angle = 0.0
        try:
            q = Quaternion(ann['rotation'])
            heading_angle = q.yaw_pitch_roll[0]
        except:
            pass

        scene = self.nusc.get('scene', entry['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        scene_location = log['location']

        past_positions_world = self._get_instance_positions(
            instance_token, entry['past_sample_tokens'], ref_pos, self.past_steps
        )
        future_positions_world = self._get_instance_positions(
            instance_token, entry['future_sample_tokens'], ref_pos, self.future_steps
        )

        past_positions_local = world_to_local(past_positions_world, ref_pos, heading_angle)
        future_positions_local = world_to_local(future_positions_world, ref_pos, heading_angle)

        past_displacements = np.diff(past_positions_local, axis=0, prepend=past_positions_local[:1])[:, :2]

        neighbors, neighbor_mask = self._extract_neighbors(
            entry, instance_token, ref_pos, heading_angle
        )

        map_patch = self._get_map_patch(ref_pos, heading_angle, scene_location)

        is_cyclist = 'bicycle' in category_name.lower() or 'cyclist' in category_name.lower()
        is_motorcycle = 'motorcycle' in category_name.lower()

        if is_cyclist:
            agent_type = 'cyclist'
        elif is_motorcycle:
            agent_type = 'motorcycle'
        else:
            agent_type = 'pedestrian'

        item = {
            'history': torch.FloatTensor(past_displacements),
            'history_abs': torch.FloatTensor(past_positions_local),
            'future': torch.FloatTensor(future_positions_world),
            'future_rel': torch.FloatTensor(future_positions_local),
            'agent_type': agent_type,
            'origin': torch.FloatTensor(ref_pos),
            'angle': torch.FloatTensor([heading_angle]),
            'neighbors': neighbors,
            'neighbor_mask': neighbor_mask,
            'instance_token': instance_token,
            'category_name': category_name,
            'location': scene_location,
            'map': map_patch,
        }

        if self.transform:
            item = self.transform(item)

        return item

    def _get_instance_positions(self, instance_token: str, sample_tokens: List[str],
                                  ref_pos: np.ndarray, expected_steps: int) -> np.ndarray:
        """Get positions for an instance across sample tokens.

        Returns (expected_steps, 2) array of world positions. Missing frames use ref_pos.
        """
        positions = []
        for sample_token in sample_tokens:
            ann_tokens = self.sample_to_anns[sample_token]
            pos = ref_pos.copy()
            found = False
            for at in ann_tokens:
                a = self.nusc.get('sample_annotation', at)
                if a['instance_token'] == instance_token:
                    pos = np.array(a['translation'][:2])
                    found = True
                    break
            positions.append(pos)

        positions = np.stack(positions, axis=0)

        if positions.shape[0] < expected_steps:
            padding = np.tile(ref_pos.reshape(1, 2), (expected_steps - positions.shape[0], 1))
            positions = np.vstack([positions, padding])

        return positions

    def _extract_neighbors(self, entry: Dict, ego_instance: str, ref_pos: np.ndarray,
                           heading_angle: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract neighbor trajectories around the reference agent.

        Only uses PAST samples (for encoding). Returns trajectories in local coords.

        Returns:
            neighbors: (max_neighbors, past_steps, 2)
            mask: (max_neighbors,) True for valid neighbors
        """
        neighbor_trajectories = []

        for past_sample_token in entry['past_sample_tokens']:
            ann_tokens = self.sample_to_anns[past_sample_token]
            frame_neighbors = []

            for ann_token in ann_tokens:
                ann = self.nusc.get('sample_annotation', ann_token)
                category_name = ann['category_name']

                if ann['instance_token'] == ego_instance:
                    continue
                if not any(cat in category_name.lower() for cat in self.agents_of_interest):
                    continue

                pos = np.array(ann['translation'][:2])
                dist = np.linalg.norm(pos - ref_pos)
                if dist <= self.neighbor_radius:
                    local_pos = world_to_local(pos.reshape(1, 2), ref_pos, heading_angle).squeeze()
                    frame_neighbors.append((dist, local_pos))

            frame_neighbors.sort(key=lambda x: x[0])
            neighbor_trajectories.append(frame_neighbors)

        max_timesteps = len(entry['past_sample_tokens'])
        neighbor_obs = np.zeros((self.max_neighbors, max_timesteps, 2))
        neighbor_mask = np.zeros(self.max_neighbors, dtype=bool)

        for t, frame_neighbors in enumerate(neighbor_trajectories):
            n_found = min(len(frame_neighbors), self.max_neighbors)
            for n in range(n_found):
                _, local_pos = frame_neighbors[n]
                neighbor_obs[n, t] = local_pos
                neighbor_mask[n] = True

        return torch.FloatTensor(neighbor_obs), torch.BoolTensor(neighbor_mask)


def nuscenes_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for nuScenes trajectories.

    Shapes:
        history: (B, past_steps, 2)
        future_rel: (B, future_steps, 2)
        neighbors: (B, max_neighbors, past_steps, 2)
        neighbor_mask: (B, max_neighbors)
        map: (B, 3, 64, 64)
    """
    max_neighbors = max(item['neighbors'].shape[0] for item in batch)
    max_past_steps = max(item['history'].shape[0] for item in batch)
    max_future_steps = max(item['future_rel'].shape[0] for item in batch)

    batch_size = len(batch)
    history = torch.zeros(batch_size, max_past_steps, 2)
    history_abs = torch.zeros(batch_size, max_past_steps, 2)
    future = torch.zeros(batch_size, max_future_steps, 2)
    future_rel = torch.zeros(batch_size, max_future_steps, 2)
    origin = torch.zeros(batch_size, 2)
    angle = torch.zeros(batch_size, 1)

    neighbors = torch.zeros(batch_size, max_neighbors, max_past_steps, 2)
    neighbor_mask = torch.zeros(batch_size, max_neighbors, dtype=torch.bool)

    map_patches = torch.zeros(batch_size, 3, 64, 64)

    instance_tokens = []
    agent_types = []
    category_names = []
    locations = []

    for i, item in enumerate(batch):
        p_steps = item['history'].shape[0]
        f_steps = item['future_rel'].shape[0]
        n_neighbors = item['neighbors'].shape[0]

        history[i, :p_steps] = item['history']
        history_abs[i, :p_steps] = item['history_abs']
        future[i, :f_steps] = item['future']
        future_rel[i, :f_steps] = item['future_rel']
        origin[i] = item['origin']
        angle[i] = item['angle']

        neighbors[i, :n_neighbors, :p_steps] = item['neighbors'][:, :p_steps]
        neighbor_mask[i, :n_neighbors] = item['neighbor_mask']

        if 'map' in item and item['map'] is not None:
            map_patches[i] = item['map']

        instance_tokens.append(item['instance_token'])
        agent_types.append(item['agent_type'])
        category_names.append(item['category_name'])
        locations.append(item['location'])

    return {
        'history': history,
        'history_abs': history_abs,
        'future': future,
        'future_rel': future_rel,
        'origin': origin,
        'angle': angle,
        'neighbors': neighbors,
        'neighbor_mask': neighbor_mask,
        'instance_tokens': instance_tokens,
        'agent_types': agent_types,
        'category_names': category_names,
        'locations': locations,
        'map': map_patches,
    }