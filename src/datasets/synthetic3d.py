"""
3D Synthetic Dataset for volumetric data (e.g., 64x64x64 volumes).
Analogous to synthetic.py but for true 3D volumetric data.
"""
import itertools
import os
from typing import Literal, List, Tuple
from glob import glob

import numpy as np
from torch.utils.data import Dataset


class Synthetic3DDataset(Dataset):
    """
    Dataset for loading 3D synthetic volumes.
    
    Expected directory structure:
        base_path/
            image_folder/
                subject_00000/
                    subject_00000_time_000.npy
                    subject_00000_time_001.npy
                    ...
                subject_00001/
                    ...
    
    Each .npy file contains a 3D volume of shape (D, H, W).
    """

    def __init__(self,
                 base_path: str = '../../data/synthesized3d/',
                 image_folder: str = 'base/',
                 target_dim: Tuple[int, int, int] = (64, 64, 64)):
        """
        Args:
            base_path: Path to the synthesized3d directory
            image_folder: Subfolder name (base, translation, rotation, mixing)
            target_dim: Target volume dimensions (D, H, W). If volumes don't match,
                       they could be resized, but typically should match.
        """
        super().__init__()

        self.target_dim = target_dim
        # Normalize path construction to avoid issues with trailing slashes
        folder_glob = os.path.join(base_path, image_folder, '*/')
        all_subject_folders = sorted(glob(folder_glob))

        self.volumes_by_subject = []

        # Track maximum timestamp across the dataset
        self.max_t = 0

        for folder in all_subject_folders:
            paths = sorted(glob(os.path.join(folder, '*.npy')))
            if len(paths) >= 2:
                self.volumes_by_subject.append(paths)
            for p in paths:
                try:
                    self.max_t = max(self.max_t, get_time_3d(p))
                except Exception:
                    pass

        # Fallback if max_t is still 0
        if self.max_t == 0:
            self.max_t = 1.0

    def __len__(self) -> int:
        return len(self.volumes_by_subject)

    def num_image_channel(self) -> int:
        """Number of image channels. 3D volumes are single-channel."""
        return 1


class Synthetic3DSubset(Synthetic3DDataset):
    """
    A subset of Synthetic3DDataset.
    
    Organizes volumes such that each __getitem__ call returns a pair of
    [volume_start, volume_end] and [t_start, t_end].
    """

    def __init__(self,
                 main_dataset: Synthetic3DDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs', 'array']):
        """
        Args:
            main_dataset: The parent Synthetic3DDataset
            subset_indices: List of subject indices to include in this subset
            return_format: How to return data
                - 'one_pair': Return one randomly sampled pair per subject
                - 'all_pairs': Return all possible pairs
                - 'array': Return all timepoints as an array
        """
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format

        self.volumes_by_subject = [
            main_dataset.volumes_by_subject[i] for i in subset_indices
        ]

        self.all_volume_pairs = []
        for volume_list in self.volumes_by_subject:
            pair_indices = list(
                itertools.combinations(np.arange(len(volume_list)), r=2))
            for (idx1, idx2) in pair_indices:
                self.all_volume_pairs.append(
                    [volume_list[idx1], volume_list[idx2]])

    def __len__(self) -> int:
        if self.return_format == 'one_pair':
            return len(self.volumes_by_subject)
        elif self.return_format == 'all_pairs':
            return len(self.all_volume_pairs)
        elif self.return_format == 'array':
            return len(self.volumes_by_subject)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        if self.return_format == 'one_pair':
            volume_list = self.volumes_by_subject[idx]
            pair_indices = list(
                itertools.combinations(np.arange(len(volume_list)), r=2))
            sampled_pair = [
                volume_list[i]
                for i in pair_indices[np.random.choice(len(pair_indices))]
            ]
            # Shape: [2, 1, D, H, W] - batch of 2 volumes with channel dim
            volumes = np.array([
                load_volume(p, target_dim=self.target_dim) for p in sampled_pair
            ])
            timestamps = np.array([get_time_3d(p) for p in sampled_pair])

        elif self.return_format == 'all_pairs':
            queried_pair = self.all_volume_pairs[idx]
            volumes = np.array([
                load_volume(p, target_dim=self.target_dim) for p in queried_pair
            ])
            timestamps = np.array([get_time_3d(p) for p in queried_pair])

        elif self.return_format == 'array':
            queried_subject = self.volumes_by_subject[idx]
            volumes = np.array([
                load_volume(p, target_dim=self.target_dim)
                for p in queried_subject
            ])
            timestamps = np.array([get_time_3d(p) for p in queried_subject])

        return volumes, timestamps


def load_volume(path: str, target_dim: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Load a 3D volume as numpy array from a path string.
    
    Args:
        path: Path to .npy file containing volume
        target_dim: Target dimensions (D, H, W). Currently not resizing, 
                   assumes volumes are already correct size.
    
    Returns:
        Volume of shape (1, D, H, W) with channel dimension
    """
    volume = np.load(path)  # Shape: (D, H, W)
    
    # Normalize to [-1, 1] range
    # Assuming volumes are in [0, 1] or similar range
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        volume = (volume - vmin) / (vmax - vmin)  # Normalize to [0, 1]
    volume = volume * 2 - 1  # Scale to [-1, 1]
    
    # Add channel dimension: (D, H, W) -> (1, D, H, W)
    volume = volume[np.newaxis, ...]
    
    return volume.astype(np.float32)


def get_time_3d(path: str) -> float:
    """
    Get the timestamp information from a path string.
    
    Expected filename format: subject_XXXXX_time_YYY.npy
    
    Args:
        path: Path to the .npy file
        
    Returns:
        Timestamp as float
    """
    basename = os.path.basename(path)
    # Extract time from filename like "subject_00000_time_005.npy"
    time_str = basename.split('time_')[1].replace('.npy', '')
    time = float(time_str)
    return time
