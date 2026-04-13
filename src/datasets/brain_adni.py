import itertools
import os
from typing import Literal, List, Tuple
from glob import glob

import numpy as np
from torch.utils.data import Dataset

root_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])

class BrainADNIDataset(Dataset):
    """
    Dataset for loading 3D ADNI brain volumes.
    
    Expected directory structure:
        base_path/
            subject_ID/
                subject_ID_time_DDDD.npy
                ...
    
    Each .npy file contains a 3D volume of shape (D, H, W).
    """

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_ADNI/',
                 target_dim: Tuple[int, int, int] = (256, 256, 256)):
        """
        Args:
            base_path: Path to the brain_ADNI directory
            target_dim: Target volume dimensions (D, H, W).
        """
        super().__init__()

        self.target_dim = target_dim
        # Normalize path construction
        folder_glob = os.path.join(base_path, '*/')
        all_subject_folders = sorted(glob(folder_glob))

        self.volumes_by_subject = []
        self.subject_ids = []

        # Track maximum timestamp across the dataset
        self.max_t = 0

        for folder in all_subject_folders:
            paths = sorted(glob(os.path.join(folder, '*.npy')))
            # Filter out subjects with fewer than 2 timepoints
            if len(paths) >= 2:
                self.volumes_by_subject.append(paths)
                self.subject_ids.append(os.path.basename(os.path.normpath(folder)))
            
            for p in paths:
                try:
                    self.max_t = max(self.max_t, get_time_adni(p))
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


class BrainADNISubset(BrainADNIDataset):
    """
    A subset of BrainADNIDataset.
    
    Organizes volumes such that each __getitem__ call returns a pair of
    [volume_start, volume_end] and [t_start, t_end].
    """

    def __init__(self,
                 main_dataset: BrainADNIDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs', 'array', 'min_max_pair']):
        """
        Args:
            main_dataset: The parent BrainADNIDataset
            subset_indices: List of subject indices to include in this subset
            return_format: How to return data
                - 'one_pair': Return one randomly sampled pair per subject
                - 'all_pairs': Return all possible pairs
                - 'array': Return all timepoints as an array
                - 'min_max_pair': Return only the pair (min_t, max_t) for each subject
        """
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format

        self.volumes_by_subject = [
            main_dataset.volumes_by_subject[i] for i in subset_indices
        ]
        if hasattr(main_dataset, 'subject_ids'):
            self.subject_ids = [main_dataset.subject_ids[i] for i in subset_indices]

        self.all_volume_pairs = []
        if self.return_format == 'min_max_pair':
            for volume_list in self.volumes_by_subject:
                # Sort by time to ensure min and max
                sorted_volumes = sorted(volume_list, key=get_time_adni)
                if len(sorted_volumes) >= 2:
                    self.all_volume_pairs.append([sorted_volumes[0], sorted_volumes[-1]])
        else:
            for volume_list in self.volumes_by_subject:
                pair_indices = list(
                    itertools.combinations(np.arange(len(volume_list)), r=2))
                for (idx1, idx2) in pair_indices:
                    self.all_volume_pairs.append(
                        [volume_list[idx1], volume_list[idx2]])

    def __len__(self) -> int:
        if self.return_format == 'one_pair':
            return len(self.volumes_by_subject)
        elif self.return_format in ['all_pairs', 'min_max_pair']:
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
            timestamps = np.array([get_time_adni(p) for p in sampled_pair])

        elif self.return_format in ['all_pairs', 'min_max_pair']:
            queried_pair = self.all_volume_pairs[idx]
            volumes = np.array([
                load_volume(p, target_dim=self.target_dim) for p in queried_pair
            ])
            timestamps = np.array([get_time_adni(p) for p in queried_pair])

        elif self.return_format == 'array':
            queried_subject = self.volumes_by_subject[idx]
            volumes = np.array([
                load_volume(p, target_dim=self.target_dim)
                for p in queried_subject
            ])
            timestamps = np.array([get_time_adni(p) for p in queried_subject])

        return volumes, timestamps


def load_volume(path: str, target_dim: Tuple[int, int, int] = None) -> np.ndarray:
    """
    Load a 3D volume as numpy array from a path string.
    
    Args:
        path: Path to .npy file containing volume
        target_dim: Target dimensions (D, H, W). Currently not resizing, 
                   assumes volumes are already correct size (256x256x256).
    
    Returns:
        Volume of shape (1, D, H, W) with channel dimension
    """
    volume = np.load(path)  # Shape: (D, H, W)
    
    # Normalize to [-1, 1] range
    # Preprocessing now saves in [-1, 1] directly
    # volume = volume * 2 - 1
    
    # Add channel dimension: (D, H, W) -> (1, D, H, W)
    volume = volume[np.newaxis, ...]
    
    return volume.astype(np.float32)


def get_time_adni(path: str) -> float:
    """
    Get the timestamp information from a path string.
    
    Expected filename format: subject_ID_time_DDDD.npy
    
    Args:
        path: Path to the .npy file
        
    Returns:
        Timestamp as float (days)
    """
    basename = os.path.basename(path)
    # Extract time from filename like "002_S_5018_time_0095.npy"
    # Split by '_time_'
    time_str = basename.split('_time_')[1].replace('.npy', '')
    time = float(time_str)
    return time
