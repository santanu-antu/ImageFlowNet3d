"""Disk-backed Synthetic 3D dataset analogous to 2D synthetic.

Loads volumes saved as .npy from folders:
  base_path/{base,translation,rotation,mixing}/subject_00000/subject_00000_time_000.npy

Returns pairs [2, C=1, D, H, W] with timestamps parsed from filenames.
"""

import os
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class Synthetic3DDataset(Dataset):
    def __init__(self,
                 base_path: str = '../../data/synthesized3d/',
                 image_folder: str = 'base/',
                 target_dim: Tuple[int, int, int] = (32, 128, 128)):
        super().__init__()
        self.base_path = base_path
        self.image_folder = image_folder
        self.target_dim = tuple(target_dim)

        folder_glob = os.path.join(base_path, image_folder, '*/')
        all_folders = sorted(glob(folder_glob))
        self.image_by_patient: List[List[str]] = []
        self.max_t = 0.0
        for folder in all_folders:
            paths = sorted(glob(os.path.join(folder, '*.npy')))
            if len(paths) >= 2:
                self.image_by_patient.append(paths)
            for p in paths:
                try:
                    self.max_t = max(self.max_t, get_time_3d(p))
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        return 1


class Synthetic3DSubset(Synthetic3DDataset):
    def __init__(self,
                 main_dataset: Synthetic3DDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = 'one_pair'):
        super().__init__(base_path=main_dataset.base_path,
                         image_folder=main_dataset.image_folder,
                         target_dim=main_dataset.target_dim)
        self.return_format = return_format
        self.image_by_patient = [main_dataset.image_by_patient[i] for i in subset_indices]

        self.all_image_pairs: List[List[str]] = []
        for image_list in self.image_by_patient:
            n = len(image_list)
            for i in range(n):
                for j in range(i + 1, n):
                    self.all_image_pairs.append([image_list[i], image_list[j]])

    def __len__(self) -> int:
        if self.return_format == 'one_pair':
            return len(self.image_by_patient)
        elif self.return_format == 'all_pairs':
            return len(self.all_image_pairs)
        elif self.return_format == 'array':
            return len(self.image_by_patient)
        else:
            raise ValueError('Unsupported return_format: %s' % self.return_format)

    def __getitem__(self, idx):
        if self.return_format == 'one_pair':
            image_list = self.image_by_patient[idx]
            if len(image_list) < 2:
                raise IndexError('Patient has fewer than 2 visits')
            # Deterministic first/last pair to mirror 2D behavior
            pair = [image_list[0], image_list[-1]]
        elif self.return_format == 'all_pairs':
            pair = self.all_image_pairs[idx]
        elif self.return_format == 'array':
            pair = self.image_by_patient[idx]
        else:
            raise ValueError('Unsupported return_format')

        images = np.array([load_volume_3d(p, target_dim=self.target_dim) for p in pair])
        timestamps = np.array([get_time_3d(p) for p in pair], dtype=np.float32)
        return images, timestamps


def load_volume_3d(path: str, target_dim: Tuple[int, int, int] | None = None) -> np.ndarray:
    """Load a 3D volume saved as .npy and resize to target_dim if provided.
    Returns float32 array normalized to [-1, 1] with shape [C=1, D, H, W].
    """
    vol = np.load(path)
    assert vol.ndim == 3, f"Expected 3D volume at {path}, got shape {vol.shape}"

    if target_dim is not None:
        vol = resize_volume_nn(vol, target_dim)

    # Normalize to [-1, 1]
    if vol.dtype != np.float32:
        vol = vol.astype(np.float32)
    vol = (vol / 255.0) * 2.0 - 1.0

    # Add channel first
    vol = vol[None, ...]
    return vol


def resize_volume_nn(vol: np.ndarray, target_dim: Tuple[int, int, int]) -> np.ndarray:
    """Nearest-neighbor resize for 3D: z with index mapping, yx with cv2 per-slice."""
    D_t, H_t, W_t = target_dim
    D, H, W = vol.shape

    # Resize in-plane using OpenCV (per slice)
    if (H, W) != (H_t, W_t):
        out = np.zeros((D, H_t, W_t), dtype=vol.dtype)
        for z in range(D):
            out[z] = cv2.resize(vol[z], (W_t, H_t), interpolation=cv2.INTER_NEAREST)
    else:
        out = vol

    # Resize depth with nearest neighbor
    if D != D_t:
        z_idx = np.linspace(0, D - 1, num=D_t)
        z_idx_nn = np.round(z_idx).astype(int)
        out = out[z_idx_nn]

    return out


def get_time_3d(path: str) -> float:
    """Parse timestamp from filename '*_time_XXX.npy'."""
    basename = os.path.basename(path)
    time_str = basename.split('time_')[1].replace('.npy', '')
    assert len(time_str) == 3
    return float(time_str)
