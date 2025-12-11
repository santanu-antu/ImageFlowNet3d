"""
Utility functions for patchified 3D volume processing.

Supports:
- Extracting patches from large volumes
- Reassembling patches with overlap blending
- Both training and inference modes
"""
import numpy as np
import torch
from typing import Tuple, List, Optional


def extract_patches_3d(
    volume: torch.Tensor,
    patch_size: int,
    stride: Optional[int] = None
) -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    """
    Extract non-overlapping or overlapping patches from a 3D volume.
    
    Args:
        volume: Input volume of shape [B, C, D, H, W] or [C, D, H, W]
        patch_size: Size of cubic patches (e.g., 64 for 64x64x64)
        stride: Stride between patches. If None, equals patch_size (non-overlapping)
        
    Returns:
        patches: Tensor of shape [N, C, patch_size, patch_size, patch_size]
        positions: List of (d, h, w) starting positions for each patch
    """
    if stride is None:
        stride = patch_size
    
    # Handle batch dimension
    if len(volume.shape) == 4:
        volume = volume.unsqueeze(0)  # Add batch dim
    
    B, C, D, H, W = volume.shape
    
    # Calculate number of patches in each dimension
    n_d = (D - patch_size) // stride + 1
    n_h = (H - patch_size) // stride + 1
    n_w = (W - patch_size) // stride + 1
    
    patches = []
    positions = []
    
    for b in range(B):
        for d_idx in range(n_d):
            for h_idx in range(n_h):
                for w_idx in range(n_w):
                    d_start = d_idx * stride
                    h_start = h_idx * stride
                    w_start = w_idx * stride
                    
                    patch = volume[b, :, 
                                   d_start:d_start+patch_size,
                                   h_start:h_start+patch_size,
                                   w_start:w_start+patch_size]
                    patches.append(patch)
                    positions.append((d_start, h_start, w_start))
    
    patches = torch.stack(patches, dim=0)  # [N, C, patch_size, patch_size, patch_size]
    
    return patches, positions


def extract_random_patch_3d(
    volume: torch.Tensor,
    patch_size: int
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Extract a single random patch from a 3D volume.
    
    Args:
        volume: Input volume of shape [B, C, D, H, W] or [C, D, H, W]
        patch_size: Size of cubic patch
        
    Returns:
        patch: Tensor of shape [C, patch_size, patch_size, patch_size]
        position: (d, h, w) starting position
    """
    if len(volume.shape) == 4:
        C, D, H, W = volume.shape
    else:
        _, C, D, H, W = volume.shape
        volume = volume[0]
    
    # Random starting position
    d_start = np.random.randint(0, D - patch_size + 1)
    h_start = np.random.randint(0, H - patch_size + 1)
    w_start = np.random.randint(0, W - patch_size + 1)
    
    patch = volume[:, 
                   d_start:d_start+patch_size,
                   h_start:h_start+patch_size,
                   w_start:w_start+patch_size]
    
    return patch, (d_start, h_start, w_start)


def extract_corresponding_patches_3d(
    volume1: torch.Tensor,
    volume2: torch.Tensor,
    patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    """
    Extract corresponding random patches from two volumes at the same position.
    Used for training with paired volumes (start and end timepoints).
    
    Args:
        volume1: First volume [B, C, D, H, W] or [C, D, H, W]
        volume2: Second volume (same shape as volume1)
        patch_size: Size of cubic patch
        
    Returns:
        patch1, patch2: Corresponding patches
        position: (d, h, w) starting position
    """
    if len(volume1.shape) == 4:
        C, D, H, W = volume1.shape
    else:
        _, C, D, H, W = volume1.shape
        volume1 = volume1[0]
        volume2 = volume2[0]
    
    # Random starting position (same for both volumes)
    d_start = np.random.randint(0, D - patch_size + 1)
    h_start = np.random.randint(0, H - patch_size + 1)
    w_start = np.random.randint(0, W - patch_size + 1)
    
    patch1 = volume1[:, 
                     d_start:d_start+patch_size,
                     h_start:h_start+patch_size,
                     w_start:w_start+patch_size]
    
    patch2 = volume2[:, 
                     d_start:d_start+patch_size,
                     h_start:h_start+patch_size,
                     w_start:w_start+patch_size]
    
    return patch1, patch2, (d_start, h_start, w_start)


def reassemble_patches_3d(
    patches: torch.Tensor,
    positions: List[Tuple[int, int, int]],
    output_shape: Tuple[int, int, int, int, int],
    patch_size: int,
    overlap_mode: str = 'average'
) -> torch.Tensor:
    """
    Reassemble patches back into a full volume with overlap blending.
    
    Args:
        patches: Patches tensor of shape [N, C, patch_size, patch_size, patch_size]
        positions: List of (d, h, w) starting positions for each patch
        output_shape: Target output shape [B, C, D, H, W]
        patch_size: Size of each patch
        overlap_mode: How to handle overlapping regions
            - 'average': Average overlapping voxels
            - 'gaussian': Gaussian-weighted blending (smoother)
            - 'last': Last patch overwrites (fastest, may show seams)
            
    Returns:
        Reassembled volume of shape output_shape
    """
    B, C, D, H, W = output_shape
    device = patches.device
    dtype = patches.dtype
    
    # Create output volume and weight accumulator
    output = torch.zeros(output_shape, device=device, dtype=dtype)
    weights = torch.zeros(output_shape, device=device, dtype=dtype)
    
    # Create blending weights based on mode
    if overlap_mode == 'gaussian':
        # Gaussian weight: higher in center, lower at edges
        weight_1d = _gaussian_weights_1d(patch_size, device, dtype)
        patch_weight = weight_1d.view(-1, 1, 1) * weight_1d.view(1, -1, 1) * weight_1d.view(1, 1, -1)
        patch_weight = patch_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        patch_weight = patch_weight.expand(1, C, -1, -1, -1)
    else:
        patch_weight = torch.ones(1, C, patch_size, patch_size, patch_size, 
                                  device=device, dtype=dtype)
    
    # Accumulate patches
    for idx, (d_start, h_start, w_start) in enumerate(positions):
        patch = patches[idx:idx+1]  # [1, C, patch_size, patch_size, patch_size]
        
        if overlap_mode == 'last':
            output[0, :, 
                   d_start:d_start+patch_size,
                   h_start:h_start+patch_size,
                   w_start:w_start+patch_size] = patch[0]
        else:
            output[0, :, 
                   d_start:d_start+patch_size,
                   h_start:h_start+patch_size,
                   w_start:w_start+patch_size] += patch[0] * patch_weight[0]
            weights[0, :, 
                    d_start:d_start+patch_size,
                    h_start:h_start+patch_size,
                    w_start:w_start+patch_size] += patch_weight[0]
    
    if overlap_mode != 'last':
        # Normalize by accumulated weights
        output = output / (weights + 1e-8)
    
    return output


def _gaussian_weights_1d(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Generate 1D Gaussian weights for blending."""
    x = torch.linspace(-1, 1, size, device=device, dtype=dtype)
    sigma = 0.5  # Controls falloff speed
    weights = torch.exp(-x**2 / (2 * sigma**2))
    return weights


def sliding_window_inference_3d(
    volume: torch.Tensor,
    model: torch.nn.Module,
    patch_size: int,
    stride: Optional[int] = None,
    t: torch.Tensor = None,
    overlap_mode: str = 'gaussian',
    device: torch.device = None,
    progress_bar: bool = False
) -> torch.Tensor:
    """
    Perform inference on a large volume using sliding window approach.
    
    Args:
        volume: Input volume [B, C, D, H, W]
        model: The model to use for inference
        patch_size: Size of patches to process
        stride: Stride between patches. If None, defaults to patch_size // 2
        t: Time tensor for the model
        overlap_mode: Blending mode ('average', 'gaussian', 'last')
        device: Device to run inference on
        progress_bar: Whether to show progress bar
        
    Returns:
        Output volume of same shape as input
    """
    from tqdm import tqdm
    
    if stride is None:
        stride = patch_size // 2  # 50% overlap by default
    
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = volume.device
    
    B, C, D, H, W = volume.shape
    
    # Extract all patches with positions
    patches, positions = extract_patches_3d(volume, patch_size, stride)
    
    # Process patches
    output_patches = []
    patch_iter = range(len(patches))
    if progress_bar:
        patch_iter = tqdm(patch_iter, desc='Processing patches')
    
    for i in patch_iter:
        patch = patches[i:i+1].to(device)  # [1, C, patch_size, patch_size, patch_size]
        with torch.no_grad():
            out = model(x=patch, t=t)
        output_patches.append(out.cpu())
    
    output_patches = torch.cat(output_patches, dim=0)
    
    # Reassemble
    output = reassemble_patches_3d(
        output_patches,
        positions,
        output_shape=(B, C, D, H, W),
        patch_size=patch_size,
        overlap_mode=overlap_mode
    )
    
    return output


def get_num_patches(volume_size: int, patch_size: int, stride: Optional[int] = None) -> int:
    """
    Calculate the number of patches that will be extracted from a volume.
    
    Args:
        volume_size: Size of cubic volume (assumes D=H=W)
        patch_size: Size of cubic patches
        stride: Stride between patches. If None, equals patch_size
        
    Returns:
        Total number of patches (n_d * n_h * n_w)
    """
    if stride is None:
        stride = patch_size
    
    n = (volume_size - patch_size) // stride + 1
    return n ** 3


def check_patchify_compatibility(
    volume_size: int,
    patch_size: int,
    num_patches_per_dim: int
) -> Tuple[bool, str]:
    """
    Check if the volume can be evenly divided into the specified number of patches.
    
    Args:
        volume_size: Size of the volume (assumes cubic)
        patch_size: Desired patch size
        num_patches_per_dim: Number of patches per dimension
        
    Returns:
        (is_compatible, message)
    """
    # For non-overlapping: volume_size = num_patches_per_dim * patch_size
    expected_volume_size = num_patches_per_dim * patch_size
    
    if volume_size == expected_volume_size:
        return True, f"Volume {volume_size}³ can be evenly divided into {num_patches_per_dim}³ = {num_patches_per_dim**3} patches of size {patch_size}³"
    
    # Calculate what patch sizes would work
    if volume_size % num_patches_per_dim == 0:
        working_patch_size = volume_size // num_patches_per_dim
        return False, f"Volume {volume_size}³ with {num_patches_per_dim} patches/dim requires patch_size={working_patch_size}, not {patch_size}"
    else:
        return False, f"Volume {volume_size}³ cannot be evenly divided by {num_patches_per_dim}. Try different num_patches_per_dim."
