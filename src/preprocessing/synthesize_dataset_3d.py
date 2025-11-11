import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from typing import Tuple


def _generate_longitudinal_3d(volume_shape: Tuple[int, int, int] = (32, 128, 128),
                               num_images: int = 10,
                               initial_radius: Tuple[int, int, int] = (4, 12, 12),
                               final_radius: Tuple[int, int, int] = (8, 24, 24),
                               random_seed: int | None = None):
    """
    Generate a sequence of 3D volumes with a large cuboid (eye) and a growing ellipsoid (lesion).

    Returns a list of uint8 volumes with shape [D, H, W].
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    D, H, W = volume_shape
    volumes = [np.zeros((D, H, W), dtype=np.uint8) for _ in range(num_images)]

    # Large cuboid background ("eye")
    padD, padH, padW = max(2, D // 8), max(2, H // 8), max(2, W // 8)
    d0, d1 = padD, D - padD
    h0, h1 = padH, H - padH
    w0, w1 = padW, W - padW
    base_val = np.random.randint(64, 160)
    for vol in volumes:
        vol[d0:d1, h0:h1, w0:w1] = base_val

    # Growing ellipsoid parameters
    # Clamp radii to fit within the cuboid
    max_rz = max(1, min(final_radius[0], (d1 - d0) // 2 - 1))
    max_ry = max(1, min(final_radius[1], (h1 - h0) // 2 - 1))
    max_rx = max(1, min(final_radius[2], (w1 - w0) // 2 - 1))
    min_rz = min(initial_radius[0], max_rz)
    min_ry = min(initial_radius[1], max_ry)
    min_rx = min(initial_radius[2], max_rx)

    # Choose center ensuring space for the largest ellipsoid
    if d0 + max_rz + 1 < d1 - max_rz:
        cz = np.random.randint(d0 + max_rz, d1 - max_rz)
    else:
        cz = (d0 + d1) // 2
    if h0 + max_ry + 1 < h1 - max_ry:
        cy = np.random.randint(h0 + max_ry, h1 - max_ry)
    else:
        cy = (h0 + h1) // 2
    if w0 + max_rx + 1 < w1 - max_rx:
        cx = np.random.randint(w0 + max_rx, w1 - max_rx)
    else:
        cx = (w0 + w1) // 2

    rz_list = np.linspace(min_rz, max_rz, num_images)
    ry_list = np.linspace(min_ry, max_ry, num_images)
    rx_list = np.linspace(min_rx, max_rx, num_images)

    zz = np.arange(D)[:, None, None]
    yy = np.arange(H)[None, :, None]
    xx = np.arange(W)[None, None, :]

    lesion_val = np.random.randint(180, 255)
    for i in range(num_images):
        rz, ry, rx = float(rz_list[i]), float(ry_list[i]), float(rx_list[i])
        # Ellipsoid mask
        mask = (((zz - cz) / max(rz, 1e-6)) ** 2 +
                ((yy - cy) / max(ry, 1e-6)) ** 2 +
                ((xx - cx) / max(rx, 1e-6)) ** 2) <= 1.0
        volumes[i][mask] = lesion_val

    return volumes, (cz, cy, cx)


def _translate_3d(vol: np.ndarray, tx: int, ty: int, tz: int) -> np.ndarray:
    """Translate a 3D volume by (tz, ty, tx) with zero fill. Uses slicing/padding (no SciPy)."""
    D, H, W = vol.shape
    out = np.zeros_like(vol)
    z_src = slice(max(0, -tz), min(D, D - tz))
    y_src = slice(max(0, -ty), min(H, H - ty))
    x_src = slice(max(0, -tx), min(W, W - tx))
    z_dst = slice(max(0, tz), min(D, D + tz))
    y_dst = slice(max(0, ty), min(H, H + ty))
    x_dst = slice(max(0, tx), min(W, W + tx))
    out[z_dst, y_dst, x_dst] = vol[z_src, y_src, x_src]
    return out


def _rotate_z_3d(vol: np.ndarray, angle_deg: float, center_xy: Tuple[int, int] | None = None) -> np.ndarray:
    """
    Rotate the volume around the Z-axis by rotating each [H, W] slice with OpenCV.
    """
    D, H, W = vol.shape
    out = np.zeros_like(vol)
    if center_xy is None:
        center = (W // 2, H // 2)
    else:
        center = (int(center_xy[1]), int(center_xy[0]))
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    for z in range(D):
        out[z] = cv2.warpAffine(vol[z], rot_mat, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    return out


def synthesize_dataset_3d(save_folder: str, num_subjects: int = 100,
                          volume_shape: Tuple[int, int, int] = (32, 128, 128), num_images: int = 10) -> None:
    """
    Create 4 datasets analogous to 2D: base, translation, rotation, mixing.
    Volumes saved as .npy with shape [D, H, W] and uint8 values.
    """
    os.makedirs(save_folder, exist_ok=True)

    for subject_idx in tqdm(range(num_subjects)):
        vols, center = _generate_longitudinal_3d(volume_shape=volume_shape,
                                                 num_images=num_images,
                                                 random_seed=subject_idx)
        base_dir = os.path.join(save_folder, 'base', f'subject_{subject_idx:05d}')
        trans_dir = os.path.join(save_folder, 'translation', f'subject_{subject_idx:05d}')
        rot_dir = os.path.join(save_folder, 'rotation', f'subject_{subject_idx:05d}')
        mix_dir = os.path.join(save_folder, 'mixing', f'subject_{subject_idx:05d}')
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(trans_dir, exist_ok=True)
        os.makedirs(rot_dir, exist_ok=True)
        os.makedirs(mix_dir, exist_ok=True)

        # Save base
        for t, v in enumerate(vols):
            np.save(os.path.join(base_dir, f'subject_{subject_idx:05d}_time_{t:03d}.npy'), v)

        # Translation: linear tx, sinusoidal ty, small tz drift
        max_tx, max_ty, max_tz = volume_shape[2] // 8, volume_shape[1] // 8, max(1, volume_shape[0] // 16)
        for t, v in enumerate(vols):
            tx = int(2 * max_tx / (len(vols) - 1) * t - max_tx)
            ty = int(max_ty * np.cos(t / len(vols) * 2 * np.pi))
            tz = int((t - len(vols) // 2) / max(1, len(vols) - 1) * 2 * max_tz)
            vt = _translate_3d(v, tx=tx, ty=ty, tz=tz)
            np.save(os.path.join(trans_dir, f'subject_{subject_idx:05d}_time_{t:03d}.npy'), vt)

        # Rotation around Z with increasing angle
        for t, v in enumerate(vols):
            angle = float(np.linspace(0, 180, len(vols))[t])
            vr = _rotate_z_3d(v, angle_deg=angle)
            np.save(os.path.join(rot_dir, f'subject_{subject_idx:05d}_time_{t:03d}.npy'), vr)

        # Mixing: pick one of base/trans/rot at each time
        for t in range(len(vols)):
            choice = np.random.choice(['base', 'trans', 'rot'])
            if choice == 'base':
                src = os.path.join(base_dir, f'subject_{subject_idx:05d}_time_{t:03d}.npy')
            elif choice == 'trans':
                src = os.path.join(trans_dir, f'subject_{subject_idx:05d}_time_{t:03d}.npy')
            else:
                src = os.path.join(rot_dir, f'subject_{subject_idx:05d}_time_{t:03d}.npy')
            v = np.load(src)
            np.save(os.path.join(mix_dir, f'subject_{subject_idx:05d}_time_{t:03d}.npy'), v)

    return None


if __name__ == '__main__':
    # Default save folder under repo root
    script_dir = os.path.dirname(os.path.realpath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    default_save = os.path.join(repo_root, 'data', 'synthesized3d')

    parser = argparse.ArgumentParser(description='Synthesize 3D dataset analogous to 2D synthetic.')
    parser.add_argument('--save-folder', type=str, default=default_save)
    parser.add_argument('--num-subjects', type=int, default=100)
    parser.add_argument('--num-images', type=int, default=10)
    parser.add_argument('--volume-shape', type=str, default='(32,128,128)',
                        help='Tuple as string, e.g., (32,128,128) for (D,H,W)')
    args = parser.parse_args()

    vol_shape = eval(args.volume_shape)
    synthesize_dataset_3d(save_folder=args.save_folder,
                          num_subjects=args.num_subjects,
                          volume_shape=tuple(vol_shape),
                          num_images=args.num_images)
