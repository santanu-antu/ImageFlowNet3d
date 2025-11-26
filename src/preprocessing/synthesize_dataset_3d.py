import os
import numpy as np
from tqdm import tqdm
from typing import Tuple


def _generate_longitudinal_3d(volume_shape: Tuple[int, int, int] = (32, 32, 32),
                               num_volumes: int = 10,
                               initial_radius: Tuple[int, int, int] = (2, 2, 2),
                               final_radius: Tuple[int, int, int] = (4, 5, 6),
                               random_seed: int = None):
    '''
    Generate longitudinal 3D volumes of a big cuboid containing a small ellipsoid.
    The big cuboid (analogous to eye) remains unchanged, while the small ellipsoid (analogous to geographic atrophy) grows.
    '''

    volumes = [np.zeros(volume_shape, dtype=np.float32) for _ in range(num_volumes)]

    if random_seed is not None:
        np.random.seed(random_seed)

    # Random gray values for cuboid and ellipsoid (normalized between 0 and 1)
    gray_cuboid = np.random.uniform(0.2, 0.5)
    gray_ellipsoid = np.random.uniform(0.7, 0.9)

    # First generate the big cuboid.
    cuboid_tl = [int(np.random.uniform(1/8*volume_shape[i], 1/16*volume_shape[i])) for i in range(3)]
    cuboid_br = [int(np.random.uniform(3/4*volume_shape[i], 7/8*volume_shape[i])) for i in range(3)]
    cuboid_centroid = np.mean([cuboid_tl, cuboid_br], axis=0)
    
    for volume in volumes:
        volume[cuboid_tl[0]:cuboid_br[0],
               cuboid_tl[1]:cuboid_br[1],
               cuboid_tl[2]:cuboid_br[2]] = gray_cuboid

    # Then generate the increasingly bigger ellipsoids.
    ellipsoid_centroid = [int(np.random.uniform(cuboid_tl[i]+final_radius[i],
                                                 cuboid_br[i]-final_radius[i])) for i in range(3)]
    radius_x_list = np.linspace(initial_radius[0], final_radius[0], num_volumes)
    radius_y_list = np.linspace(initial_radius[1], final_radius[1], num_volumes)
    radius_z_list = np.linspace(initial_radius[2], final_radius[2], num_volumes)
    
    for i, volume in enumerate(volumes):
        d_arr = np.linspace(0, volume_shape[0]-1, volume_shape[0])[:, None, None]
        h_arr = np.linspace(0, volume_shape[1]-1, volume_shape[1])[None, :, None]
        w_arr = np.linspace(0, volume_shape[2]-1, volume_shape[2])[None, None, :]
        
        ellipsoid_mask = ((d_arr-ellipsoid_centroid[0])/radius_x_list[i])**2 + \
                         ((h_arr-ellipsoid_centroid[1])/radius_y_list[i])**2 + \
                         ((w_arr-ellipsoid_centroid[2])/radius_z_list[i])**2 <= 1
        volume[ellipsoid_mask] = gray_ellipsoid

    return volumes, cuboid_centroid


def _apply_translation_3d(volume: np.ndarray, translation: Tuple[int, int, int]) -> np.ndarray:
    '''Apply 3D translation to a volume using numpy roll.'''
    volume_trans = np.roll(volume, translation[0], axis=0)
    volume_trans = np.roll(volume_trans, translation[1], axis=1)
    volume_trans = np.roll(volume_trans, translation[2], axis=2)
    return volume_trans


def _apply_rotation_3d(volume: np.ndarray, angle: float, axis: int, centroid: np.ndarray) -> np.ndarray:
    '''
    Apply 3D rotation around a specified axis.
    axis: 0 for D-axis, 1 for H-axis, 2 for W-axis
    '''
    from scipy.ndimage import rotate
    # Rotate around the specified axis
    # Order determines which axes form the plane of rotation
    if axis == 0:  # Rotate in H-W plane (around D-axis)
        volume_rot = rotate(volume, angle, axes=(1, 2), reshape=False, order=1)
    elif axis == 1:  # Rotate in D-W plane (around H-axis)
        volume_rot = rotate(volume, angle, axes=(0, 2), reshape=False, order=1)
    else:  # Rotate in D-H plane (around W-axis)
        volume_rot = rotate(volume, angle, axes=(0, 1), reshape=False, order=1)
    return volume_rot


def synthesize_dataset_3d(save_folder: str = None, num_subjects: int = 200):
    '''
    Synthesize 4 3D datasets (analogous to 2D version).
    1. The first dataset has no spatial variation. It has voxel-level alignment temporally.
    2. The second dataset has a predictable translation factor.
    3. The third dataset has a predictable rotation factor.
    4. The fourth dataset is irregular. At each time point, we randomly pick a volume from 1/2/3 at that time point.
    '''
    if save_folder is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        save_folder = os.path.join(project_root, 'data', 'synthesized3d') + '/'

    for subject_idx in tqdm(range(num_subjects)):
        volumes, cuboid_centroid = _generate_longitudinal_3d(random_seed=subject_idx)
        volumes_trans, volumes_rot = [], []

        # Dataset 1: Base (no transformation)
        dataset = 'base'
        os.makedirs(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5), exist_ok=True)
        for time_idx, vol in enumerate(volumes):
            filename = save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5) + \
                      '/subject_%s_time_%s.npy' % (str(subject_idx).zfill(5), str(time_idx).zfill(3))
            np.save(filename, vol)

        # Dataset 2: Translation
        dataset = 'translation'
        os.makedirs(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5), exist_ok=True)
        max_trans_d, max_trans_h, max_trans_w = 4, 4, 4  # Proportional to 32/256 * 32 from 2D
        for time_idx, vol in enumerate(volumes):
            translation_d = int(2 * max_trans_d / (len(volumes) - 1) * time_idx - max_trans_d)
            translation_h = int(max_trans_h * np.cos(time_idx / len(volumes) * 2*np.pi))
            translation_w = int(max_trans_w * np.sin(time_idx / len(volumes) * 2*np.pi))
            vol_trans = _apply_translation_3d(vol, (translation_d, translation_h, translation_w))
            filename = save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5) + \
                      '/subject_%s_time_%s.npy' % (str(subject_idx).zfill(5), str(time_idx).zfill(3))
            np.save(filename, vol_trans)
            volumes_trans.append(vol_trans)

        # Dataset 3: Rotation
        dataset = 'rotation'
        os.makedirs(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5), exist_ok=True)
        for time_idx, vol in enumerate(volumes):
            angle = np.linspace(0, 180, len(volumes))[time_idx]
            # Rotate around D-axis (in H-W plane, analogous to 2D rotation)
            vol_rot = _apply_rotation_3d(vol, angle, axis=0, centroid=cuboid_centroid)
            filename = save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5) + \
                      '/subject_%s_time_%s.npy' % (str(subject_idx).zfill(5), str(time_idx).zfill(3))
            np.save(filename, vol_rot)
            volumes_rot.append(vol_rot)

        # Dataset 4: Mixing (randomly pick from previous lists)
        dataset = 'mixing'
        os.makedirs(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5), exist_ok=True)
        for time_idx in range(len(volumes)):
            chosen_list = np.random.choice(['volumes', 'volumes_trans', 'volumes_rot'])
            filename = save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5) + \
                      '/subject_%s_time_%s.npy' % (str(subject_idx).zfill(5), str(time_idx).zfill(3))
            np.save(filename, eval(chosen_list)[time_idx])
    return


if __name__ == '__main__':
    synthesize_dataset_3d()
