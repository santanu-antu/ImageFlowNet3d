import os
import glob
import numpy as np
import nibabel as nib
import csv
from tqdm import tqdm

def normalize(volume):
    """
    High-contrast normalization:
    Clip to [5, 95] percentiles (within mask), then rescale to [-1, 1].
    Set background (outside non-zero mask) to -1 (black).
    """
    # Create mask of non-zero elements
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = volume != 0
    
    if not np.any(mask):
        return np.full_like(volume, -1.0)
        
    vals = volume[mask]
    
    # Clip to 5th and 95th percentiles of non-zero voxels for higher contrast
    lo = np.percentile(vals, 5.0)
    hi = np.percentile(vals, 95.0)
    
    if hi - lo < 1e-8:
        return np.full_like(volume, -1.0)
        
    volume = np.clip(volume, lo, hi)
    
    # Rescale to [-1, 1]
    volume = (volume - lo) / (hi - lo)
    volume = volume * 2.0 - 1.0
    
    # Reset background to -1.0 (black)
    processed_volume = np.full_like(volume, -1.0)
    processed_volume[mask] = volume[mask]
    
    return processed_volume.astype(np.float32)

def pad_to_shape(volume, target_shape=(256, 256, 256)):
    """Pad or crop volume to target shape, centering the content."""
    current_shape = volume.shape
    pad_width = []
    slices = []
    
    for i in range(3):
        diff = target_shape[i] - current_shape[i]
        
        if diff >= 0:
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
            slices.append(slice(None)) 
        else:
            crop_start = abs(diff) // 2
            crop_end = crop_start + target_shape[i]
            pad_width.append((0, 0))
            slices.append(slice(crop_start, crop_end))
            
    if any(s != slice(None) for s in slices):
        volume = volume[tuple(slices)]
        
    if any(p != (0, 0) for p in pad_width):
        # Pad with -1 (black) to match the background
        volume = np.pad(volume, pad_width, mode='constant', constant_values=-1)
        
    return volume

def preprocess_adni():
    # Define paths relative to this script location (src/preprocessing/)
    # Script is in src/preprocessing/
    # Data is in data/registered_ADNI/ (which is ../../data/registered_ADNI relative to script)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    data_root = os.path.join(project_root, 'data', 'registered_ADNI')
    output_root = os.path.join(project_root, 'data', 'brain_ADNI')
    
    print(f"Data root: {data_root}")
    print(f"Output root: {output_root}")
    
    os.makedirs(output_root, exist_ok=True)
    
    # Subject folders look like 002_S_5018, 005_S_6427, etc.
    subject_dirs = sorted(glob.glob(os.path.join(data_root, '*_S_*')))
    
    print(f"Found {len(subject_dirs)} subjects.")
    
    for subject_dir in tqdm(subject_dirs):
        subject_id = os.path.basename(subject_dir)
        
        # Path to timepoints CSV
        csv_path = os.path.join(subject_dir, 'timepoints.csv')
        if not os.path.exists(csv_path):
            continue
            
        # Create output directory for this subject
        subject_out_dir = os.path.join(output_root, subject_id)
        os.makedirs(subject_out_dir, exist_ok=True)
        
        # Read CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timepoint = row['timepoint']
                    days = int(row['days_since_baseline'])
                    warp_file_path = row['warp_file']
    
                    filename = os.path.basename(warp_file_path)
                    
                    # Construct path to the file
                    # Structure: data_root/subject_id/timepoint_X/filename
                    nii_path = os.path.join(subject_dir, f'timepoint_{timepoint}', filename)
                    
                    if not os.path.exists(nii_path):
                        print(f"Warning: File not found: {nii_path}")
                        continue
                    
                    # Load volume
                    img = nib.load(nii_path)
                    volume = img.get_fdata()
                    
                    # Normalize
                    volume = normalize(volume)
                    
                    # Pad to 256x256x256
                    volume = pad_to_shape(volume, target_shape=(256, 256, 256))
                    
                    # Save as .npy
                    # Naming convention: subject_ID_time_DAYS.npy
                    # Using 4 digits for days to cover up to 9999 days (approx 27 years)
                    out_filename = f"{subject_id}_time_{days:04d}.npy"
                    out_path = os.path.join(subject_out_dir, out_filename)
                    
                    np.save(out_path, volume)
                    
                except Exception as e:
                    print(f"Error processing row for subject {subject_id}: {e}")

if __name__ == '__main__':
    preprocess_adni()
