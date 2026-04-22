import os
import csv
import glob
import logging
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
import nibabel as nib
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(processName)s] %(message)s',
    datefmt='%H:%M:%S',
)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def normalize(volume):
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = volume != 0
    
    if not np.any(mask):
        return np.full_like(volume, -1.0)
        
    vals = volume[mask]
    lo = np.percentile(vals, 5.0)
    hi = np.percentile(vals, 95.0)
    
    if hi - lo < 1e-8:
        return np.full_like(volume, -1.0)
        
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo)
    volume = volume * 2.0 - 1.0
    
    processed_volume = np.full_like(volume, -1.0)
    processed_volume[mask] = volume[mask]
    
    return processed_volume.astype(np.float32)

def pad_or_crop(data, target_shape=(256, 256, 256)):
    # Since background will be -1.0, pad with -1.0 instead of 0
    pad = []
    crop = []
    for i in range(3):
        diff = target_shape[i] - data.shape[i]
        if diff >= 0:
            pad_lower = diff // 2
            pad_upper = diff - pad_lower
            pad.append((pad_lower, pad_upper))
            crop.append(slice(None))
        else:
            diff = -diff
            crop_lower = diff // 2
            crop_upper = diff - crop_lower
            pad.append((0, 0))
            crop.append(slice(crop_lower, data.shape[i] - crop_upper))
    data = data[tuple(crop)]
    data = np.pad(data, tuple(pad), mode='constant', constant_values=-1.0)
    return data

def process_subject(args):
    subject_dir, out_root, target_shape = args
    patient_id = os.path.basename(subject_dir)
    csv_file = os.path.join(subject_dir, 'timepoints.csv')
    if not os.path.exists(csv_file):
        return

    patient_out = os.path.join(out_root, patient_id)
    ensure_dir(patient_out)

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            days = int(row['days_since_baseline'])
            timepoint = row['timepoint']
            
            # The files are always saved in subject_dir/timepoint_{idx}/timepoint_{idx}_to_template.nii.gz
            warp_file = os.path.join(subject_dir, f"timepoint_{timepoint}", f"timepoint_{timepoint}_to_template.nii.gz")
            
            if not os.path.exists(warp_file):
                logging.warning(f"Could not find warp file {warp_file}")
                continue

            try:
                img = nib.load(warp_file)
                data = img.get_fdata(dtype=np.float32)
                data_norm = normalize(data)
                data_padded = pad_or_crop(data_norm, target_shape)
                
                out_name = f"{patient_id}_time_{days:04d}.npy"
                out_path = os.path.join(patient_out, out_name)
                np.save(out_path, data_padded)
            except Exception as e:
                logging.error(f"Error processing {warp_file}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', '-d', required=True)
    parser.add_argument('--output-root', '-o', required=True)
    parser.add_argument('--target-shape', type=int, nargs=3, default=[256, 256, 256])
    parser.add_argument('--proc', '-p', type=int, default=4)
    args = parser.parse_args()

    subjects = [s for s in sorted(glob.glob(os.path.join(args.data_root, '*'))) if os.path.isdir(s)]
    
    n_proc = min(args.proc, cpu_count())
    pool_args = [(s, args.output_root, tuple(args.target_shape)) for s in subjects]
    
    logging.info(f"Converting {len(subjects)} subjects to NPY arrays with shape {tuple(args.target_shape)}...")
    with Pool(n_proc) as pool:
        list(tqdm(pool.imap_unordered(process_subject, pool_args), total=len(subjects)))

if __name__ == '__main__':
    main()
