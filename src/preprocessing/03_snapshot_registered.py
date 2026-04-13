import os
import glob
import argparse
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_nifti(args):
    nii_path, out_path = args
    if os.path.exists(out_path):
        return
        
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Take the middle slice in all 3 dimensions
        cx, cy, cz = [s // 2 for s in data.shape]
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # Axial
        axes[0].imshow(np.rot90(data[:, :, cz]), cmap='gray')
        axes[0].set_title('Axial')
        axes[0].axis('off')
        
        # Coronal
        axes[1].imshow(np.rot90(data[:, cy, :]), cmap='gray')
        axes[1].set_title('Coronal')
        axes[1].axis('off')
        
        # Sagittal
        axes[2].imshow(np.rot90(data[cx, :, :]), cmap='gray')
        axes[2].set_title('Sagittal')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Failed to process {nii_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Take 2D snapshots of 3D NIfTI volumes for visual QA.")
    parser.add_argument("--data-root", required=True, help="Root directory containing the NIfTI files")
    parser.add_argument("--output-root", required=True, help="Root directory to save the PNG snapshots")
    parser.add_argument("--proc", type=int, default=4, help="Number of worker processes")
    args = parser.parse_args()

    # Find all .nii.gz files recursively
    nii_files = glob.glob(os.path.join(args.data_root, "**", "*.nii.gz"), recursive=True)
    
    tasks = []
    for nii_path in nii_files:
        rel_path = os.path.relpath(nii_path, args.data_root)
        path_parts = Path(rel_path).parts
        
        if len(path_parts) >= 3:
            patient_id = path_parts[0]
            timepoint = path_parts[-2]
            filename = Path(path_parts[-1]).with_suffix('').with_suffix('')
            out_name = f"{timepoint}_{filename}.png"
            out_path = os.path.join(args.output_root, patient_id, out_name)
        else:
            stem = str(Path(rel_path).with_suffix('').with_suffix(''))
            out_path = os.path.join(args.output_root, f"{stem}.png")
            
        tasks.append((nii_path, out_path))
        
    print(f"Found {len(tasks)} files to process...")
    
    with Pool(min(args.proc, cpu_count())) as pool:
        list(tqdm(pool.imap_unordered(process_nifti, tasks), total=len(tasks)))
        
if __name__ == "__main__":
    main()
