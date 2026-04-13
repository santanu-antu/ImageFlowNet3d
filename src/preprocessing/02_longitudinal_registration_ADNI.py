import os
# one thread per registration
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import csv
import argparse
import logging
import shutil
from glob import glob
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

import ants
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Define scan type preference
PREFERENCE = [
    "MPRAGE",              # exact standard MPRAGE
    "MPR-R",               # some other strings found in the dataset
    "MPR",
]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def center_image(img):
    """
    Shifts the brain mass (non-zero voxels) to the exact center of the voxel array,
    preserving the original image dimensions and voxel spacing.
    Also shifts the physical origin so that the center of the volume is (0,0,0).
    """
    data = img.numpy()
    coords = np.argwhere(data > 0)
    if len(coords) == 0:
        return img.clone()
        
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    
    cropped_shape = max_coords - min_coords + 1
    target_shape = np.array(img.shape)
    
    total_pad = np.maximum(target_shape - cropped_shape, 0)
    pad_lower = np.floor(total_pad / 2.0).astype(int)
    pad_upper = target_shape - cropped_shape - pad_lower
    
    cropped_data = data[
        min_coords[0]:max_coords[0]+1,
        min_coords[1]:max_coords[1]+1,
        min_coords[2]:max_coords[2]+1
    ]
    
    padded_data = np.pad(
        cropped_data,
        (
            (pad_lower[0], pad_upper[0]),
            (pad_lower[1], pad_upper[1]),
            (pad_lower[2], pad_upper[2])
        ),
        mode='constant',
        constant_values=0
    )
    
    centered_img = img.new_image_like(padded_data)
    
    # Set physical origin so that the exact center of the grid is at (0, 0, 0)
    center_idx = (target_shape - 1) / 2.0
    new_origin = -centered_img.direction.dot(center_idx * np.array(centered_img.spacing))
    centered_img.set_origin(tuple(new_origin))
    
    return centered_img

def build_subject_wise_template(images, iterations=3):
    """
    Builds an unbiased subject-specific template using Iterative Rigid Registration.
    """
    n = len(images)
    if n == 1:
        return images[0]
        
    logging.info(f"Building unbiased template from {n} images...")
    # Initialize the template with the voxelwise average of all centered scans
    template_ref = images[0]
    avg_data = np.zeros(template_ref.shape, dtype=np.float32)
    for img in images:
        if img.shape != template_ref.shape:
            img_resampled = ants.resample_image_to_target(img, template_ref)
            avg_data += img_resampled.numpy() / n
        else:
            avg_data += img.numpy() / n
    template = template_ref.new_image_like(avg_data)
    
    for it in range(iterations):
        logging.info(f"Template construction iteration {it+1}/{iterations}")
        avg_data = np.zeros(template.shape, dtype=np.float32)
        for img in images:
            reg = ants.registration(
                fixed=template,
                moving=img,
                type_of_transform="Rigid"
            )
            avg_data += reg["warpedmovout"].numpy() / n
        template = template.new_image_like(avg_data)
        
    return template

def save_transform_metadata(out_dir, transforms, label, date_str):
    meta = {
        "date": date_str,
        "transforms": transforms
    }
    with open(os.path.join(out_dir, f"{label}_transform_chain.json"), "w") as f:
        json.dump(meta, f, indent=2)

# Removed register_followup_to_baseline as we now use build_subject_wise_template

def collect_representative_per_visit(patient_dir: str):
    patient = Path(patient_dir)
    visits = {}
    for vol in patient.rglob("*.nii*"):
        if vol.is_dir() or str(vol).lower().endswith("_bet.nii.gz") or str(vol).lower().endswith("_mask.nii.gz"):
            # Exclude masks based on user request ('_bet.nii.gz' are masks here)
            continue
        
        # Determine date from parent folder (e.g. 2006-08-30_12_31_31.0)
        date_str = vol.parent.name.split("_")[0]
        visits.setdefault(date_str, []).append(str(vol))

    reps = []
    for date_str, vols in sorted(visits.items()):
        chosen = None
        for pref in PREFERENCE:
            for v in vols:
                if pref.upper() in v.upper():
                    chosen = v
                    break
            if chosen:
                break
        if not chosen:
            chosen = vols[0]
        reps.append((date_str, chosen))
    return reps

def process_subject_wrapper(args_tuple):
    try:
        process_subject(args_tuple)
    except Exception as e:
        logging.error(f"Failed to process subject {args_tuple[0]}: {e}")

def process_subject(args_tuple):
    """
    Run the full registration pipeline for ONE subject.
    args_tuple = (subject_path, output_root, iterations)
    """
    subject_dir, output_root, iterations = args_tuple
    patient_id = os.path.basename(subject_dir)
    logging.info(f"Start  {patient_id}")

    reps = collect_representative_per_visit(subject_dir)
    if not reps:
        logging.warning(f"No valid volumes found for {patient_id}")
        return

    dates, scans = zip(*reps)
    dt_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    days = [(dt - dt_objs[0]).days for dt in dt_objs]

    patient_out = os.path.join(output_root, patient_id)
    ensure_dir(patient_out)

    csv_path = os.path.join(patient_out, "timepoints.csv")
    with open(csv_path, "w", newline="") as csvf:
        w = csv.writer(csvf)
        w.writerow(["timepoint", "date", "days_since_baseline", "warp_file"])
        for idx, date in enumerate(dates):
            warp = os.path.join(patient_out, f"timepoint_{idx}", f"timepoint_{idx}_to_template.nii.gz")
            w.writerow([idx, date, days[idx], warp])

    # 1. Read and center all images
    centered_scans = []
    for scan_path in scans:
        img = ants.image_read(scan_path)
        centered_scans.append(center_image(img))

    # 2. Build unbiased template
    template_path = os.path.join(patient_out, "subject_template.nii.gz")
    if not os.path.exists(template_path):
        subject_template = build_subject_wise_template(centered_scans, iterations=iterations)
        ants.image_write(subject_template, template_path)
        logging.info(f"Saved unbiased subject template to {template_path}")
    else:
        subject_template = ants.image_read(template_path)

    # 3. Register all centered scans to the unbiased template
    for idx, (original_scan_path, c_img, date) in enumerate(zip(scans, centered_scans, dates)):
        out_dir = os.path.join(patient_out, f"timepoint_{idx}")
        ensure_dir(out_dir)
        warp_file = os.path.join(out_dir, f"timepoint_{idx}_to_template.nii.gz")
        
        if not os.path.exists(warp_file):
            logging.info(f"Registering timepoint {idx} for {patient_id} to unbiased template.")
            reg = ants.registration(
                fixed=subject_template,
                moving=c_img,
                type_of_transform="Rigid"
            )
            ants.image_write(reg["warpedmovout"], warp_file)
            
            save_transform_metadata(
                out_dir,
                transforms=reg["fwdtransforms"],
                label=f"timepoint_{idx}_to_template",
                date_str=date
            )
            logging.info(f"Wrote timepoint {idx} warped image to {warp_file}")

    logging.info(f"Done   {patient_id}")

def main():
    parser = argparse.ArgumentParser(description="Multiprocess longitudinal registration to baseline")
    parser.add_argument("--data-root", "-d", required=True, help="Root dir with subject subfolders")
    # Template argument removed since baseline is now the template
    parser.add_argument("--output-root", "-o", default="results/longitudinal_ADNI", help="Where to write outputs")
    parser.add_argument("--proc", "-p", type=int, default=1, help="Number of worker processes (default 1)")
    parser.add_argument("--iters", type=int, default=3, help="Number of template building iterations (default 3)")
    args = parser.parse_args()

    subjects = [s for s in sorted(glob(os.path.join(args.data_root, "*"))) if os.path.isdir(s)]
    if not subjects:
        raise RuntimeError("No subject folders found under --data-root")

    n_proc = max(1, min(args.proc, cpu_count()))
    logging.info(f"Spawning pool with {n_proc} worker(s) for {len(subjects)} subjects")

    pool_args = [(s, args.output_root, args.iters) for s in subjects]
    with Pool(processes=n_proc) as pool:
        list(tqdm(pool.imap_unordered(process_subject_wrapper, pool_args), total=len(subjects)))

    logging.info("All subjects finished")

if __name__ == "__main__":
    main()
