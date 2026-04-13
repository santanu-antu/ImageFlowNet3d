import argparse
import glob
import logging
import os
import subprocess
from pathlib import Path


def is_nifti(path: str) -> bool:
    low = path.lower()
    return low.endswith(".nii") or low.endswith(".nii.gz")


def strip_nii_ext(path: str) -> str:
    name = os.path.basename(path)
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def resolve_nifti_from_json(json_path: str) -> str | None:
    base = json_path[:-5]
    candidate_nii_gz = base + ".nii.gz"
    candidate_nii = base + ".nii"
    if os.path.exists(candidate_nii_gz):
        return candidate_nii_gz
    if os.path.exists(candidate_nii):
        return candidate_nii
    return None


def run_hdbet(
    input_nifti: str,
    output_prefix: str,
    device_mode: str = "cpu",
    mode: str = "fast",
    tta: int = 0,
) -> None:
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    if mode not in {"fast", "accurate"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode != "fast":
        logging.warning("Installed HD-BET CLI does not expose mode selection; proceeding with default behavior.")
    cmd = [
        "hd-bet",
        "-i",
        input_nifti,
        "-o",
        output_prefix,
        "-device",
        device_mode,
    ]
    if tta == 0:
        cmd.append("--disable_tta")
    cmd.append("--save_bet_mask")
    logging.info("Running HD-BET: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def pick_preferred_path(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    nii_gz = [p for p in candidates if p.lower().endswith(".nii.gz")]
    if nii_gz:
        return sorted(nii_gz)[0]
    return sorted(candidates)[0]


def find_outputs(output_prefix: str) -> tuple[str | None, str | None]:
    if output_prefix.lower().endswith(".nii.gz"):
        base = output_prefix[:-7]
        stripped_candidates = [base + "_bet.nii.gz", output_prefix]
        mask_candidates = [base + "_bet_mask.nii.gz", base + "_mask.nii.gz"]
        stripped = pick_preferred_path([p for p in stripped_candidates if os.path.exists(p)])
        mask = pick_preferred_path([p for p in mask_candidates if os.path.exists(p)])
        return stripped, mask
    if output_prefix.lower().endswith(".nii"):
        base = output_prefix[:-4]
        stripped_candidates = [base + "_bet.nii.gz", output_prefix]
        mask_candidates = [base + "_bet_mask.nii.gz", base + "_mask.nii.gz"]
        stripped = pick_preferred_path([p for p in stripped_candidates if os.path.exists(p)])
        mask = pick_preferred_path([p for p in mask_candidates if os.path.exists(p)])
        return stripped, mask

    brain_candidates = [
        p
        for p in glob.glob(output_prefix + ".nii*")
        if not p.lower().endswith("_mask.nii") and not p.lower().endswith("_mask.nii.gz")
    ]
    mask_candidates = glob.glob(output_prefix + "_mask.nii*")
    return pick_preferred_path(brain_candidates), pick_preferred_path(mask_candidates)


def plot_comparison(original_path: str, stripped_path: str, output_png: str) -> None:
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np

    orig = nib.load(original_path).get_fdata()
    strip = nib.load(stripped_path).get_fdata()

    ox, oy, oz = [d // 2 for d in orig.shape]
    sx, sy, sz = [d // 2 for d in strip.shape]

    slices_orig = {
        "Axial": orig[:, :, oz],
        "Coronal": orig[:, oy, :],
        "Sagittal": orig[ox, :, :],
    }
    slices_strip = {
        "Axial": strip[:, :, sz],
        "Coronal": strip[:, sy, :],
        "Sagittal": strip[sx, :, :],
    }

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for idx, key in enumerate(["Axial", "Coronal", "Sagittal"]):
        axes[0, idx].imshow(np.rot90(slices_orig[key]), cmap="gray", interpolation="nearest")
        axes[0, idx].set_title(f"Orig {key}")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(np.rot90(slices_strip[key]), cmap="gray", interpolation="nearest")
        axes[1, idx].set_title(f"Stripped {key}")
        axes[1, idx].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    logging.info("Saved comparison image to %s", output_png)


def process_single(
    input_path: str,
    input_root: str,
    skull_root: str,
    figure_root: str,
    device: str,
    mode: str,
    tta: int,
    overwrite: bool,
    plot_figures: bool,
) -> bool:
    if input_path.lower().endswith(".json"):
        resolved = resolve_nifti_from_json(input_path)
        if resolved is None:
            logging.warning("JSON sidecar provided but corresponding NIfTI missing: %s", input_path)
            return False
        input_path = resolved

    if not is_nifti(input_path):
        return False

    input_abs = os.path.abspath(input_path)
    input_root_abs = os.path.abspath(input_root)
    rel_dir = os.path.relpath(os.path.dirname(input_abs), input_root_abs)
    out_dir = os.path.join(skull_root, rel_dir) if rel_dir != "." else skull_root

    stem = strip_nii_ext(input_abs)
    output_prefix = os.path.join(out_dir, stem + ".nii.gz")
    stripped_path, mask_path = find_outputs(output_prefix)

    if not overwrite and stripped_path:
        logging.info("Skipping already processed: %s", input_path)
        return False

    run_hdbet(input_abs, output_prefix, device_mode=device, mode=mode, tta=tta)
    stripped_path, mask_path = find_outputs(output_prefix)
    if stripped_path is None:
        raise RuntimeError(f"HD-BET did not produce a skull-stripped output for: {input_path}")
    if mask_path is None:
        logging.warning("Mask output was not found for %s", input_path)

    if plot_figures:
        figure_dir = os.path.join(figure_root, rel_dir) if rel_dir != "." else figure_root
        figure_path = os.path.join(figure_dir, f"{stem}_comparison.png")
        plot_comparison(input_abs, stripped_path, figure_path)

    return True


def discover_targets(input_dir: str, name_contains: str) -> list[str]:
    base = Path(input_dir)
    keyword = name_contains.upper().strip()
    targets: list[str] = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        p_str = str(p)
        if not is_nifti(p_str):
            continue
        if keyword and keyword not in p_str.upper():
            continue
        targets.append(p_str)
    return sorted(targets)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(
        description="Batch skull-strip ADNI NIfTIs with HD-BET and optional comparison PNGs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", "-i", help="Single NIfTI (.nii/.nii.gz) or .json sidecar file")
    group.add_argument("--input-dir", "-I", help="Root directory to recurse for NIfTI files")
    parser.add_argument(
        "--input-root",
        default=None,
        help="Root used to mirror relative output structure (default: --input-dir, or parent dir of --input)",
    )
    parser.add_argument(
        "--skull-root",
        "-o",
        default=str(project_root / "data" / "processed" / "skull_strip_ADNI"),
        help="Base output root for skull-stripped volumes",
    )
    parser.add_argument(
        "--figure-root",
        "-F",
        default=str(project_root / "data" / "figures" / "ADNI" / "skull_strip"),
        help="Base output root for comparison figures",
    )
    parser.add_argument(
        "--name-contains",
        default="",
        help="Optional uppercase-insensitive substring filter on full input path (e.g., MPRAGE or MPR-R)",
    )
    parser.add_argument("--device", "-d", default="cpu", help="'cpu' or GPU index string (e.g., 0)")
    parser.add_argument("--mode", "-m", choices=["fast", "accurate"], default="fast")
    parser.add_argument("--tta", type=int, choices=[0, 1], default=0, help="Test-time augmentation for HD-BET")
    parser.add_argument("--overwrite", action="store_true", help="Re-run HD-BET even if outputs already exist")
    parser.add_argument("--plot-comparison", action="store_true", help="Save original vs stripped comparison PNG")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Starting skull strip with args=%s", args)

    if args.input_dir:
        input_root = args.input_root or args.input_dir
        targets = discover_targets(args.input_dir, args.name_contains)
    else:
        if args.input is None:
            raise RuntimeError("Either --input or --input-dir is required.")
        input_root = args.input_root or os.path.dirname(os.path.abspath(args.input))
        targets = [args.input]
        if args.name_contains and args.name_contains.upper() not in args.input.upper():
            logging.info("Single input does not match --name-contains filter; nothing to do.")
            return

    logging.info("Discovered %d target files.", len(targets))

    processed = 0
    skipped = 0
    for target in targets:
        did_process = process_single(
            input_path=target,
            input_root=input_root,
            skull_root=args.skull_root,
            figure_root=args.figure_root,
            device=args.device,
            mode=args.mode,
            tta=args.tta,
            overwrite=args.overwrite,
            plot_figures=args.plot_comparison,
        )
        if did_process:
            processed += 1
        else:
            skipped += 1

    logging.info("Finished. Processed=%d, Skipped=%d", processed, skipped)


if __name__ == "__main__":
    main()
