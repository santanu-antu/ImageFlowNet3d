# Nearest Interpolation Locations in ImageFlowNet3d

This document identifies all locations in the repository where "nearest" interpolation is used.

## Summary

There are **11 occurrences** of "nearest" interpolation across **6 files** in the repository:

1. **PyTorch F.interpolate** (2 occurrences)
2. **OpenCV cv2.resize/cv2.warpAffine** (4 occurrences)
3. **ANTs nearestNeighbor interpolator** (2 occurrences)
4. **Custom nearest-neighbor implementation** (1 occurrence)
5. **Documentation/comments** (2 occurrences)

---

## Detailed Locations

### 1. PyTorch F.interpolate with mode="nearest"

#### File: `external_src/I2SB/guided_diffusion/unet.py`

**Line 112:** 3D upsampling in UNet architecture
```python
x = F.interpolate(
    x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
)
```
- **Context:** Used in the `Upsample` class forward method for 3D data
- **Purpose:** Upsamples 3D volumes by a factor of 2 in spatial dimensions

**Line 115:** 2D/default upsampling in UNet architecture
```python
x = F.interpolate(x, scale_factor=2, mode="nearest")
```
- **Context:** Used in the `Upsample` class forward method for 2D data
- **Purpose:** Upsamples 2D images by a scale factor of 2

---

### 2. OpenCV cv2.INTER_NEAREST

#### File: `src/datasets/synthetic3d.py`

**Line 125:** Resizing 3D volumes slice-by-slice
```python
out[z] = cv2.resize(vol[z], (W_t, H_t), interpolation=cv2.INTER_NEAREST)
```
- **Context:** Function `resize_volume_nn_3d` - Nearest-neighbor resize for 3D volumes
- **Purpose:** Resizes in-plane (H, W) dimensions using OpenCV per slice

#### File: `src/preprocessing/01_preprocess_brain_MS.py`

**Line 77:** Resizing mask images
```python
msk = cv2.resize(msk,
                 dsize=tmp_out_shape[::-1],
                 interpolation=cv2.INTER_NEAREST)
```
- **Context:** Preprocessing brain Multiple Sclerosis dataset
- **Purpose:** Resizes segmentation masks to target shape while preserving discrete label values

#### File: `src/preprocessing/01_preprocess_brain_GBM.py`

**Line 212:** Resizing mask images
```python
msk = cv2.resize(msk, dsize=tmp_out_shape[::-1], interpolation=cv2.INTER_NEAREST)
```
- **Context:** Preprocessing brain Glioblastoma dataset (LUMIERE)
- **Purpose:** Resizes tumor masks to target shape while preserving discrete label values

#### File: `src/preprocessing/synthesize_dataset_3d.py`

**Line 103:** Rotating 3D volumes slice-by-slice
```python
out[z] = cv2.warpAffine(vol[z], rot_mat, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
```
- **Context:** Function `rotate_volume_3d` for synthetic dataset generation
- **Purpose:** Applies rotation transformation to each slice of a 3D volume

---

### 3. ANTs nearestNeighbor interpolator

#### File: `src/preprocessing/01_preprocess_brain_GBM.py`

**Line 153:** Applying affine transformation to masks
```python
affine_mask_ants = ants.apply_transforms(fixed=fixed_ants,
                                         moving=moving_mask_ants,
                                         transformlist=reg_affine['fwdtransforms'],
                                         interpolator='nearestNeighbor')
```
- **Context:** Affine registration step for tumor masks
- **Purpose:** Applies affine transformation to binary masks using nearest neighbor to preserve discrete values

**Line 168:** Applying diffeomorphic transformation to masks
```python
diffeo_mask_ants = ants.apply_transforms(fixed=fixed_ants,
                                        moving=affine_mask_ants,
                                        transformlist=reg_diffeomorphic['fwdtransforms'],
                                        interpolator='nearestNeighbor')
```
- **Context:** Diffeomorphic registration step for tumor masks
- **Purpose:** Applies diffeomorphic (SyN) transformation to binary masks using nearest neighbor

---

### 4. Custom Nearest-Neighbor Implementation

#### File: `src/datasets/synthetic3d.py`

**Lines 129-133:** Manual nearest-neighbor depth resizing
```python
# Resize depth with nearest neighbor
if D != D_t:
    z_idx = np.linspace(0, D - 1, num=D_t)
    z_idx_nn = np.round(z_idx).astype(int)
    out = out[z_idx_nn]
```
- **Context:** Function `resize_volume_nn_3d`
- **Purpose:** Resizes the depth (D) dimension by computing nearest-neighbor indices

---

### 5. Documentation/Comments

#### File: `src/datasets/synthetic3d.py`

**Line 117:** Function docstring
```python
"""Nearest-neighbor resize for 3D: z with index mapping, yx with cv2 per-slice."""
```
- **Context:** Docstring for `resize_volume_nn_3d` function
- **Purpose:** Documents that the function uses nearest-neighbor interpolation

**Line 129:** Inline comment
```python
# Resize depth with nearest neighbor
```
- **Context:** Comment describing the depth resizing operation
- **Purpose:** Clarifies the interpolation method used

#### File: `external_src/SuperRetina/loss/triplet_loss.py`

**Line 100:** Documentation comment
```python
``numpy.percentile(..., interpolation="nearest")``.
```
- **Context:** Docstring for `percentile` function
- **Purpose:** Documents that torch.kthvalue() behavior corresponds to numpy's nearest interpolation

---

## Key Observations

### Why Nearest Interpolation is Used

1. **Preserving Discrete Values:** For segmentation masks and binary images, nearest-neighbor interpolation is essential to maintain discrete label values (0, 1, 2, etc.) without introducing intermediate values.

2. **3D Volume Processing:** When processing 3D medical imaging data, nearest-neighbor is used for depth dimension resampling and mask transformations.

3. **Registration Workflows:** In ANTs-based registration pipelines, nearest-neighbor is specifically used for masks while linear interpolation is used for intensity images.

4. **Upsampling in Neural Networks:** The UNet architecture uses nearest-neighbor upsampling as a simple, non-learnable upsampling operation.

### Files Summary

| File | Occurrences | Purpose |
|------|-------------|---------|
| `external_src/I2SB/guided_diffusion/unet.py` | 2 | UNet upsampling layers |
| `src/datasets/synthetic3d.py` | 4 | 3D volume resizing utilities |
| `src/preprocessing/01_preprocess_brain_GBM.py` | 3 | Preprocessing Glioblastoma data |
| `src/preprocessing/01_preprocess_brain_MS.py` | 1 | Preprocessing MS data |
| `src/preprocessing/synthesize_dataset_3d.py` | 1 | Synthetic data generation |
| `external_src/SuperRetina/loss/triplet_loss.py` | 1 | Documentation only |

---

## Technical Details

### Interpolation Methods by Use Case

1. **Masks/Segmentations:** Always use nearest-neighbor (cv2.INTER_NEAREST or ANTs nearestNeighbor)
2. **Intensity Images:** Use linear or cubic interpolation (cv2.INTER_CUBIC or ANTs linear)
3. **Neural Network Upsampling:** Use nearest-neighbor as a simple baseline (can be replaced with learned upsampling)
4. **3D Depth Resampling:** Custom nearest-neighbor using index rounding

### Alternatives to Nearest Interpolation

The codebase also uses:
- `cv2.INTER_CUBIC` for intensity images (lines 74, 211 in preprocessing files)
- `interpolator='linear'` for ANTs intensity image transforms
- Learned convolutions after upsampling in UNet (`self.conv` after interpolation)

---

## Conclusion

Nearest-neighbor interpolation is strategically used throughout the ImageFlowNet3d codebase primarily for:
1. Preserving discrete segmentation mask values during spatial transformations
2. Simple upsampling operations in neural network architectures
3. 3D volume resampling operations where maintaining exact values is important

All occurrences are well-motivated and follow standard practices in medical image processing.
