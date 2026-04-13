"""
Training script for 3D ImageFlowNetODE on volumetric data.

This is analogous to train_2pt_all.py but designed for true 3D volumetric data
(e.g., 64x64x64 volumes) rather than 2D images.
"""
import argparse
import ast
import os
import sys
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Dataset
from tqdm import tqdm
import monai.transforms as mt

from data_utils.prepare_dataset import prepare_dataset
from nn.scheduler import LinearWarmupCosineAnnealingLR
from utils.attribute_hashmap import AttributeHashmap
from utils.log_util import log
from utils.metrics import psnr, ssim
from utils.parse import parse_settings
from utils.seed import seed_everything
from utils.patch_utils import (
    extract_corresponding_patches_3d,
    sliding_window_inference_3d,
    get_num_patches,
    check_patchify_compatibility
)

# Import 3D model
from nn.imageflownet_3d_ode import ImageFlowNet3DODE


def add_random_noise(vol: torch.Tensor, max_intensity: float = 0.1) -> torch.Tensor:
    """Add random noise to volume for regularization."""
    intensity = max_intensity * torch.rand(1).to(vol.device)
    noise = intensity * torch.randn_like(vol)
    return vol + noise


def neg_cos_sim(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity for SimSiam-style contrastive learning."""
    z = z.detach()  # stop gradient
    p = torch.nn.functional.normalize(p, p=2, dim=1)
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    return -(p * z).sum(dim=1).mean()


def train(config: AttributeHashmap):
    device = torch.device(f"cuda:{int(config.gpu_id)}" if torch.cuda.is_available() else "cpu")

    # For 3D data, we don't use albumentations (2D only)
    # Instead, we pass None transforms and let the dataset handle normalization
    transforms_list = [None, None, None]

    train_set, val_set, test_set, num_image_channel, max_t = \
        prepare_dataset(config=config, transforms_list=transforms_list)

    log('Using device: %s' % device, to_console=True)
    log('Dataset loaded: %d train, %d val, %d test subjects' % (
        len(train_set.dataset), len(val_set.dataset), len(test_set.dataset)), to_console=True)
    log('Max timestamp in dataset: %.2f' % max_t, to_console=True)

    # Log train/val/test split subject IDs if available
    if hasattr(train_set.dataset.dataset, 'subject_ids') and len(train_set.dataset.dataset.subject_ids) > 0:
        log('Train subjects (%d): %s' % (len(train_set.dataset.dataset.subject_ids), train_set.dataset.dataset.subject_ids), to_console=True)
    if hasattr(val_set.dataset, 'subject_ids') and len(val_set.dataset.subject_ids) > 0:
        log('Val subjects (%d): %s' % (len(val_set.dataset.subject_ids), val_set.dataset.subject_ids), to_console=True)
    if hasattr(test_set.dataset, 'subject_ids') and len(test_set.dataset.subject_ids) > 0:
        log('Test subjects (%d): %s' % (len(test_set.dataset.subject_ids), test_set.dataset.subject_ids), to_console=True)

    # Build the 3D model
    model = ImageFlowNet3DODE(
        device=device,
        volume_size=config.volume_size,
        num_filters=config.num_filters,
        in_channels=num_image_channel,
        out_channels=num_image_channel,
        ode_location=config.ode_location,
        attention_resolutions=config.attention_resolutions,
        contrastive=config.coeff_contrastive + config.coeff_invariance > 0,
        use_checkpoint=config.use_checkpoint,
    )

    log('Model: ImageFlowNet3DODE', to_console=True)
    log('Volume size: %d' % config.volume_size, to_console=True)
    log('ODE location: %s' % config.ode_location, to_console=True)
    log('Attention resolutions: %s' % config.attention_resolutions, to_console=True)
    log('Use checkpoint: %s' % config.use_checkpoint, to_console=True)
    log('Number of parameters: %d' % sum(p.numel() for p in model.parameters()), to_console=True)
    
    # Patchified training configuration
    use_patches = config.num_patches_per_dim > 1
    if use_patches:
        full_volume_size = config.target_dim[0]  # Assuming cubic
        patch_size = config.volume_size
        is_compatible, msg = check_patchify_compatibility(
            full_volume_size, patch_size, config.num_patches_per_dim)
        log('Patchified training: %s' % msg, to_console=True)
        log('Training on patches of size %d from volumes of size %d' % (
            patch_size, full_volume_size), to_console=True)
    else:
        log('Patchified training: Disabled (full volume training)', to_console=True)

    model.to(device)
    model.init_params()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.max_epochs // 10,
        max_epochs=config.max_epochs
    )

    # Initialize training state
    start_epoch = 0
    best_val_psnr = 0
    recon_psnr_thr, recon_good_enough = 20, False  # Slightly lower threshold for 3D

    # Resume from checkpoint if requested
    if config.resume or config.resume_from is not None:
        checkpoint_path = config.resume_from if config.resume_from else \
            config.model_save_path.replace('.pty', '_latest.pty')
        
        if os.path.exists(checkpoint_path):
            log('Resuming from checkpoint: %s' % checkpoint_path, to_console=True)
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_psnr = checkpoint.get('best_val_psnr', 0)
            recon_good_enough = checkpoint.get('recon_good_enough', False)
            
            log('Resumed from epoch %d, best_val_psnr: %.3f, recon_good_enough: %s' % (
                start_epoch, best_val_psnr, recon_good_enough), to_console=True)
        else:
            log('No checkpoint found at %s, starting from scratch' % checkpoint_path, to_console=True)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
    ema.to(device)

    mse_loss = torch.nn.MSELoss()
    backprop_freq = config.batch_size

    os.makedirs(config.save_folder + 'train/', exist_ok=True)
    os.makedirs(config.save_folder + 'val/', exist_ok=True)

    # Only relevant to ODE
    config.t_multiplier = config.ode_max_t / max_t

    for epoch_idx in tqdm(range(start_epoch, config.max_epochs)):
        model, ema, optimizer, scheduler = train_epoch(
            config=config,
            device=device,
            train_set=train_set,
            model=model,
            epoch_idx=epoch_idx,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            mse_loss=mse_loss,
            backprop_freq=backprop_freq,
            train_time_dependent=recon_good_enough
        )

        with ema.average_parameters():
            model.eval()
            val_recon_psnr, val_pred_psnr = val_epoch(
                config=config,
                device=device,
                val_set=val_set,
                model=model,
                epoch_idx=epoch_idx
            )

        if val_recon_psnr > recon_psnr_thr:
            recon_good_enough = True

        # Save latest checkpoint (for resumption)
        latest_checkpoint_path = config.model_save_path.replace('.pty', '_latest.pty')
        torch.save({
            'epoch': epoch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_psnr': best_val_psnr,
            'recon_good_enough': recon_good_enough,
        }, latest_checkpoint_path)

        if val_pred_psnr > best_val_psnr:
            best_val_psnr = val_pred_psnr
            model.save_weights(config.model_save_path.replace('.pty', '_best_pred_psnr.pty'))
            log('%s: Model weights successfully saved for best pred PSNR (%.3f).' % (
                'ImageFlowNet3DODE', val_pred_psnr),
                filepath=config.log_dir,
                to_console=False)

    return


def train_epoch(config: AttributeHashmap,
                device: torch.device,
                train_set: Dataset,
                model: torch.nn.Module,
                epoch_idx: int,
                ema: ExponentialMovingAverage,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                mse_loss: torch.nn.Module,
                backprop_freq: int,
                train_time_dependent: bool):
    """Training epoch for 3D ImageFlowNetODE with optional patchified training."""

    train_loss_recon, train_loss_pred = 0, 0
    train_recon_psnr, train_pred_psnr = 0, 0
    model.train()
    optimizer.zero_grad()

    # Check if using patchified training
    use_patches = config.num_patches_per_dim > 1
    patch_size = config.volume_size

    if not train_time_dependent:
        log('[Epoch %d] Will not train the time-dependent modules until the reconstruction is good enough.' % (epoch_idx + 1),
            filepath=config.log_dir,
            to_console=False)

    assert len(train_set) == len(train_set.dataset)
    num_train_samples = min(config.max_training_samples, len(train_set))
    plot_freq = max(1, num_train_samples // config.n_plot_per_epoch)

    for iter_idx, (volumes, timestamps) in enumerate(tqdm(train_set, desc=f'Epoch {epoch_idx+1} Train')):

        if iter_idx > config.max_training_samples:
            break

        shall_plot = iter_idx % plot_freq == 0

        # volumes: [1, 2, C, D, H, W], containing [vol_start, vol_end]
        # timestamps: [1, 2], containing [t_start, t_end]
        assert volumes.shape[1] == 2
        assert timestamps.shape[1] == 2

        x_list, t_list = convert_variables_3d(volumes, timestamps, device)
        x_start_full, x_end_full = x_list

        # Extract patches if using patchified training
        if use_patches:
            # Extract corresponding random patches from both volumes
            x_start_patch, x_end_patch, _ = extract_corresponding_patches_3d(
                x_start_full, x_end_full, patch_size)
            x_start = x_start_patch.unsqueeze(0).to(device)  # Add batch dim
            x_end = x_end_patch.unsqueeze(0).to(device)
        else:
            x_start = x_start_full
            x_end = x_end_full

        x_start_noisy = add_random_noise(x_start)
        x_end_noisy = add_random_noise(x_end)

        ################### Recon Loss to update Encoder/Decoder ##################
        model.unfreeze()

        x_start_recon = model(x=x_start_noisy, t=torch.zeros(1).to(device))
        x_end_recon = model(x=x_end_noisy, t=torch.zeros(1).to(device))

        contrastive_loss = 0
        if config.coeff_contrastive > 0:
            z1, z2 = model.simsiam_project(x_start), model.simsiam_project(x_end)
            p1, p2 = model.simsiam_predict(z1), model.simsiam_predict(z2)
            contrastive_loss = neg_cos_sim(p1, z2) / 2 + neg_cos_sim(p2, z1) / 2

        loss_recon = mse_loss(x_start, x_start_recon) + mse_loss(x_end, x_end_recon) \
            + config.coeff_contrastive * contrastive_loss

        train_loss_recon += loss_recon.item()

        # Simulate `config.batch_size` by batched optimizer update.
        loss_recon = loss_recon / backprop_freq
        loss_recon.backward()

        ################## Pred Loss to update time-dependent modules #############
        try:
            model.freeze_time_independent()
        except AttributeError:
            pass

        if train_time_dependent:
            assert torch.diff(t_list).item() > 0

            smoothness_loss = 0
            if config.coeff_smoothness > 0:
                x_end_pred, smoothness_loss = model(
                    x=x_start_noisy,
                    t=torch.diff(t_list) * config.t_multiplier,
                    return_grad=True
                )
            else:
                x_end_pred = model(x=x_start_noisy, t=torch.diff(t_list) * config.t_multiplier)

            loss_pred = mse_loss(x_end, x_end_pred) + config.coeff_smoothness * smoothness_loss
            train_loss_pred += loss_pred.item()

            loss_pred = loss_pred / backprop_freq
            loss_pred.backward()

        else:
            with torch.no_grad():
                x_end_pred = model(x=x_start_noisy, t=torch.diff(t_list) * config.t_multiplier)
                loss_pred = mse_loss(x_end, x_end_pred)
                train_loss_pred += loss_pred.item()

        # Batched optimizer update
        if iter_idx % config.batch_size == config.batch_size - 1:
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

        # Compute metrics
        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            numpy_variables_3d(x_start, x_start_recon, x_end, x_end_recon, x_end_pred)

        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            cast_to_0to1(x0_true, x0_recon, xT_true, xT_recon, xT_pred)

        train_recon_psnr += psnr_3d(x0_true, x0_recon) / 2 + psnr_3d(xT_true, xT_recon) / 2
        train_pred_psnr += psnr_3d(xT_true, xT_pred)

        if shall_plot:
            save_path_fig = '%s/train/figure_log_epoch%s_sample%s.png' % (
                config.save_folder, str(epoch_idx + 1).zfill(5), str(iter_idx + 1).zfill(5))
            
            # Helper to get first item for plotting (handle batch > 1)
            def _get_first(arr):
                while arr.ndim > 3:
                    arr = arr[0]
                return arr
                
            plot_volume_slices(t_list, 
                               _get_first(x0_true), 
                               _get_first(xT_true), 
                               _get_first(x0_recon), 
                               _get_first(xT_recon), 
                               _get_first(xT_pred), 
                               save_path=save_path_fig,
                               slice_idx=140 if not use_patches else None)

    train_loss_pred, train_loss_recon, train_recon_psnr, train_pred_psnr = \
        [item / num_train_samples for item in (train_loss_pred, train_loss_recon, train_recon_psnr, train_pred_psnr)]

    scheduler.step()

    log('Train [%s/%s] loss [recon: %.3f, pred: %.3f], PSNR (recon): %.3f, PSNR (pred): %.3f'
        % (epoch_idx + 1, config.max_epochs, train_loss_recon, train_loss_pred, train_recon_psnr, train_pred_psnr),
        filepath=config.log_dir,
        to_console=False)

    return model, ema, optimizer, scheduler


@torch.no_grad()
def val_epoch(config: AttributeHashmap,
              device: torch.device,
              val_set: Dataset,
              model: torch.nn.Module,
              epoch_idx: int):
    """Validation epoch for 3D ImageFlowNetODE with optional patchified inference."""

    val_recon_psnr, val_pred_psnr = 0, 0

    # Check if using patchified training
    use_patches = config.num_patches_per_dim > 1
    patch_size = config.volume_size

    assert len(val_set) == len(val_set.dataset)
    num_val_samples = min(config.max_validation_samples, len(val_set))
    plot_freq = max(1, num_val_samples // config.n_plot_per_epoch)

    for iter_idx, (volumes, timestamps) in enumerate(tqdm(val_set, desc=f'Epoch {epoch_idx+1} Val')):
        shall_plot = iter_idx % plot_freq == 0

        if iter_idx > config.max_validation_samples:
            break

        assert volumes.shape[1] == 2
        assert timestamps.shape[1] == 2

        x_list, t_list = convert_variables_3d(volumes, timestamps, device)
        x_start, x_end = x_list

        if use_patches:
            # Use sliding window inference for validation on full volumes
            x_start_recon = sliding_window_inference_3d(
                x_start, model, patch_size,
                stride=patch_size,  # Non-overlapping for speed
                t=torch.zeros(1).to(device),
                overlap_mode='average',
                device=device
            ).to(device)
            x_end_recon = sliding_window_inference_3d(
                x_end, model, patch_size,
                stride=patch_size,
                t=torch.zeros(1).to(device),
                overlap_mode='average',
                device=device
            ).to(device)
            x_end_pred = sliding_window_inference_3d(
                x_start, model, patch_size,
                stride=patch_size,
                t=torch.diff(t_list) * config.t_multiplier,
                overlap_mode='average',
                device=device
            ).to(device)
        else:
            x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
            x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))
            x_end_pred = model(x=x_start, t=torch.diff(t_list) * config.t_multiplier)

        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            numpy_variables_3d(x_start, x_start_recon, x_end, x_end_recon, x_end_pred)

        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            cast_to_0to1(x0_true, x0_recon, xT_true, xT_recon, xT_pred)

        val_recon_psnr += psnr_3d(x0_true, x0_recon) / 2 + psnr_3d(xT_true, xT_recon) / 2
        val_pred_psnr += psnr_3d(xT_true, xT_pred)

        if shall_plot:
            save_path_fig = '%s/val/figure_log_epoch%s_sample%s.png' % (
                config.save_folder, str(epoch_idx + 1).zfill(5), str(iter_idx + 1).zfill(5))
            plot_volume_slices(t_list, x0_true, xT_true, x0_recon, xT_recon, xT_pred, save_path_fig, save_npy=False, slice_idx=140)

    val_recon_psnr, val_pred_psnr = \
        [item / num_val_samples for item in (val_recon_psnr, val_pred_psnr)]

    log('Validation [%s/%s] PSNR (recon): %.3f, PSNR (pred): %.3f'
        % (epoch_idx + 1, config.max_epochs, val_recon_psnr, val_pred_psnr),
        filepath=config.log_dir,
        to_console=False)

    return val_recon_psnr, val_pred_psnr


@torch.no_grad()
def test(config: AttributeHashmap):
    """Test the trained 3D model with optional patchified inference."""
    device = torch.device(f"cuda:{int(config.gpu_id)}" if torch.cuda.is_available() else "cpu")
    train_set, val_set, test_set, num_image_channel, max_t = \
        prepare_dataset(config=config)

    model = ImageFlowNet3DODE(
        device=device,
        volume_size=config.volume_size,
        num_filters=config.num_filters,
        in_channels=num_image_channel,
        out_channels=num_image_channel,
        ode_location=config.ode_location,
        attention_resolutions=config.attention_resolutions,
        contrastive=config.coeff_contrastive + config.coeff_invariance > 0,
        use_checkpoint=config.use_checkpoint,
    )

    model.to(device)
    
    # Use custom checkpoint path if provided, otherwise use default
    if config.checkpoint_path is not None:
        checkpoint_path = config.checkpoint_path
    else:
        checkpoint_path = config.model_save_path.replace('.pty', '_best_pred_psnr.pty')
    
    model.load_weights(checkpoint_path, device=device)
    model.eval()

    log('%s: Model weights successfully loaded from %s' % ('ImageFlowNet3DODE', checkpoint_path), to_console=True)

    config.t_multiplier = config.ode_max_t / max_t
    mse_loss = torch.nn.MSELoss()

    # Check if using patchified inference
    use_patches = config.num_patches_per_dim > 1
    patch_size = config.volume_size
    # For test, use overlapping patches with Gaussian blending for best quality
    test_stride = patch_size // 2 if use_patches else patch_size

    if use_patches:
        log('Using patchified inference with patch_size=%d, stride=%d, overlap_mode=gaussian' %
            (patch_size, test_stride), to_console=True)

    assert len(test_set) == len(test_set.dataset)
    num_test_samples = min(config.max_testing_samples, len(test_set))

    save_path_fig_summary = '%s/results/summary.png' % config.save_folder
    os.makedirs(os.path.dirname(save_path_fig_summary), exist_ok=True)

    deltaT_list, psnr_list = [], []
    test_loss = 0
    test_pred_psnr, test_pred_mse = [], []

    for iter_idx, (volumes, timestamps) in enumerate(tqdm(test_set, desc='Testing')):
        if iter_idx > config.max_testing_samples:
            break

        assert volumes.shape[1] == 2
        assert timestamps.shape[1] == 2

        x_list, t_list = convert_variables_3d(volumes, timestamps, device)
        x_start, x_end = x_list

        if use_patches:
            # Use sliding window inference with overlapping patches
            x_start_recon = sliding_window_inference_3d(
                x_start, model, patch_size,
                stride=test_stride,
                t=torch.zeros(1).to(device),
                overlap_mode='gaussian',
                device=device
            ).to(device)
            x_end_recon = sliding_window_inference_3d(
                x_end, model, patch_size,
                stride=test_stride,
                t=torch.zeros(1).to(device),
                overlap_mode='gaussian',
                device=device
            ).to(device)
            x_end_pred = sliding_window_inference_3d(
                x_start, model, patch_size,
                stride=test_stride,
                t=torch.diff(t_list) * config.t_multiplier,
                overlap_mode='gaussian',
                device=device
            ).to(device)
        else:
            x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
            x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))
            x_end_pred = model(x=x_start, t=torch.diff(t_list) * config.t_multiplier)

        loss_recon = mse_loss(x_start, x_start_recon) + mse_loss(x_end, x_end_recon)
        loss_pred = mse_loss(x_end, x_end_pred)

        test_loss += (loss_recon + loss_pred).item()

        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            numpy_variables_3d(x_start, x_start_recon, x_end, x_end_recon, x_end_pred)

        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            cast_to_0to1(x0_true, x0_recon, xT_true, xT_recon, xT_pred)

        test_pred_psnr.append(psnr_3d(xT_true, xT_pred))
        test_pred_mse.append(np.mean((xT_pred - xT_true) ** 2))

        deltaT_list.append((t_list[1] - t_list[0]).item())
        psnr_list.append(psnr_3d(xT_true, xT_pred))

        # Save visualization and numpy volumes
        save_path_fig = '%s/results/figure_%s.png' % (config.save_folder, str(iter_idx + 1).zfill(5))
        plot_volume_slices(t_list, x0_true, xT_true, x0_recon, xT_recon, xT_pred, save_path_fig, save_npy=True)

    # Plot summary
    fig_summary = plt.figure(figsize=(12, 5))
    ax = fig_summary.add_subplot(1, 1, 1)
    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.scatter(deltaT_list, psnr_list, color='black', s=50, alpha=0.5)
    ax.set_xlabel('Time difference', fontsize=20)
    ax.set_ylabel('PSNR', fontsize=20)
    fig_summary.tight_layout()
    fig_summary.savefig(save_path_fig_summary)
    plt.close(fig=fig_summary)

    test_loss /= num_test_samples
    test_pred_psnr = np.array(test_pred_psnr)
    test_pred_mse = np.array(test_pred_mse)

    log('Test loss: %.3f, PSNR (pred): %.3f ± %.3f, MSE (pred): %.5f ± %.5f' % (
        test_loss,
        np.mean(test_pred_psnr), np.std(test_pred_psnr) / np.sqrt(len(test_pred_psnr)),
        np.mean(test_pred_mse), np.std(test_pred_mse) / np.sqrt(len(test_pred_mse))),
        filepath=config.log_dir,
        to_console=True)

    return


def convert_variables_3d(volumes: torch.Tensor,
                         timestamps: torch.Tensor,
                         device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert volume tensors for 3D processing."""
    # volumes: [batch, 2, C, D, H, W]
    x_start = volumes[:, 0, ...].float().to(device)
    x_end = volumes[:, 1, ...].float().to(device)
    t_list = timestamps[0].float().to(device)
    return [x_start, x_end], t_list


def numpy_variables_3d(*tensors: torch.Tensor) -> Tuple[np.ndarray]:
    """Convert 3D volume tensors to numpy arrays."""
    # Input: [batch, C, D, H, W] -> Output: [D, H, W] (squeeze batch and channel)
    return [_tensor.cpu().detach().numpy().squeeze() for _tensor in tensors]


def cast_to_0to1(*np_arrays: np.ndarray) -> Tuple[np.ndarray]:
    """Cast volumes to [0, 1] range from [-1, 1]."""
    return [np.clip((_arr + 1) / 2, 0, 1) for _arr in np_arrays]


def psnr_3d(vol1: np.ndarray, vol2: np.ndarray) -> float:
    """Compute PSNR for 3D volumes."""
    mse = np.mean((vol1 - vol2) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def plot_volume_slices(t_list: torch.Tensor,
                       x0_true: np.ndarray,
                       xT_true: np.ndarray,
                       x0_recon: np.ndarray,
                       xT_recon: np.ndarray,
                       xT_pred: np.ndarray,
                       save_path: str,
                       save_npy: bool = False,
                       slice_idx: int = None):
    """
    Plot axial slices of 3D volumes for visualization.
    Uses same format as 2D case with side-by-side comparison.
    
    Optionally saves the full 3D volumes as .npy files.
    """
    D, H, W = x0_true.shape
    # Plotting Sagittal view (slice along Width, dim 2) instead of Axial
    if slice_idx is None:
        slice_idx = W // 2
    
    # Clip slice_idx if out of bounds (e.g. if patches are small)
    if slice_idx >= W:
        print(f"Slice {slice_idx} out of bounds for width {W}, using center {W//2}")
        slice_idx = W // 2
    
    # Extract slices
    x0_true_slice = x0_true[:, :, slice_idx]
    xT_true_slice = xT_true[:, :, slice_idx]
    x0_recon_slice = x0_recon[:, :, slice_idx]
    xT_recon_slice = xT_recon[:, :, slice_idx]
    xT_pred_slice = xT_pred[:, :, slice_idx]
    
    # Create figure similar to 2D case
    fig_sbs = plt.figure(figsize=(20, 8))
    plt.rcParams['font.family'] = 'serif'
    
    aspect_ratio = D / H

    # First column: Ground Truth
    ax = fig_sbs.add_subplot(2, 5, 1)
    ax.imshow(x0_true_slice, cmap='gray', vmin=0, vmax=1)
    ax.set_title('GT(t=0), time: %.1f\n[vs GT(t=T)]: PSNR=%.2f' % (
        t_list[0].item(), psnr_3d(x0_true, xT_true)), fontsize=10)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    
    ax = fig_sbs.add_subplot(2, 5, 6)
    ax.imshow(xT_true_slice, cmap='gray', vmin=0, vmax=1)
    ax.set_title('GT(t=T), time: %.1f' % t_list[1].item(), fontsize=10)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Second column: Reconstruction
    ax = fig_sbs.add_subplot(2, 5, 2)
    ax.imshow(x0_recon_slice, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Recon(t=0)\nPSNR=%.2f' % psnr_3d(x0_true, x0_recon), fontsize=10)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    
    ax = fig_sbs.add_subplot(2, 5, 7)
    ax.imshow(xT_recon_slice, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Recon(t=T)\nPSNR=%.2f' % psnr_3d(xT_true, xT_recon), fontsize=10)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Third column: Prediction
    ax = fig_sbs.add_subplot(2, 5, 3)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    
    ax = fig_sbs.add_subplot(2, 5, 8)
    ax.imshow(xT_pred_slice, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Pred(t=T), time: %.1f -> %.1f\n[vs GT(t=T)]: PSNR=%.2f' % (
        t_list[0].item(), t_list[1].item(), psnr_3d(xT_true, xT_pred)), fontsize=10)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Fourth column: |Ground Truth t0 - Ground Truth tT|, |Ground Truth - Prediction|
    ax = fig_sbs.add_subplot(2, 5, 4)
    ax.imshow(np.abs(x0_true_slice - xT_true_slice), cmap='gray', vmin=0, vmax=1)
    ax.set_title('|GT(t=0) - GT(t=T)|\n[MAE=%.4f]' % np.mean(np.abs(x0_true - xT_true)), fontsize=10)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    
    ax = fig_sbs.add_subplot(2, 5, 9)
    ax.imshow(np.abs(xT_pred_slice - xT_true_slice), cmap='gray', vmin=0, vmax=1)
    ax.set_title('|Pred(t=T) - GT(t=T)|\n[MAE=%.4f]' % np.mean(np.abs(xT_pred - xT_true)), fontsize=10)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Fifth column: Summary info
    ax = fig_sbs.add_subplot(2, 5, 5)
    ax.text(0.5, 0.5, f'Slice: {slice_idx}/{D}\n(Axial view)', 
            ha='center', va='center', fontsize=12)
    ax.set_axis_off()
    
    ax = fig_sbs.add_subplot(2, 5, 10)
    ax.text(0.5, 0.5, f'Volume: {D}x{H}x{W}', 
            ha='center', va='center', fontsize=12)
    ax.set_axis_off()

    fig_sbs.tight_layout()
    fig_sbs.savefig(save_path, dpi=100)
    plt.close(fig=fig_sbs)
    
    # Save numpy volumes if requested
    if save_npy:
        npy_dir = os.path.dirname(save_path)
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        
        np.save(os.path.join(npy_dir, f'{base_name}_x0_true.npy'), x0_true)
        np.save(os.path.join(npy_dir, f'{base_name}_xT_true.npy'), xT_true)
        np.save(os.path.join(npy_dir, f'{base_name}_x0_recon.npy'), x0_recon)
        np.save(os.path.join(npy_dir, f'{base_name}_xT_recon.npy'), xT_recon)
        np.save(os.path.join(npy_dir, f'{base_name}_xT_pred.npy'), xT_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3D ImageFlowNetODE.')
    parser.add_argument('--mode', help='`train` or `test`?', default='train')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0, type=int)
    parser.add_argument('--run-count', default=None, type=int)
    parser.add_argument('--checkpoint-path', default=None, type=str,
                        help='Path to checkpoint file for testing (overrides default path)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--resume-from', default=None, type=str,
                        help='Path to checkpoint to resume from (overrides auto-detection)')

    # Dataset settings
    parser.add_argument('--dataset-name', default='synthetic3d', type=str)
    parser.add_argument('--target-dim', default='(64, 64, 64)', type=ast.literal_eval)
    parser.add_argument('--dataset-path', default='$ROOT/data/synthesized3d/', type=str)
    parser.add_argument('--image-folder', default='base', type=str)
    parser.add_argument('--output-save-folder', default='$ROOT/results/', type=str)
    parser.add_argument('--segmentor-ckpt', default='', type=str)  # Not used for 3D synthetic

    # Model settings
    parser.add_argument('--model', default='ImageFlowNet3DODE', type=str)
    parser.add_argument('--volume-size', default=64, type=int,
                        help='Size of volume patches for training. When num-patches-per-dim=1, this equals target-dim.')
    parser.add_argument('--num-patches-per-dim', default=1, type=int,
                        help='Number of patches per dimension. 1=full volume (unpatchified), 2=8 patches from 2x2x2 grid, etc.')
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--max-epochs', default=120, type=int)
    parser.add_argument('--batch-size', default=4, type=int)  # Smaller batch for 3D
    parser.add_argument('--ode-max-t', default=5.0, type=float)
    parser.add_argument('--ode-location', default='all_connections', type=str)
    parser.add_argument('--attention-resolutions', default='8', type=str)  # Memory efficient
    parser.add_argument('--use-checkpoint', action='store_true', default=False,
                        help='Use gradient checkpointing to reduce memory (slower training)')
    parser.add_argument('--num-filters', default=64, type=int)
    parser.add_argument('--depth', default=5, type=int)  # Not used, keeping for compatibility
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--max-training-samples', default=512, type=int)
    parser.add_argument('--max-validation-samples', default=64, type=int)
    parser.add_argument('--max-testing-samples', default=100, type=int)
    parser.add_argument('--subject-idx', default=None, type=int, nargs='+', help='Index of specific subject(s) to test on')
    parser.add_argument('--test-min-max', action='store_true', help='Only test on min-max time pairs for each subject')
    parser.add_argument('--n-plot-per-epoch', default=4, type=int)

    # Loss coefficients
    parser.add_argument('--no-l2', action='store_true')
    parser.add_argument('--coeff-smoothness', default=0, type=float)
    parser.add_argument('--coeff-latent', default=0, type=float)
    parser.add_argument('--coeff-contrastive', default=0, type=float)
    parser.add_argument('--coeff-invariance', default=0, type=float)
    parser.add_argument('--pretrained-vision-model', default='convnext_tiny', type=str)  # Not used for 3D

    args = vars(parser.parse_args())
    config = AttributeHashmap(args)
    config = parse_settings(config, log_settings=config.mode == 'train', run_count=config.run_count)

    assert config.mode in ['train', 'test']

    seed_everything(config.random_seed)

    if config.mode == 'train':
        train(config=config)
    elif config.mode == 'test':
        test(config=config)
