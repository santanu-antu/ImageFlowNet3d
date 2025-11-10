"""
3D version of ImageFlowNet with ODE blocks.

This file is self-contained: it includes minimal 3D equivalents for the
BaseNetwork utilities and ODE helpers that exist in base.py and nn_utils.py,

Key differences vs 2D:
- Uses Conv3d / InstanceNorm3d throughout
- A lightweight UNet3D backbone (no external diffusion UNet dependency)
- ODE vector field implemented over 3D feature maps

Model entrypoint: ImageFlowNet3DODE
"""

from __future__ import annotations

import os
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional torchdiffeq import. If SciPy/GLIBCXX issues arise, we'll fallback to a pure-PyTorch RK4 integrator inside ODEBlock3D.
try:
    from torchdiffeq import odeint as _tde_odeint
    _HAS_TORCHDIFFEQ = True
except Exception:
    _tde_odeint = None
    _HAS_TORCHDIFFEQ = False



# Minimal BaseNetwork for 3D

class BaseNetwork3D(nn.Module):
    """A minimal base network with save/load helpers and 3D-friendly init."""

    def __init__(self, **kwargs):
        super(BaseNetwork3D, self).__init__()

    def save_weights(self, model_save_path: str) -> None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(self.state_dict(), model_save_path)

    def load_weights(self, model_save_path: str, device: torch.device) -> None:
        self.load_state_dict(torch.load(model_save_path, map_location=device))

    def init_params(self):
        '''
        Parameter initialization for 3D layers.
        '''
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True



# 3D ODE utilities

class PPODEfunc3D(nn.Module):
    """Position-parameterized ODE vector field for 3D feature maps.
    This mirrors PPODEfunc in 2D but uses Conv3d/InstanceNorm3d and ignores t.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.InstanceNorm3d(dim)
        self.conv1 = nn.Conv3d(dim, dim, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(dim)
        self.conv2 = nn.Conv3d(dim, dim, 3, 1, 1)
        self.dim = dim
        self._nfe = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        self._nfe += 1
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.norm2(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out

    @property
    def nfe(self) -> int:
        return self._nfe

    @nfe.setter
    def nfe(self, value: int) -> None:
        self._nfe = value


class ODEBlock3D(nn.Module):
    """3D ODEBlock wrapper.

    Uses torchdiffeq when available; otherwise falls back to a fixed-step RK4
    integrator implemented in pure PyTorch to avoid SciPy dependency issues.
    """

    def __init__(self, odefunc: nn.Module, tolerance: float = 1e-3, solver: str = 'auto', steps: int = 16):
        super().__init__()
        self.odefunc = odefunc
        self.tolerance = tolerance
        self.steps = steps
        if solver == 'auto':
            self.solver = 'tde' if _HAS_TORCHDIFFEQ else 'rk4'
        else:
            assert solver in ['tde', 'rk4']
            self.solver = solver

    @staticmethod
    def _rk4_step(func, t, x, dt):
        k1 = func(t, x)
        k2 = func(t + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = func(t + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = func(t + dt, x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, x: torch.Tensor, integration_time: torch.Tensor) -> torch.Tensor:
        integration_time = integration_time.type_as(x)
        t0 = integration_time[0]
        t1 = integration_time[-1]

        if self.solver == 'tde':
            out = _tde_odeint(self.odefunc, x, integration_time, rtol=self.tolerance, atol=self.tolerance)
            return out[-1]

        # Fallback fixed-step RK4 across [t0, t1]
        total_dt = (t1 - t0).item()
        if total_dt == 0.0:
            return x
        steps = max(1, int(self.steps))
        dt = torch.tensor(total_dt / steps, dtype=x.dtype, device=x.device)
        t = t0
        y = x
        for _ in range(steps):
            y = self._rk4_step(self.odefunc, t, y, dt)
            t = t + dt
        return y

    def vec_grad(self) -> torch.Tensor:
        """Sum of squared Conv3d weight norms for regularization."""
        total = 0.0
        for m in self.odefunc.modules():
            if isinstance(m, nn.Conv3d):
                total = total + (m.weight ** 2).sum()
        return total

    @property
    def nfe(self) -> int:
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value: int) -> None:
        self.odefunc.nfe = value



# 3D UNet backbone (Just a basic one for demonstration)

class _ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """A compact 3D UNet with named blocks to mirror the 2D structure.

    Exposes:
      - input_blocks: ModuleList of encoder blocks (after each, spatial dims shrink via pooling)
      - middle_block: bottleneck block
      - output_blocks: ModuleList of decoder blocks (each expects concatenation with skip)
      - out: final conv to in_channels
    """

    def __init__(self, in_channels: int, base_ch: int = 32, num_levels: int = 4):
        super().__init__()
        chs: List[int] = [base_ch * (2 ** i) for i in range(num_levels)]

        # Encoder
        enc_blocks = []
        prev = in_channels
        for c in chs:
            enc_blocks.append(_ConvBlock3D(prev, c))
            prev = c
        self.input_blocks = nn.ModuleList(enc_blocks)
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.middle_block = _ConvBlock3D(chs[-1], chs[-1] * 2)
        bottleneck_ch = chs[-1] * 2

        # Decoder
        dec_blocks = []
        ups = []
        decoder_in_ch = bottleneck_ch
        for c in reversed(chs):
            # Upsample from current decoder channels to target c channels
            ups.append(nn.ConvTranspose3d(decoder_in_ch, c, kernel_size=2, stride=2))
            # After upsampling, h has c channels; concatenating with skip (also c) yields 2*c input channels
            dec_blocks.append(_ConvBlock3D(2 * c, c))
            decoder_in_ch = c
        self.up = nn.ModuleList(ups)
        self.output_blocks = nn.ModuleList(dec_blocks)

        # Final output conv
        self.out = nn.Conv3d(chs[0], in_channels, kernel_size=1)

    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = x
        skips: List[torch.Tensor] = []
        for idx, block in enumerate(self.input_blocks):
            h = block(h)
            skips.append(h)
            if idx < len(self.input_blocks) - 1:
                h = self.down(h)
        return h, skips

    def forward_decoder(self, h: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        # middle block
        h = self.middle_block(h)
        # decode with skip connections
        for up, block in zip(self.up, self.output_blocks):
            h = up(h)
            skip = skips.pop()  # last skip first
            # Pad if necessary due to odd dims after pooling/upsampling
            if h.shape[-3:] != skip.shape[-3:]:
                diffZ = skip.size(-3) - h.size(-3)
                diffY = skip.size(-2) - h.size(-2)
                diffX = skip.size(-1) - h.size(-1)
                h = F.pad(h, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
            h = torch.cat([h, skip], dim=1)
            h = block(h)
        return self.out(h)

    # For compatibility with 2D code paths that probe channel dims
    @property
    def model_channels(self) -> int:
        return 0  # unused in this lightweight 3D UNet



# ImageFlowNet3D with ODE

class ImageFlowNet3DODE(BaseNetwork3D):
    """3D ImageFlowNet with optional ODE at bottleneck and/or skips.

    Parameters
    ----------
    device: torch.device
    in_channels: int
    ode_location: 'bottleneck' | 'all_resolutions' | 'all_connections'
    contrastive: bool (unused placeholder for parity)
    volume_size: Tuple[int,int,int] default volume to infer dims for ODE shape probing
    """

    def __init__(
        self,
        device: torch.device,
        in_channels: int,
        ode_location: str = 'all_connections',
        contrastive: bool = False,
        volume_size: Tuple[int, int, int] = (32, 128, 128),
        base_ch: int = 32,
        num_levels: int = 4,
        **kwargs,
    ):
        super().__init__()
        assert ode_location in ['bottleneck', 'all_resolutions', 'all_connections']

        self.device = device
        self.ode_location = ode_location
        self.contrastive = contrastive

        # Backbone UNet3D
        self.unet = UNet3D(in_channels=in_channels, base_ch=base_ch, num_levels=num_levels)
        self.unet.to(self.device)

        # Probe channel dims by a dummy forward
        D, H, W = volume_size
        h_dummy = torch.zeros((1, in_channels, D, H, W), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # Run encoder only to collect dims similar to 2D code
            h_enc, skips = self.unet.forward_encoder(h_dummy)
            # Channel dims after each encoder block
            self.dim_list: List[int] = [t.shape[1] for t in skips]
            # Middle block dim
            h_bottleneck = self.unet.middle_block(h_enc)
            self.dim_list.append(h_bottleneck.shape[1])

        # Build ODE modules
        self.ode_list = nn.ModuleList([])
        if self.ode_location == 'bottleneck':
            self.ode_list.append(ODEBlock3D(PPODEfunc3D(dim=h_bottleneck.shape[1])))
        elif self.ode_location == 'all_resolutions':
            for dim in sorted(set(self.dim_list)):
                self.ode_list.append(ODEBlock3D(PPODEfunc3D(dim=dim)))
        elif self.ode_location == 'all_connections':
            for dim in self.dim_list:
                self.ode_list.append(ODEBlock3D(PPODEfunc3D(dim=dim)))

        self.ode_list.to(self.device)

    def time_independent_parameters(self):
        return set(self.parameters()) - set(self.ode_list.parameters())

    def freeze_time_independent(self):
        for p in self.time_independent_parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, t: torch.Tensor, return_grad: bool = False):
        """
        x: [N, C, D, H, W]
        t: scalar (tensor shape [1]) time difference
        """
        # If no time difference, skip ODE
        use_ode = t.item() != 0
        if use_ode:
            integration_time = torch.tensor([0, float(t.item())], dtype=torch.float32, device=x.device)

        # Encoder path (collect skips) and bottleneck
        h, skips = self.unet.forward_encoder(x)
        h = self.unet.middle_block(h)

        # Bottleneck ODE (apply on the bottleneck features)
        if use_ode:
            # The last recorded dim in dim_list corresponds to bottleneck.
            # Our ODE list ordering places bottleneck at -1.
            h = self.ode_list[-1](h, integration_time)

        # Decoder path; apply ODE over skip connections as configured
        # We must mirror the ordering used when building ode_list
        ode_idx_from_tail = 1  # after bottleneck
        for up, block in zip(self.unet.up, self.unet.output_blocks):
            h = up(h)
            h_skip = skips.pop()

            if use_ode and self.ode_location in ['all_resolutions', 'all_connections']:
                if self.ode_location == 'all_connections':
                    curr_ode_block = self.ode_list[::-1][ode_idx_from_tail]
                else:
                    # map by channel dimension
                    target_dim = h_skip.shape[1]
                    # find the first block whose func dim matches
                    curr_ode_block = None
                    for blk in self.ode_list:
                        if isinstance(blk.odefunc, PPODEfunc3D) and blk.odefunc.dim == target_dim:
                            curr_ode_block = blk
                            break
                    if curr_ode_block is None:
                        raise RuntimeError('No matching ODE block for skip connection dim={}'.format(target_dim))
                h_skip = curr_ode_block(h_skip, integration_time)
                ode_idx_from_tail += 1

            # cat and decode block
            # pad handled inside UNet3D decoder earlier; we ensure size match here too
            if h.shape[-3:] != h_skip.shape[-3:]:
                diffZ = h_skip.size(-3) - h.size(-3)
                diffY = h_skip.size(-2) - h.size(-2)
                diffX = h_skip.size(-1) - h.size(-1)
                h = F.pad(h, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
            h = torch.cat([h, h_skip], dim=1)
            h = block(h)

        output = self.unet.out(h).type_as(x)

        if return_grad:
            vec_field_gradients = 0
            for blk in self.ode_list:
                vec_field_gradients = vec_field_gradients + blk.vec_grad()
            return output, vec_field_gradients.mean() / (len(self.ode_list) if len(self.ode_list) > 0 else 1)
        else:
            return output
