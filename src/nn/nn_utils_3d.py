import math
import numpy as np
import torch
from torchdiffeq import odeint
import torchsde


class ODEfunc3D(torch.nn.Module):
    """3D version of ODEfunc using Conv3d and InstanceNorm3d."""

    def __init__(self, dim):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.norm1 = torch.nn.InstanceNorm3d(dim)
        self.conv1 = ConcatConv3d(dim, dim, 3, 1, 1)
        self.norm2 = torch.nn.InstanceNorm3d(dim)
        self.conv2 = ConcatConv3d(dim, dim, 3, 1, 1)
        self.dim = dim
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.conv1(t, out)
        out = self.relu(out)
        out = self.norm2(out)
        out = self.conv2(t, out)
        out = self.relu(out)
        return out


class PPODEfunc3D(torch.nn.Module):
    """3D Position-Parameterized ODE function using Conv3d and InstanceNorm3d."""

    def __init__(self, dim):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.norm1 = torch.nn.InstanceNorm3d(dim)
        self.conv1 = torch.nn.Conv3d(dim, dim, 3, 1, 1)
        self.norm2 = torch.nn.InstanceNorm3d(dim)
        self.conv2 = torch.nn.Conv3d(dim, dim, 3, 1, 1)
        self.dim = dim
        self.nfe = 0

    def forward(self, t, x):
        # `t` is a dummy variable here.
        self.nfe += 1
        out = self.norm1(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.norm2(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out


class ODEBlock3D(torch.nn.Module):
    """3D ODE Block - works with any 3D ODE function."""

    def __init__(self, odefunc, tolerance: float = 1e-3):
        super().__init__()
        self.odefunc = odefunc
        self.tolerance = tolerance

    def forward(self, x, integration_time):
        integration_time = integration_time.type_as(x)
        out = odeint(self.odefunc,
                     x,
                     integration_time,
                     rtol=self.tolerance,
                     atol=self.tolerance)
        return out[-1]  # equivalent to `out[1]` if len(integration_time) == 2.

    def flow_field_norm(self, x):
        return torch.norm(self.odefunc(x), p=2)

    def vec_grad(self):
        """
        NOTE: Taking care of Conv3d weights for 3D.
        """
        sum_weight_sq_norm = 0
        for m in self.odefunc.modules():
            if isinstance(m, torch.nn.Conv3d):
                sum_weight_sq_norm += (m.weight ** 2).sum()
        return sum_weight_sq_norm

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class SDEFunc3D(torch.nn.Module):
    """
    3D Stochastic Differential Equation Func.

    It has to include 2 methods:
    self.f: the drift term.
    self.g: the diffusion term.

    NOTE: self.noise_type and self.sde_type are required for torchsde.
    """

    def __init__(self, sde_mu, sde_sigma=0.5, noise_type='diagonal', sde_type='ito'):
        super().__init__()
        self.sde_mu = sde_mu  # drift term
        self.sde_sigma = torch.nn.Parameter(torch.tensor(sde_sigma), requires_grad=True)
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.dim = self.sde_mu.dim

    def f(self, t, x):
        """
        Assuming x is a flattened tensor of [B, C, D, H, W].
        For simplicity, assuming D == H == W (cubic volumes).
        """
        # Calculate spatial dimension (assuming cubic volume)
        x_spatial_dim = int(round((x.shape[-1] / self.dim) ** (1/3)))
        out = x.reshape(x.shape[0], self.dim, x_spatial_dim, x_spatial_dim, x_spatial_dim)
        sde_drift = self.sde_mu(t, out)
        return sde_drift.reshape(sde_drift.shape[0], -1)

    def g(self, t, x):
        return self.sde_sigma.expand_as(x)

    def init_params(self):
        pass


class SDEBlock3D(torch.nn.Module):
    """3D Stochastic Differential Equation block."""

    def __init__(self,
                 sdefunc,
                 tolerance: float = 1e-3,
                 adjoint: bool = False):
        super().__init__()
        self.sdefunc = sdefunc
        self.tolerance = tolerance
        self.adjoint = adjoint

    def forward(self, x, integration_time):
        integration_time = integration_time.type_as(x)
        x = x.reshape(x.shape[0], -1)

        sde_int = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint

        out = sde_int(self.sdefunc,
                      x,
                      integration_time,
                      dt=5e-2,
                      method='euler',
                      rtol=self.tolerance,
                      atol=self.tolerance)
        
        # Calculate spatial dimension (assuming cubic volume)
        out_spatial_dim = int(round((out.shape[-1] / self.sdefunc.dim) ** (1/3)))
        out = out.reshape(out.shape[0], self.sdefunc.dim, 
                         out_spatial_dim, out_spatial_dim, out_spatial_dim)
        return out[-1].unsqueeze(0)

    def init_params(self):
        self.sdefunc.init_params()

    def vec_grad(self):
        """NOTE: Taking care of Conv3d weights."""
        sum_weight_sq_norm = 0
        for m in self.sdefunc.modules():
            if isinstance(m, torch.nn.Conv3d):
                sum_weight_sq_norm += (m.weight ** 2).sum()
        return sum_weight_sq_norm

    @torch.no_grad()
    def forward_traj(self, x, integration_time):
        integration_time = integration_time.type_as(x)
        x = x.reshape(x.shape[0], -1)

        sde_int = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint

        out = sde_int(self.sdefunc,
                      x,
                      integration_time,
                      dt=1e-4,
                      method='euler',
                      rtol=self.tolerance,
                      atol=self.tolerance)
        
        out_spatial_dim = int(round((out.shape[-1] / self.sdefunc.dim) ** (1/3)))
        out = out.reshape(out.shape[0], self.sdefunc.dim,
                         out_spatial_dim, out_spatial_dim, out_spatial_dim)
        return out


class LatentClassifier3D(torch.nn.Module):
    """
    3D classifier model that produces a scalar (-inf, inf) from a tensor.
    Uses global average pooling over 3D spatial dimensions.
    """

    def __init__(self, dim, emb_channels):
        super().__init__()
        self.emb_layer = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(emb_channels, dim),
        )
        self.relu = torch.nn.ReLU(inplace=True)
        self.norm1 = torch.nn.InstanceNorm3d(dim)
        self.conv1 = torch.nn.Conv3d(dim, dim, 3, 1, 1)
        self.norm2 = torch.nn.InstanceNorm3d(dim)
        self.conv2 = torch.nn.Conv3d(dim, dim, 3, 1, 1)
        self.linear = torch.nn.Linear(dim, 1)

    def forward(self, z, emb):
        out = self.norm1(z)
        out = self.conv1(out)
        out = self.relu(out)

        emb_out = self.emb_layer(emb).type(out.dtype)
        while len(emb_out.shape) < len(out.shape):
            emb_out = emb_out[..., None]
        out = out + emb_out

        out = self.norm2(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = out.mean([2, 3, 4])  # global average pooling over D, H, W
        out = out.view(out.shape[0], -1)
        out = self.linear(out)
        return out


class Combine2Channels3D(torch.nn.Module):
    """3D version of Combine2Channels."""

    def __init__(self, dim):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.norm1 = torch.nn.InstanceNorm3d(dim * 2)
        self.conv1 = torch.nn.Conv3d(dim * 2, dim, 3, 1, 1)

    def forward(self, z_concat):
        out = self.norm1(z_concat)
        out = self.conv1(out)
        out = self.relu(out)
        return out


class SODEBlock3D(torch.nn.Module):
    """
    3D State-augmented ODE block.
    z_tj = z_ti + \int_ti^tj f_theta(z_tau, z_s) d tau
    z_s = \sum_k gamma_k z_tk, tk < tau
    gamma_k = softmax(g(z_tk)), tk < tau
    """

    def __init__(self,
                 odefunc,
                 combine_2channels,
                 latent_cls,
                 tolerance: float = 1e-3):
        super().__init__()
        self.odefunc = odefunc
        self.combine_2channels = combine_2channels
        self.latent_cls = latent_cls
        self.tolerance = tolerance

    def forward(self, z_arr, emb, integration_time):
        integration_time = integration_time.type_as(z_arr)
        if len(integration_time) == 1:
            integration_time = torch.tensor([0, integration_time[0]]).float().to(z_arr.device)
        else:
            integration_time = torch.tensor([0, integration_time[-1] - integration_time[-2]]).float().to(z_arr.device)

        num_obs = z_arr.shape[0]
        if num_obs == 1:
            z_cat = torch.cat([z_arr[0].unsqueeze(0),
                               z_arr[0].unsqueeze(0)], dim=1)
        else:
            cls_outputs = self.latent_cls(z=z_arr, emb=emb)
            coeffs = torch.nn.functional.softmax(cls_outputs, dim=0)
            assert len(coeffs.shape) == 2 and coeffs.shape[1] == 1
            coeffs = coeffs.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Extra dim for 3D
            zs = (coeffs * z_arr).sum(dim=0, keepdim=True)
            z_cat = torch.cat([z_arr[0].unsqueeze(0),
                               zs], dim=1)

        z = self.combine_2channels(z_cat)
        out = odeint(self.odefunc,
                     z,
                     integration_time,
                     rtol=self.tolerance,
                     atol=self.tolerance)

        return out[1]

    def vec_grad(self):
        """NOTE: Taking care of Conv3d weights."""
        sum_weight_sq_norm = 0
        for m in self.odefunc.modules():
            if isinstance(m, torch.nn.Conv3d):
                sum_weight_sq_norm += (m.weight ** 2).sum()
        return sum_weight_sq_norm

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ConvBlock3D(torch.nn.Module):
    """3D Convolutional Block."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(num_filters,
                            num_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True),
            torch.nn.InstanceNorm3d(num_filters),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(num_filters,
                            num_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True),
            torch.nn.InstanceNorm3d(num_filters),
            torch.nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class ResConvBlock3D(torch.nn.Module):
    """3D Residual Convolutional Block."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv3d(num_filters,
                            num_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True),
            torch.nn.InstanceNorm3d(num_filters),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv3d(num_filters,
                            num_filters,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True),
            torch.nn.InstanceNorm3d(num_filters),
            torch.nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x) + x


class ConcatConv3d(torch.nn.Module):
    """3D Concatenation Convolution - concatenates time channel before conv."""

    def __init__(self,
                 dim_in,
                 dim_out,
                 ksize=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 transpose=False):
        super(ConcatConv3d, self).__init__()
        module = torch.nn.ConvTranspose3d if transpose else torch.nn.Conv3d
        self._layer = module(dim_in + 1,
                             dim_out,
                             kernel_size=ksize,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             groups=groups,
                             bias=bias)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/nn.py

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
