import torch
import torch.nn.functional as F


def gradient_3d(vol):
    dz = vol[:, :, :, 1:] - vol[:, :, :, :-1]
    dy = vol[:, :, 1:, :] - vol[:, :, :-1, :]
    dx = vol[:, 1:, :, :] - vol[:, :-1, :, :]
    return dx, dy, dz


def tv_3d(vol):
    dx, dy, dz = gradient_3d(vol)
    return (dx.abs().mean() + dy.abs().mean() + dz.abs().mean())


def second_derivative_z(vol):
    kernel = torch.tensor([1.0, -2.0, 1.0], device=vol.device).view(1, 1, 1, 1, 3)
    return F.conv3d(vol, kernel, padding=(0, 0, 1))
