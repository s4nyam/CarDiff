from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """Spatially-Adaptive (DE)normalization layer."""

    def __init__(self, norm_channels: int, label_nc: int, hidden_channels: int = 128,
                 param_free_norm: str = 'batch') -> None:
        super().__init__()
        norm_type = param_free_norm.lower()
        if norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_channels, affine=False)
        elif norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_channels, affine=False)
        else:
            raise ValueError(f"Unsupported param-free norm: {param_free_norm}")

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden_channels, norm_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_channels, norm_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        normalized = self.param_free_norm(x)
        seg_resized = F.interpolate(segmap, size=x.shape[2:], mode='nearest')
        actv = self.mlp_shared(seg_resized)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1.0 + gamma) + beta


class SPADEResnetBlock(nn.Module):
    """Residual block with SPADE conditioning."""

    def __init__(self, input_nc: int, output_nc: int, label_nc: int,
                 hidden_channels: Optional[int] = None,
                 param_free_norm: str = 'batch') -> None:
        super().__init__()
        self.learned_shortcut = input_nc != output_nc
        middle_nc = hidden_channels or min(input_nc, output_nc)

        self.conv_0 = nn.Conv2d(input_nc, middle_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(middle_nc, output_nc, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(input_nc, output_nc, kernel_size=1, bias=False)
        else:
            self.conv_s = None

        self.norm_0 = SPADE(input_nc, label_nc, param_free_norm=param_free_norm)
        self.norm_1 = SPADE(middle_nc, label_nc, param_free_norm=param_free_norm)
        if self.learned_shortcut:
            self.norm_s = SPADE(input_nc, label_nc, param_free_norm=param_free_norm)
        else:
            self.norm_s = None

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        shortcut = self._shortcut(x, segmap)
        dx = self.conv_0(F.leaky_relu(self.norm_0(x, segmap), negative_slope=0.2, inplace=True))
        dx = self.conv_1(F.leaky_relu(self.norm_1(dx, segmap), negative_slope=0.2, inplace=True))
        return shortcut + dx

    def _shortcut(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut and self.conv_s is not None and self.norm_s is not None:
            return self.conv_s(self.norm_s(x, segmap))
        return x


class SPADEGenerator(nn.Module):
    """Generator architecture closely following the SPADE design."""

    def __init__(self,
                 label_nc: int,
                 output_nc: int = 3,
                 ngf: int = 64,
                 z_dim: int = 256,
                 image_size: Tuple[int, int] = (256, 256),
                 num_upsampling_layers: str = 'normal',
                 param_free_norm: str = 'batch') -> None:
        super().__init__()
        self.label_nc = label_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.z_dim = z_dim
        self.param_free_norm = param_free_norm
        self.num_upsampling_layers = num_upsampling_layers

        self.sh, self.sw = self._compute_initial_hw(image_size, num_upsampling_layers)

        self.fc = nn.Linear(z_dim, 16 * ngf * self.sh * self.sw)
        self.head_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, label_nc, param_free_norm=param_free_norm)
        self.G_middle_0 = SPADEResnetBlock(16 * ngf, 16 * ngf, label_nc, param_free_norm=param_free_norm)
        self.G_middle_1 = SPADEResnetBlock(16 * ngf, 16 * ngf, label_nc, param_free_norm=param_free_norm)
        self.up_0 = SPADEResnetBlock(16 * ngf, 8 * ngf, label_nc, param_free_norm=param_free_norm)
        self.up_1 = SPADEResnetBlock(8 * ngf, 4 * ngf, label_nc, param_free_norm=param_free_norm)
        self.up_2 = SPADEResnetBlock(4 * ngf, 2 * ngf, label_nc, param_free_norm=param_free_norm)
        self.up_3 = SPADEResnetBlock(2 * ngf, 1 * ngf, label_nc, param_free_norm=param_free_norm)

        if num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(ngf, ngf // 2, label_nc, param_free_norm=param_free_norm)
            final_nc = ngf // 2
        else:
            self.up_4 = None
            final_nc = ngf

        self.conv_img = nn.Conv2d(final_nc, output_nc, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    @staticmethod
    def _num_layers(setting: str) -> int:
        if setting == 'normal':
            return 5
        if setting == 'more':
            return 6
        if setting == 'most':
            return 7
        raise ValueError(f"Unknown num_upsampling_layers setting: {setting}")

    def _compute_initial_hw(self, image_size: Tuple[int, int], setting: str) -> Tuple[int, int]:
        num_up = self._num_layers(setting)
        height, width = image_size
        sw = max(1, width // (2 ** num_up))
        sh = max(1, height // (2 ** num_up))
        return sh, sw

    def forward(self, segmap: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = segmap.size(0)
        if z is None:
            z = torch.randn(batch_size, self.z_dim, device=segmap.device)
        x = self.fc(z).view(batch_size, 16 * self.ngf, self.sh, self.sw)

        x = self.head_0(x, segmap)
        x = self.upsample(x)
        x = self.G_middle_0(x, segmap)

        if self.num_upsampling_layers in {'more', 'most'}:
            x = self.upsample(x)

        x = self.G_middle_1(x, segmap)
        x = self.upsample(x)
        x = self.up_0(x, segmap)
        x = self.upsample(x)
        x = self.up_1(x, segmap)
        x = self.upsample(x)
        x = self.up_2(x, segmap)
        x = self.upsample(x)
        x = self.up_3(x, segmap)

        if self.up_4 is not None:
            x = self.upsample(x)
            x = self.up_4(x, segmap)

        x = self.conv_img(F.leaky_relu(x, negative_slope=0.2, inplace=True))
        return torch.tanh(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator that conditions on the segmentation map."""

    def __init__(self, input_nc: int, ndf: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, num_layers):
            nf_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class CheckpointConfig:
    label_nc: int
    ngf: int
    ndf: int
    z_dim: int
    num_upsampling_layers: str
    image_size: Tuple[int, int]
    param_free_norm: str = 'batch'

