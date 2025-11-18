from __future__ import annotations

import time
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import utils
from .networks import CheckpointConfig, PatchDiscriminator, SPADEGenerator

if TYPE_CHECKING:
    from .data import SegmentationImageDataset


@dataclass
class TrainingConfig:
    label_nc: int
    image_size: tuple[int, int]
    output_dir: Path
    epochs: int
    max_steps: Optional[int]
    lr: float
    beta1: float
    beta2: float
    lambda_l1: float
    save_every: int
    sample_every: int
    log_every: int
    z_dim: int
    ngf: int
    ndf: int
    num_upsampling_layers: str
    param_free_norm: str
    device: torch.device
    dataset_metadata: Dict[str, object]
    resume: Optional[Path] = None


class SPADETrainer:
    def __init__(self, config: TrainingConfig, dataset: 'SegmentationImageDataset') -> None:
        self.cfg = config
        self.device = config.device
        self.dataset = dataset

        palette_source = getattr(dataset, 'palette', None) or config.dataset_metadata.get('palette', {})
        self.palette = {
            int(k): (int(v[0]), int(v[1]), int(v[2]))
            for k, v in palette_source.items()
        } if palette_source else {
            idx: (idx, idx, idx) for idx in range(config.label_nc)
        }

        self.generator = SPADEGenerator(
            label_nc=config.label_nc,
            output_nc=3,
            ngf=config.ngf,
            z_dim=config.z_dim,
            image_size=config.image_size,
            num_upsampling_layers=config.num_upsampling_layers,
            param_free_norm=config.param_free_norm,
        ).to(self.device)

        self.discriminator = PatchDiscriminator(
            input_nc=config.label_nc + 3,
            ndf=config.ndf,
        ).to(self.device)

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()

        self.global_step = 0
        self.start_epoch = 0

        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.sample_dir = self.output_dir / 'samples'
        utils.ensure_dir(self.checkpoint_dir)
        utils.ensure_dir(self.sample_dir)

        if config.resume:
            self._load_checkpoint(config.resume)

    def _save_checkpoint(self, epoch: int) -> Path:
        ckpt_config = CheckpointConfig(
            label_nc=self.cfg.label_nc,
            ngf=self.cfg.ngf,
            ndf=self.cfg.ndf,
            z_dim=self.cfg.z_dim,
            num_upsampling_layers=self.cfg.num_upsampling_layers,
            image_size=self.cfg.image_size,
            param_free_norm=self.cfg.param_free_norm,
        )
        payload = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'config': asdict(ckpt_config),
            'dataset_metadata': self.cfg.dataset_metadata,
            'global_step': self.global_step,
            'epoch': epoch,
            'timestamp': time.time(),
        }
        path = self.checkpoint_dir / f'spade_step_{self.global_step:07d}.pt'
        torch.save(payload, path)
        return path

    def _load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        self.global_step = int(checkpoint.get('global_step', 0))
        self.start_epoch = int(checkpoint.get('epoch', 0)) + 1

    def _log(self, message: str) -> None:
        print(message, flush=True)

    def train(self, dataloader: DataLoader) -> None:
        device = self.device
        cfg = self.cfg

        for epoch in range(self.start_epoch, cfg.epochs):
            self.generator.train()
            self.discriminator.train()
            epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
            for batch in epoch_bar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                onehot = utils.masks_to_onehot(masks, cfg.label_nc).to(device)

                ############################
                # Train Discriminator
                ############################
                self.d_optimizer.zero_grad(set_to_none=True)

                fake_images = self.generator(onehot)
                real_input = torch.cat([onehot, images], dim=1)
                fake_input = torch.cat([onehot, fake_images.detach()], dim=1)

                pred_real = self.discriminator(real_input)
                pred_fake = self.discriminator(fake_input)

                real_loss = self.criterion_gan(pred_real, torch.ones_like(pred_real))
                fake_loss = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward()
                self.d_optimizer.step()

                ############################
                # Train Generator
                ############################
                self.g_optimizer.zero_grad(set_to_none=True)

                fake_input = torch.cat([onehot, fake_images], dim=1)
                pred_fake_for_g = self.discriminator(fake_input)

                gan_loss = self.criterion_gan(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
                l1_loss = self.criterion_l1(fake_images, images) * cfg.lambda_l1
                g_loss = gan_loss + l1_loss
                g_loss.backward()
                self.g_optimizer.step()

                self.global_step += 1

                if self.global_step % cfg.log_every == 0:
                    epoch_bar.set_postfix({
                        'd_loss': f"{d_loss.item():.3f}",
                        'g_loss': f"{g_loss.item():.3f}",
                        'l1': f"{l1_loss.item():.3f}",
                    })

                if cfg.sample_every and self.global_step % cfg.sample_every == 0:
                    self._write_samples()

                if cfg.save_every and self.global_step % cfg.save_every == 0:
                    ckpt_path = self._save_checkpoint(epoch)
                    self._log(f"Saved checkpoint to {ckpt_path}")

                if cfg.max_steps and self.global_step >= cfg.max_steps:
                    self._log("Reached max_steps, stopping training loop.")
                    return

    def _write_samples(self, num_items: int = 8, grid: Sequence[int] = (2, 4)) -> None:
        if self.dataset is None or len(self.dataset) == 0:
            return

        rows, cols = int(grid[0]), int(grid[1])
        max_items = rows * cols
        count = min(num_items, max_items, len(self.dataset))
        if count == 0:
            return

        indices = random.sample(range(len(self.dataset)), count)
        original_flip = getattr(self.dataset, 'random_flip', False)
        self.dataset.random_flip = False
        try:
            samples = [self.dataset[idx] for idx in indices]
        finally:
            self.dataset.random_flip = original_flip

        images = torch.stack([sample['image'] for sample in samples], dim=0)
        masks = torch.stack([sample['mask'] for sample in samples], dim=0)

        onehot = utils.masks_to_onehot(masks.to(self.device), self.cfg.label_nc)

        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(count, self.cfg.z_dim, device=self.device)
            synth = self.generator(onehot[:count], z)
        self.generator.train()

        originals_pil = [utils.tensor_to_pil(image) for image in images]
        masks_pil = [utils.mask_to_color_image(mask, self.palette) for mask in masks]
        synth_pil = [utils.tensor_to_pil(image) for image in synth]

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        step_dir = self.sample_dir / f"step_{self.global_step:07d}_{timestamp}"
        utils.ensure_dir(step_dir)

        utils.save_mosaic(originals_pil, (rows, cols), step_dir / 'mosaic_originals.png')
        utils.save_mosaic(masks_pil, (rows, cols), step_dir / 'mosaic_segmentation.png')
        utils.save_mosaic(synth_pil, (rows, cols), step_dir / 'mosaic_synth.png')
