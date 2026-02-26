from __future__ import annotations

import contextlib
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import utils
from .guided_diffusion import script_util, logger
from .guided_diffusion.resample import LossAwareSampler, create_named_schedule_sampler


@dataclass
class TrainingConfig:
    label_nc: int  # Number of segmentation classes
    conditioning_channels: int  # Total conditioning channels for UNet (segmentation + optional edges)
    image_size: Tuple[int, int]
    output_dir: Path
    epochs: int
    max_steps: Optional[int]
    lr: float
    weight_decay: float
    ema_decay: float
    log_every: int
    save_every: int
    sample_every: int
    device: torch.device
    dataset_metadata: Dict[str, Any]
    resume: Optional[Path]
    diffusion_params: Dict[str, Any]
    schedule_sampler: str = "uniform"
    gradient_clip: Optional[float] = None
    use_fp16: bool = False
    condition_with_edges: bool = True
    num_visualization_items: int = 8
    visualization_grid: Tuple[int, int] = (2, 4)
    preview_current_model: bool = True
    preview_ema_model: bool = False
    class_cond: bool = True

    def to_serializable(self) -> Dict[str, Any]:
        data = asdict(self)
        data['output_dir'] = str(self.output_dir)
        data['device'] = str(self.device)
        data['resume'] = str(self.resume) if self.resume else None
        data['image_size'] = tuple(int(x) for x in self.image_size)
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'TrainingConfig':
        data = dict(data)
        data['output_dir'] = Path(data['output_dir'])
        data['device'] = torch.device(data.get('device', 'cpu'))
        resume = data.get('resume')
        data['resume'] = Path(resume) if resume else None
        data['image_size'] = tuple(data['image_size'])
        data.setdefault('schedule_sampler', 'uniform')
        data.setdefault('preview_current_model', True)
        data.setdefault('preview_ema_model', False)
        data.setdefault('class_cond', True)
        if 'conditioning_channels' not in data:
            data['conditioning_channels'] = data['label_nc'] + (1 if data.get('condition_with_edges', True) else 0)
        return TrainingConfig(**data)


class SISDMTrainer:
    """Trainer for Segmentation-guided Image Synthesis using Diffusion Models."""
    
    def __init__(self, config: TrainingConfig) -> None:
        self.cfg = config
        self.device = config.device

        utils.ensure_dir(config.output_dir)
        self.checkpoint_dir = Path(config.output_dir) / 'checkpoints'
        self.sample_dir = Path(config.output_dir) / 'samples'
        utils.ensure_dir(self.checkpoint_dir)
        utils.ensure_dir(self.sample_dir)

        # Setup diffusion model
        model_kwargs = script_util.model_and_diffusion_defaults()
        model_kwargs.update(config.diffusion_params)
        
        # Set image size and conditioning parameters
        model_kwargs['image_size'] = config.image_size[0]  # Assume square images
        
        # Validate conditioning configuration
        expected_channels = config.label_nc + (1 if config.condition_with_edges else 0)
        if expected_channels != config.conditioning_channels:
            raise ValueError(
                "conditioning_channels mismatch: expected"
                f" {expected_channels} (label_nc {config.label_nc}"
                f" + edges {int(config.condition_with_edges)}) but got"
                f" {config.conditioning_channels}."
            )

        # Align UNet conditioning with segmentation classes; guided_diffusion will
        # automatically add an instance/edge channel when `no_instance` is False.
        if config.class_cond:
            model_kwargs['num_classes'] = config.label_nc
        else:
            model_kwargs['num_classes'] = 0

        model_kwargs['class_cond'] = config.class_cond

        self.model, self.diffusion = script_util.create_model_and_diffusion(**model_kwargs)
        self.model.to(self.device)

        self.schedule_sampler = create_named_schedule_sampler(config.schedule_sampler, self.diffusion)

        # EMA model
        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Mixed precision training
        self.use_amp = config.use_fp16 and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.global_step = 0
        self.start_epoch = 0

        if config.resume:
            self.load_checkpoint(config.resume)

        # Save config
        config_path = config.output_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(config.to_serializable(), f, indent=2)

    def train(self, data_loader: DataLoader) -> None:
        """Run the training loop."""
        self.model.train()
        
        total_steps = 0
        if self.cfg.max_steps:
            total_steps = self.cfg.max_steps
        else:
            total_steps = len(data_loader) * self.cfg.epochs

        with tqdm(total=total_steps, desc="Training", initial=self.global_step) as pbar:
            for epoch in range(self.start_epoch, self.cfg.epochs):
                if self.cfg.max_steps and self.global_step >= self.cfg.max_steps:
                    break
                    
                epoch_losses = []
                
                for batch_idx, (images, batch_dict) in enumerate(data_loader):
                    if self.cfg.max_steps and self.global_step >= self.cfg.max_steps:
                        break
                        
                    # Move to device
                    images = images.to(self.device)
                    masks = batch_dict['label'].to(self.device)
                    
                    # Prepare conditioning
                    model_kwargs = {}
                    if self.cfg.class_cond:
                        # Use segmentation masks as class conditioning
                        conditioning = utils.compute_diffusion_input_conditioning(
                            masks, 
                            self.cfg
                        ).to(self.device)
                        
                        model_kwargs['y'] = conditioning
                    
                    # Training step
                    loss = self._train_step(images, model_kwargs)
                    epoch_losses.append(loss)
                    
                    # Update EMA
                    self._update_ema()
                    
                    # Logging
                    if self.global_step % self.cfg.log_every == 0:
                        avg_loss = sum(epoch_losses[-self.cfg.log_every:]) / len(epoch_losses[-self.cfg.log_every:])
                        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                    
                    # Sampling
                    if self.global_step % self.cfg.sample_every == 0:
                        # Collect enough samples for visualization
                        vis_images_list = [images]
                        vis_masks_list = [masks]
                        collected = images.shape[0]
                        temp_iter = iter(data_loader)
                        while collected < self.cfg.num_visualization_items:
                            try:
                                next_images, next_batch_dict = next(temp_iter)
                                vis_images_list.append(next_images.to(self.device))
                                vis_masks_list.append(next_batch_dict['label'].to(self.device))
                                collected += next_images.shape[0]
                            except StopIteration:
                                break
                        vis_images = torch.cat(vis_images_list, dim=0)[:self.cfg.num_visualization_items]
                        vis_masks = torch.cat(vis_masks_list, dim=0)[:self.cfg.num_visualization_items]
                        self._generate_samples(vis_images, vis_masks)
                    
                    # Checkpointing
                    if self.global_step % self.cfg.save_every == 0 and self.global_step > 0:
                        self._save_checkpoint()
                    
                    self.global_step += 1
                    pbar.update(1)

    def _train_step(self, images: torch.Tensor, model_kwargs: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step."""
        self.optimizer.zero_grad()
        
        # Sample timesteps
        t, weights = self.schedule_sampler.sample(images.shape[0], self.device)
        
        # Forward pass with mixed precision
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            loss_dict = self.diffusion.training_losses(
                self.model, images, t, model_kwargs=model_kwargs
            )
            loss = (loss_dict['loss'] * weights).mean()
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        if self.cfg.gradient_clip is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

    def _update_ema(self) -> None:
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.cfg.ema_decay).add_(param.data, alpha=1 - self.cfg.ema_decay)

    def _generate_samples(self, images: torch.Tensor, masks: torch.Tensor) -> None:
        """Generate and save sample images."""
        self.model.eval()
        
        with torch.no_grad():
            # Prepare conditioning from provided samples
            num_items = masks.shape[0]
            masks = masks[:num_items]
            image_slice = images[:num_items]
            
            model_kwargs = {}
            if self.cfg.class_cond:
                conditioning = utils.compute_diffusion_input_conditioning(
                    masks,
                    self.cfg
                ).to(self.device)
                model_kwargs['y'] = conditioning
            
            # Sample from current model
            if self.cfg.preview_current_model or self.cfg.preview_ema_model:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                step_dir = self.sample_dir / f"step_{self.global_step:07d}_{timestamp}"
                utils.ensure_dir(step_dir)

                palette = self.cfg.dataset_metadata.get('palette') if isinstance(self.cfg.dataset_metadata, dict) else None
                mask_images = [
                    utils.mask_to_color_image(mask.cpu(), size=self.cfg.image_size[::-1], palette=palette)
                    for mask in masks
                ]
                real_images = [utils.tensor_to_pil_image(img) for img in image_slice]

                utils.save_mosaic(real_images, self.cfg.visualization_grid, step_dir / 'mosaic_real.png')
                utils.save_mosaic(mask_images, self.cfg.visualization_grid, step_dir / 'mosaic_mask.png')

                if self.cfg.preview_current_model:
                    sample_shape = (num_items, 3, *self.cfg.image_size)
                    samples = self.diffusion.p_sample_loop(
                        self.model,
                        sample_shape,
                        model_kwargs=model_kwargs,
                        clip_denoised=True,
                        device=self.device
                    )
                    synth_images = [utils.tensor_to_pil_image(sample) for sample in samples]
                    utils.save_mosaic(synth_images, self.cfg.visualization_grid, step_dir / 'mosaic_synth_current.png')

                if self.cfg.preview_ema_model:
                    sample_shape = (num_items, 3, *self.cfg.image_size)
                    samples_ema = self.diffusion.p_sample_loop(
                        self.ema_model,
                        sample_shape,
                        model_kwargs=model_kwargs,
                        clip_denoised=True,
                        device=self.device
                    )
                    synth_images_ema = [utils.tensor_to_pil_image(sample) for sample in samples_ema]
                    utils.save_mosaic(synth_images_ema, self.cfg.visualization_grid, step_dir / 'mosaic_synth_ema.png')
        
        self.model.train()

    def _save_checkpoint(self) -> None:
        """Save model checkpoint."""
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'global_step': self.global_step,
            'config': self.cfg.to_serializable(),
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step:06d}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint_data, latest_path)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model from checkpoint."""
        checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint_data['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint_data and checkpoint_data['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
        
        self.global_step = checkpoint_data['global_step']
        self.start_epoch = self.global_step // self.cfg.dataset_metadata.get('num_samples', 1000)  # Rough estimate

    def generate(self, masks: torch.Tensor, instance_maps: Optional[torch.Tensor] = None,
                 use_ema: bool = True, num_samples: int = 1) -> torch.Tensor:
        """Generate images conditioned on segmentation masks."""
        model = self.ema_model if use_ema else self.model
        model.eval()
        
        with torch.no_grad():
            batch_size = masks.size(0)
            
            model_kwargs = {}
            if self.cfg.class_cond:
                conditioning = utils.compute_diffusion_input_conditioning(
                    masks, self.cfg
                ).to(self.device)
                model_kwargs['y'] = conditioning
            
            sample_shape = (batch_size * num_samples, 3, *self.cfg.image_size)
            
            samples = self.diffusion.p_sample_loop(
                model,
                sample_shape,
                model_kwargs=model_kwargs,
                clip_denoised=True,
                device=self.device
            )
            
            if num_samples > 1:
                samples = samples.view(batch_size, num_samples, *samples.shape[1:])
            
        return samples


def load_trainer_from_checkpoint(checkpoint_path: Path, device: Optional[torch.device] = None) -> SISDMTrainer:
    """Load a trainer instance from a saved checkpoint."""
    checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
    config_data = checkpoint_data['config']
    
    if device is not None:
        config_data['device'] = str(device)
    
    config = TrainingConfig.from_dict(config_data)
    trainer = SISDMTrainer(config)
    trainer.load_checkpoint(checkpoint_path)
    
    return trainer