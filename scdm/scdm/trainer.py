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
from .guided_diffusion import script_util
from .guided_diffusion.resample import LossAwareSampler, create_named_schedule_sampler


@dataclass
class TrainingConfig:
    label_nc: int
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
        return TrainingConfig(**data)


class SCDMTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.cfg = config
        self.device = config.device

        utils.ensure_dir(config.output_dir)
        self.checkpoint_dir = Path(config.output_dir) / 'checkpoints'
        self.sample_dir = Path(config.output_dir) / 'samples'
        utils.ensure_dir(self.checkpoint_dir)
        utils.ensure_dir(self.sample_dir)

        model_kwargs = script_util.model_and_diffusion_defaults()
        model_kwargs.update(config.diffusion_params)
        self.model, self.diffusion = script_util.create_model_and_diffusion(**model_kwargs)
        self.model.to(self.device)

        self.schedule_sampler = create_named_schedule_sampler(config.schedule_sampler, self.diffusion)

        self.ema_model = copy.deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        self.use_amp = config.use_fp16 and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.global_step = 0
        self.start_epoch = 0

        if config.resume:
            self._load_checkpoint(config.resume)

    def _save_metadata(self) -> None:
        metadata_path = Path(self.cfg.output_dir) / 'dataset_metadata.json'
        with metadata_path.open('w') as f:
            json.dump(self.cfg.dataset_metadata, f, indent=2)

    def _save_checkpoint(self, epoch: int) -> Path:
        payload = {
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.cfg.to_serializable(),
            'dataset_metadata': self.cfg.dataset_metadata,
            'global_step': self.global_step,
            'epoch': epoch,
            'timestamp': time.time(),
        }
        path = self.checkpoint_dir / f'scdm_step_{self.global_step:07d}.pt'
        torch.save(payload, path)
        return path

    def _load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        if 'ema_model' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = int(checkpoint.get('global_step', 0))
        self.start_epoch = int(checkpoint.get('epoch', 0)) + 1

    def _update_ema(self) -> None:
        decay = self.cfg.ema_decay
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.mul_(decay).add_(param, alpha=1.0 - decay)
            for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers()):
                ema_buffer.copy_(buffer)

    def _generate_samples(self, model: torch.nn.Module, cond: torch.Tensor, count: int) -> torch.Tensor:
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                samples = self.diffusion.p_sample_loop(
                    model,
                    (count, 3, self.cfg.image_size[0], self.cfg.image_size[1]),
                    clip_denoised=True,
                    model_kwargs={'y': cond},
                    progress=False,
                )
        finally:
            if was_training:
                model.train()
        return samples

    def train(self, dataloader: DataLoader) -> None:
        self._save_metadata()
        device = self.device
        cfg = self.cfg

        if getattr(self, 'dataset_cache', None) is None and hasattr(dataloader, 'dataset'):
            try:
                self.dataset_cache = dataloader.dataset
            except AttributeError:
                pass

        for epoch in range(self.start_epoch, cfg.epochs):
            progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False)
            for batch in progress:
                images = batch['image'].to(device)

                masks = batch['mask'].to(device)
                onehot = utils.masks_to_onehot(masks, cfg.label_nc).to(device)
                if cfg.condition_with_edges:
                    edges = utils.mask_to_edge_map(masks).to(device)
                    cond = torch.cat([onehot, edges], dim=1)
                else:
                    cond = onehot

                batch_size = images.shape[0]
                t, weights = self.schedule_sampler.sample(batch_size, device)
                weights = weights.to(device, dtype=torch.float32)

                self.optimizer.zero_grad(set_to_none=True)
                autocast_ctx = (
                    torch.amp.autocast(device_type='cuda', enabled=True)
                    if self.use_amp
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    losses = self.diffusion.training_losses(
                        self.model,
                        images,
                        t,
                        model_kwargs={'y': cond},
                    )
                    per_example_loss = losses['loss']
                    loss = (per_example_loss * weights).mean()

                self.scaler.scale(loss).backward()

                if cfg.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.global_step += 1
                self._update_ema()

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(t, per_example_loss.detach())

                if self.global_step % cfg.log_every == 0:
                    loss_values = {k: v.mean().item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
                    progress.set_postfix({
                        'loss': f"{loss.detach().item():.4f}",
                        **{k: f"{val:.4f}" for k, val in loss_values.items() if isinstance(val, float)}
                    })

                if cfg.sample_every and self.global_step % cfg.sample_every == 0:
                    self._write_samples()

                if cfg.save_every and self.global_step % cfg.save_every == 0:
                    ckpt_path = self._save_checkpoint(epoch)
                    print(f"Saved checkpoint to {ckpt_path}")

                if cfg.max_steps and self.global_step >= cfg.max_steps:
                    print("Reached max_steps, stopping training loop.")
                    return

            # Save checkpoint at end of epoch if not already saved this step
            if cfg.save_every:
                ckpt_path = self._save_checkpoint(epoch)
                print(f"Saved checkpoint to {ckpt_path}")

    def _write_samples(self) -> None:
        dataset = getattr(self, 'dataset_cache', None)
        if dataset is None:
            return

        cfg = self.cfg
        count = min(cfg.num_visualization_items, len(dataset))
        if count == 0:
            return

        indices = torch.randperm(len(dataset))[:count].tolist()
        items = [dataset[idx] for idx in indices]

        images = torch.stack([item['image'] for item in items])
        masks = torch.stack([item['mask'] for item in items])

        masks_device = masks.to(self.device)
        onehot = utils.masks_to_onehot(masks_device, cfg.label_nc).to(self.device)
        if cfg.condition_with_edges:
            edges = utils.mask_to_edge_map(masks_device).to(self.device)
            cond = torch.cat([onehot, edges], dim=1)
        else:
            cond = onehot

        palette = cfg.dataset_metadata.get('palette') or {
            idx: (int(idx), int(idx), int(idx)) for idx in range(cfg.label_nc)
        }
        mask_pil = [utils.mask_to_color_image(mask, palette) for mask in masks]
        real_images = [utils.tensor_to_pil(image) for image in images]

        if not cfg.preview_current_model:
            return

        current_samples = self._generate_samples(self.model, cond, count)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        step_dir = self.sample_dir / f"step_{self.global_step:07d}_{timestamp}"
        utils.ensure_dir(step_dir)

        utils.save_mosaic(real_images, cfg.visualization_grid, step_dir / 'mosaic_real.png')
        utils.save_mosaic(mask_pil, cfg.visualization_grid, step_dir / 'mosaic_mask.png')
        synth_images = [utils.tensor_to_pil(sample) for sample in current_samples]
        utils.save_mosaic(synth_images, cfg.visualization_grid, step_dir / 'mosaic_synth_current.png')

    def attach_dataset(self, dataset: Sequence[Dict[str, Any]]) -> None:
        """Provide dataset samples for visualization without augmentations."""
        self.dataset_cache = dataset


def load_trainer_from_checkpoint(path: Path, device: Optional[torch.device] = None) -> Tuple[SCDMTrainer, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device or 'cpu')
    cfg_dict = checkpoint['config']
    # Force path rewrite
    if cfg_dict.get("resume"):
        cfg_dict["resume"] = str(cfg_dict["resume"]).replace(
            "project_465001696",
            "project_465002351"
        )
    cfg = TrainingConfig.from_dict(cfg_dict)
    if device is not None:
        cfg.device = device

    trainer = SCDMTrainer(cfg)
    trainer.model.load_state_dict(checkpoint['model'])
    if 'ema_model' in checkpoint:
        trainer.ema_model.load_state_dict(checkpoint['ema_model'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    trainer.global_step = int(checkpoint.get('global_step', 0))
    trainer.start_epoch = int(checkpoint.get('epoch', 0))
    trainer.cfg.dataset_metadata = checkpoint.get('dataset_metadata', {})
    return trainer, checkpoint
