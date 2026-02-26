# CarDiff

Causally-Structured Diffusion for Anatomically-Aware Dental Caries Synthesis.

## Training

```bash
# CarDiff with dual-path (Path A + Path B)
python main.py --cardiff --segmentation_guided --enable_path_b --mode train

# CarDiff Path A only
python main.py --cardiff --segmentation_guided --mode train

# With pre-trained VAE
python main.py --cardiff --segmentation_guided --enable_path_b --mode train --vae_pretrained_path /path/to/vae.pt

# Resume from checkpoint
python main.py --cardiff --segmentation_guided --enable_path_b --mode train --resume_epoch 100
```

## Inference

```bash
# Generate samples
python main.py --cardiff --segmentation_guided --mode eval_many --eval_sample_size 1000
```

## Baseline (original SegDiff-style, unchanged)

```bash
python main.py --segmentation_guided --mode train
python main.py --segmentation_guided --mode eval_many
```

## Key flags

| Flag | Description |
|---|---|
| `--cardiff` | Use CarDiff instead of baseline |
| `--enable_path_b` | Enable self-supervised Path B |
| `--path_b_start_epoch N` | Start Path B after N epochs (default 10) |
| `--vae_pretrained_path` | Path to pre-trained VAE weights |
| `--img_dir` | Image directory (default `DATA_FOLDER`) |
| `--seg_dir` | Mask directory (default `MASK_FOLDER`) |
| `--img_size` | Input resolution (default 384) |
| `--num_epochs` | Training epochs (default 2000) |
| `--train_batch_size` | Batch size (default 4) |

## Module structure

```
cardiff/
├── vae.py              # Frozen VAE encoder/decoder
├── mask_encoder.py     # Mask encoder C_m(M)
├── causal_module.py    # Patch-graph GNN + Global Attention + FiLM
├── film_unet.py        # FiLM-conditioned denoising U-Net
├── discriminators.py   # D_i (image) and D_p (pair)
├── seg_head.py         # Segmentation consistency head S(·)
├── augmentation.py     # Mask augmentation A(·) for Path B
├── losses.py           # All loss functions
├── model.py            # CarDiffModel (top-level)
├── cardiff_trainer.py  # Dual-path training loop
└── cardiff_pipeline.py # Inference pipeline
```
