"""
CarDiff — Causally-Structured Diffusion for Anatomically-Aware Dental Caries Synthesis
========================================================================================

Package layout
--------------
vae.py             Frozen VAE encoder / decoder (latent-space diffusion)
mask_encoder.py    Mask encoder  C_m(M)
causal_module.py   Patch-graph GNN + Global Attention Block + FiLM conditioning
film_unet.py       FiLM-conditioned denoising U-Net  G_{ε_θ}
discriminators.py  Image discriminator  D_i  and  pair discriminator  D_p
seg_head.py        Segmentation consistency head  S(·)
augmentation.py    Mask augmentation  A(·)  for self-supervised Path B
losses.py          All CarDiff losses
model.py           CarDiffModel  – top-level module that wires everything together
cardiff_trainer.py Dual-path training loop
cardiff_pipeline.py Inference / sampling pipeline
"""

from .vae import DentalVAE
from .mask_encoder import MaskEncoder
from .causal_module import CausalModule
from .film_unet import FiLMUNet
from .discriminators import ImageDiscriminator, PairDiscriminator
from .seg_head import SegmentationHead
from .augmentation import MaskAugmentation
from .losses import CarDiffLoss
from .model import CarDiffModel
from .cardiff_trainer import CarDiffTrainingConfig, cardiff_train_loop
from .cardiff_pipeline import CarDiffPipeline
