import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp


# =========================
# Argument Parser
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="train_data")
parser.add_argument("--aug_img_dir", type=str, default="aug_img")
parser.add_argument("--aug_mask_dir", type=str, default="aug_seg")
parser.add_argument("--model", type=str, required=True,
                    choices=[
                        "Unet", "UnetPlusPlus", "MAnet", "Linknet",
                        "FPN", "PSPNet", "PAN"
                    ])
parser.add_argument("--encoder", type=str, default="resnet34")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--img_size", type=int, default=256)
parser.add_argument("--aug_multiplier", type=int, default=1)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(42)
np.random.seed(42)


# =========================
# Class Mapping
# =========================
PIXEL_MAP = {
    0: 0,
    102: 1,  # SC
    153: 2,  # MC
    255: 3   # DC
}
NUM_CLASSES = 4


# =========================
# Dataset
# =========================
class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths, img_size):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        self.mask_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.img_paths)

    def encode_mask(self, mask):
        mask_np = np.array(mask)
        encoded = np.zeros_like(mask_np)

        for pixel_val, class_id in PIXEL_MAP.items():
            encoded[mask_np == pixel_val] = class_id

        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image = self.img_transform(image)
        mask = self.mask_resize(mask)
        mask = self.encode_mask(mask)

        return image, mask


# =========================
# Load Splits
# =========================
def load_split(img_dir, mask_dir):
    imgs = sorted(os.listdir(img_dir))
    img_paths = [os.path.join(img_dir, x) for x in imgs]
    mask_paths = [os.path.join(mask_dir, x) for x in imgs]
    return img_paths, mask_paths


train_imgs, train_masks = load_split(
    os.path.join(args.data_dir, "train"),
    os.path.join(args.data_dir, "trainannot")
)

val_imgs, val_masks = load_split(
    os.path.join(args.data_dir, "val"),
    os.path.join(args.data_dir, "valannot")
)

test_imgs, test_masks = load_split(
    os.path.join(args.data_dir, "test"),
    os.path.join(args.data_dir, "testannot")
)


# =========================
# Augmentation (TRAIN ONLY)
# =========================
original_train_size = len(train_imgs)

if args.aug_multiplier > 1:
    target_train_size = original_train_size * args.aug_multiplier
    extra_needed = target_train_size - original_train_size

    aug_files = sorted(os.listdir(args.aug_img_dir))
    aug_img_paths = [os.path.join(args.aug_img_dir, x) for x in aug_files]
    aug_mask_paths = [os.path.join(args.aug_mask_dir, x) for x in aug_files]

    if extra_needed > len(aug_img_paths):
        print("Warning: Not enough augmented data. Sampling with replacement.")
        indices = np.random.choice(len(aug_img_paths), extra_needed, replace=True)
    else:
        indices = np.random.choice(len(aug_img_paths), extra_needed, replace=False)

    extra_imgs = [aug_img_paths[i] for i in indices]
    extra_masks = [aug_mask_paths[i] for i in indices]

    train_imgs += extra_imgs
    train_masks += extra_masks

print(f"Train size after augmentation: {len(train_imgs)}")
print(f"Validation size (unchanged): {len(val_imgs)}")
print(f"Test size (unchanged): {len(test_imgs)}")


# =========================
# DataLoaders
# =========================
train_dataset = SegmentationDataset(train_imgs, train_masks, args.img_size)
test_dataset = SegmentationDataset(test_imgs, test_masks, args.img_size)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)


# =========================
# Model
# =========================
model_class = getattr(smp, args.model)

model = model_class(
    encoder_name=args.encoder,
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
)

model.to(DEVICE)

dice_loss = smp.losses.DiceLoss(mode="multiclass")
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# =========================
# Metrics
# =========================
def compute_metrics(pred, target, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    results = {}

    class_map = {
        1: "SC",
        2: "MC",
        3: "DC"
    }

    for cls_id, cls_name in class_map.items():
        pred_cls = (pred == cls_id)
        target_cls = (target == cls_id)

        intersection = (pred_cls & target_cls).sum().float()
        union = pred_cls.sum().float() + target_cls.sum().float() - intersection

        dice = (2 * intersection + smooth) / (pred_cls.sum() + target_cls.sum() + smooth)
        iou = (intersection + smooth) / (union + smooth)

        results[f"{cls_name}_dice"] = dice.item()
        results[f"{cls_name}_iou"] = iou.item()

    combinations = {
        "MC_SC": [1, 2],
        "MC_DC": [2, 3],
        "SC_MC_DC": [1, 2, 3]
    }

    for name, cls_list in combinations.items():
        pred_comb = torch.zeros_like(pred, dtype=torch.bool)
        target_comb = torch.zeros_like(target, dtype=torch.bool)

        for cls in cls_list:
            pred_comb |= (pred == cls)
            target_comb |= (target == cls)

        intersection = (pred_comb & target_comb).sum().float()
        union = pred_comb.sum().float() + target_comb.sum().float() - intersection

        dice = (2 * intersection + smooth) / (pred_comb.sum() + target_comb.sum() + smooth)
        iou = (intersection + smooth) / (union + smooth)

        results[f"{name}_dice"] = dice.item()
        results[f"{name}_iou"] = iou.item()

    return results


# =========================
# Training Loop
# =========================
train_losses = []
all_test_metrics = []

for epoch in range(args.epochs):
    model.train()
    epoch_loss = 0

    for images, masks in tqdm(train_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    model.eval()
    epoch_metrics = {}

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            batch_metrics = compute_metrics(outputs, masks)

            for k, v in batch_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v

    for k in epoch_metrics:
        epoch_metrics[k] /= len(test_loader)

    epoch_metrics["epoch"] = epoch + 1
    all_test_metrics.append(epoch_metrics)

    print(f"Epoch {epoch+1}: Train Dice Loss={epoch_loss:.4f}")


# =========================
# Save Results
# =========================
os.makedirs("results_aug", exist_ok=True)

model_name = args.model
aug_mult = args.aug_multiplier

train_csv = f"results_aug/{model_name}_aug{aug_mult}_train_loss.csv"
test_csv = f"results_aug/{model_name}_aug{aug_mult}_test_detailed_metrics.csv"

pd.DataFrame({
    "epoch": range(1, args.epochs + 1),
    "train_dice_loss": train_losses
}).to_csv(train_csv, index=False)

pd.DataFrame(all_test_metrics).to_csv(test_csv, index=False)

print(f"Saved results for {model_name} with augmentation x{aug_mult}")