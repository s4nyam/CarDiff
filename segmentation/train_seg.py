import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Class Mapping
# =========================
PIXEL_MAP = {
    0: 0,      # Background
    102: 1,    # SC
    153: 2,    # MC
    255: 3     # DC
}

NUM_CLASSES = 4


# =========================
# Dataset
# =========================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        self.mask_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.images)

    def encode_mask(self, mask):
        mask_np = np.array(mask)
        encoded = np.zeros_like(mask_np)

        for pixel_val, class_id in PIXEL_MAP.items():
            encoded[mask_np == pixel_val] = class_id

        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.img_transform(image)
        mask = self.mask_resize(mask)
        mask = self.encode_mask(mask)

        return image, mask


# =========================
# Load Data
# =========================
train_dataset = SegmentationDataset(
    os.path.join(args.data_dir, "train"),
    os.path.join(args.data_dir, "trainannot"),
    args.img_size
)

test_dataset = SegmentationDataset(
    os.path.join(args.data_dir, "test"),
    os.path.join(args.data_dir, "testannot"),
    args.img_size
)

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

    # Per-class metrics
    for cls_id, cls_name in class_map.items():
        pred_cls = (pred == cls_id)
        target_cls = (target == cls_id)

        intersection = (pred_cls & target_cls).sum().float()
        union = pred_cls.sum().float() + target_cls.sum().float() - intersection

        dice = (2 * intersection + smooth) / (pred_cls.sum() + target_cls.sum() + smooth)
        iou = (intersection + smooth) / (union + smooth)

        results[f"{cls_name}_dice"] = dice.item()
        results[f"{cls_name}_iou"] = iou.item()

    # Combined regions
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

    # Evaluation
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

    print(f"Epoch [{epoch+1}/{args.epochs}] "
          f"Train Dice Loss: {epoch_loss:.4f}")


# =========================
# Save Results
# =========================
os.makedirs("results_org", exist_ok=True)

# Training loss CSV
pd.DataFrame({
    "epoch": range(1, args.epochs + 1),
    "train_dice_loss": train_losses
}).to_csv(f"results_org/{args.model}_train_loss.csv", index=False)

# Detailed test metrics CSV
pd.DataFrame(all_test_metrics).to_csv(
    f"results_org/{args.model}_test_detailed_metrics.csv",
    index=False
)

# Plot training loss
plt.figure()
plt.plot(train_losses)
plt.title(f"{args.model} - Training Dice Loss")
plt.xlabel("Epoch")
plt.ylabel("Dice Loss")
plt.savefig(f"results_org/{args.model}_train_loss.png")
plt.close()

print("Training complete. Detailed metrics saved.")