import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

PROJECT = "pothole"

def class_name(name):
    return name.replace("_", " ").strip() if PROJECT == "tsr" else name.strip().lower()

def is_image(path):
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"} or bool(re.search(r"\.jpg(?:\b|[^a-z0-9].*)$", path.name.lower()))

def classes_from(root):
    names = [class_name(p.name) for p in root.iterdir() if p.is_dir()]
    names = sorted(set(n for n in names if not (PROJECT == "tsr" and n == "Speed Limit 10")))
    if not names:
        raise ValueError(f"No class directories: {root}")
    if PROJECT == "pothole" and names != ["normal", "pothole"]:
        raise ValueError(f"Expected normal and pothole, found {names}")
    return names

class SplitDataset(Dataset):
    def __init__(self, root, names, transform):
        self.transform = transform
        self.samples = [(p, names.index(class_name(p.parent.name))) for p in sorted(root.glob("*/*")) if p.is_file() and is_image(p) and class_name(p.parent.name) in names]
        if not self.samples:
            raise ValueError(f"No images in {root}")
        self.targets = [y for _, y in self.samples]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        with Image.open(p) as im:
            return self.transform(im.convert("RGB")), y

def get_transforms(name):
    IMG_SIZE = 224
    BATCH_SIZE = 32

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, val_test_transforms

def build_model(name, count, pretrained=False):
    if name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, count)
        return model
    import timm
    return timm.create_model({"vit":"vit_base_patch16_224", "swin":"swin_tiny_patch4_window7_224"}[name], pretrained=pretrained, num_classes=count)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    loss_sum = 0.0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    return loss_sum / len(y_true), y_true, y_pred

def save_report(model, loader, device, names, out, split):
    loss, truth, pred = evaluate(model, loader, device)
    report = classification_report(truth, pred, labels=list(range(len(names))), target_names=names, output_dict=True, zero_division=0)
    report["loss"] = loss
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{split}_report.json").write_text(json.dumps(report, indent=2))
    matrix = confusion_matrix(truth, pred, labels=list(range(len(names))))
    with (out / f"{split}_confusion_matrix.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["true/predicted", *names]); writer.writerows([[n, *row] for n, row in zip(names, matrix)])
    print(split, "accuracy", sum(a == b for a,b in zip(truth,pred))/len(truth), "macro F1", report["macro avg"]["f1-score"])

def main():
    p = argparse.ArgumentParser(description="Train or evaluate image classifiers")
    p.add_argument("mode", choices=["train", "evaluate"])
    p.add_argument("--model", choices=['vit', 'swin', 'densenet121'], required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/classification"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", nargs="+", default=['test'])
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_root = a.data / 'train'
    names = classes_from(train_root)
    train_tfms, val_tfms = get_transforms(a.model)
    model = build_model(a.model, len(names), pretrained=a.mode == "train").to(device)
    def loader(split, tfms, training=False):
        ds = SplitDataset(a.data / split, names, tfms)
        sampler = None
        if training and PROJECT == "tsr":
            counts = np.bincount(ds.targets, minlength=len(names))
            sampler = WeightedRandomSampler([1.0/counts[y] for y in ds.targets], len(ds), replacement=True)
        return DataLoader(ds, batch_size=a.batch_size, sampler=sampler, shuffle=training and sampler is None, num_workers=0, pin_memory=device.type == "cuda")
    if a.mode == "train":
        tr, va = loader('train', train_tfms, True), loader("val", val_tfms)
        lr = 3e-4 if a.model == "densenet121" else 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) if a.model == "resnet50" else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5) if PROJECT == "pothole" else None
        a.weights.parent.mkdir(parents=True, exist_ok=True)
        if a.weights.exists():
            raise FileExistsError(f"Refusing to overwrite {a.weights}; choose a new path.")
        best, stale = float("inf"), 0
        for epoch in range(a.epochs):
            model.train()
            for x, y in tr:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(); loss = nn.functional.cross_entropy(model(x), y); loss.backward(); optimizer.step()
            loss, truth, pred = evaluate(model, va, device)
            acc = sum(x == y for x,y in zip(truth,pred))/len(truth)
            if scheduler: scheduler.step(loss)
            score = -acc if PROJECT == "tsr" else loss
            print(f"Epoch {epoch+1}: validation loss={loss:.4f}, accuracy={acc:.4f}")
            if score < best:
                best, stale = score, 0
                torch.save({"model_state_dict":model.state_dict(), "classes":names, "architecture":a.model}, a.weights)
            else:
                stale += 1
            if PROJECT == "pothole" and stale >= (5 if a.model == "densenet121" else 3): break
    state = torch.load(a.weights, map_location=device, weights_only=True)
    if isinstance(state, dict) and "classes" in state and state["classes"] != names:
        raise ValueError("Checkpoint class order differs from training dataset")
    if isinstance(state, dict) and "architecture" in state and state["architecture"] != a.model:
        raise ValueError("Checkpoint architecture differs from --model")
    model.load_state_dict(state.get("model_state_dict", state.get("model_state", state)))
    for split in a.splits:
        save_report(model, loader(split, val_tfms), device, names, a.out, split)

if __name__ == "__main__":
    main()
