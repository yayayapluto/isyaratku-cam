"""Fine-tune EfficientNet-B3 A-Z BISINDO di atas bobot `recognizer`.

Dataset: <data>/train/<LETTER>/NNNN.jpg + <data>/val/<LETTER>/NNNN.jpg
(susunan keluaran `scripts/build_dataset.py`). Preprocessing sengaja
dilewatkan per citra (BGR->RGB + resize 260 + normalisasi ImageNet) supaya
identik dengan `recognizer.preprocess` di jalur inferensi.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # import recognizer dari akar repo

import recognizer  # noqa: E402


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def letter_labels() -> list[str]:
    """Urutan label A-Z dari checkpoint bobot awal (sumber kebenaran)."""
    if letter_labels._cache is None:  # type: ignore[attr-defined]
        _, letter_labels._cache = recognizer.build_model()  # type: ignore[attr-defined]
    return list(letter_labels._cache)  # type: ignore[attr-defined]


letter_labels._cache: Optional[list[str]] = None  # type: ignore[attr-defined]


def _splits_for(data_dir: str) -> tuple[str, str]:
    """Path train/val; folder val boleh tidak ada untuk dataset uji kecil."""
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(train_dir):
        raise SystemExit(f"ERROR: butuh folder {train_dir}")
    if not os.path.isdir(val_dir):
        val_dir = train_dir  # ponytail: val = train, cukup untuk smoke test
    return train_dir, val_dir


class LetterDataset(Dataset):
    """Pemuat citra per huruf; augmentasi hanya aktif di split train."""
    def __init__(self, root: str, labels: list[str], train: bool) -> None:
        self.labels = labels
        self.items: list[tuple[str, str]] = []
        for label in labels:
            letter_dir = os.path.join(root, label)
            if not os.path.isdir(letter_dir):
                continue
            for name in sorted(os.listdir(letter_dir)):
                if name.lower().endswith(IMAGE_EXTS):
                    self.items.append((os.path.join(letter_dir, name), label))
        self.augment = _build_augment() if train else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"citra tidak terbaca: {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
        if self.augment is not None:
            tensor = self.augment(tensor)
        # normalisasi setelah augmentasi: ColorJitter/rotation bekerja di skala
        # uint8, lalu diskala ke [0,1] + normalisasi ImageNet seperti recognizer.
        tensor = tensor.float().div_(255.0)
        mean = torch.tensor(recognizer.MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(recognizer.STD, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, self.labels.index(label)


def _build_augment() -> nn.Module:
    """Augmentasi train-only. Flip TIDAK dipakai: menurunkan akurasi terukur
    (68,65% -> 64,62%, docs/MODEL_SELECTION.md §4)."""
    from torchvision import transforms as T

    return T.Compose([
        T.ColorJitter(0.2, 0.2, 0.2),
        T.RandomRotation(15),
        T.RandomResizedCrop(recognizer.INPUT_SIZE, scale=(0.9, 1.0)),
    ])


def _load_pretrained(resume: Optional[str]) -> tuple[nn.Module, list[str]]:
    """Bobot awal: --resume bila ada, selain itu recognizer.build_model()."""
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model, labels = recognizer.build_model()
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        return model, labels
    return recognizer.build_model()


def _param_groups(model: nn.Module, lr: float, mode: str
                  ) -> list[dict]:
    """mode=head: backbone dibekukan (tidak masuk optimizer).
    mode=full: backbone lr = lr/10 lewat dua parameter-group."""
    backbone = list(model.features.parameters())
    head = [p for p in model.parameters() if not any(
        p is q for q in backbone
    )]
    if mode == "head":
        for p in backbone:
            p.requires_grad = False
        return [{"params": head, "lr": lr}]
    return [{"params": head, "lr": lr},
            {"params": backbone, "lr": lr / 10.0}]


def _has_images(letter_dir: str) -> bool:
    return any(n.lower().endswith(IMAGE_EXTS) for n in os.listdir(letter_dir))


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: str,
                    clip: float) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], clip
        )
        optimizer.step()
        total_loss += float(loss.item()) * targets.size(0)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += targets.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: str) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += targets.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def train(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    device = _resolve_device(args.device)
    train_dir, val_dir = _splits_for(args.data)
    letters = [
        letter for letter in letter_labels()
        if os.path.isdir(os.path.join(train_dir, letter))
        and _has_images(os.path.join(train_dir, letter))
    ]
    if not letters:
        raise SystemExit(f"ERROR: tidak ada huruf di {train_dir}")
    if len(letters) != 26:
        raise SystemExit(
            f"ERROR: hanya {len(letters)} huruf punya citra di {train_dir}; "
            f"butuh 26 (A-Z) untuk fine-tune. Build dulu: "
            "python scripts/build_dataset.py ... (lihat docs/training_plan.md §4)"
        )
    model, labels = _load_pretrained(args.resume)
    missing = [l for l in letters if l not in labels]
    if missing:
        raise SystemExit(
            f"ERROR: huruf dataset tidak ada di label bobot: {missing}"
        )
    model.to(device)

    train_set = LetterDataset(train_dir, letters, True)
    val_set = LetterDataset(val_dir, letters, False)

    for name, dataset, split_dir in (
        ("train", train_set, train_dir), ("val", val_set, val_dir)
    ):
        if len(dataset) == 0:
            raise SystemExit(
                f"ERROR: split {name} kosong di {split_dir}; "
                "jalankan build dulu (docs/training_plan.md §4)"
            )
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=0)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        _param_groups(model, args.lr, args.mode), weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    patience = 5 if args.mode == "head" else 3
    best_loss = float("inf")
    best_state: Optional[dict] = None
    best_epoch = 0
    stale = 0

    print(f"mode={args.mode} device={device} letters={len(letters)} "
          f"train={len(train_set)} val={len(val_set)} epochs={args.epochs} "
          f"batch={args.batch} lr={args.lr}")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 1.0
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"epoch {epoch:>2}/{args.epochs} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} "
              f"({time.time() - start:.1f}s)")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop: val_loss tidak membaik selama {patience} epoch")
                break

    if best_state is None:
        best_state = model.state_dict()
        best_epoch = args.epochs
    model.load_state_dict(best_state)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "class_to_idx": {label: i for i, label in enumerate(letters)},
                "val_loss": best_loss, "epoch": best_epoch},
               os.path.join(args.out, "model.pth"))
    with open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8") as fh:
        json.dump({"labels": letters, "val_loss": best_loss,
                   "epoch": best_epoch}, fh, indent=2)
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, default=str)

    final_loss, final_acc = evaluate(model, val_loader, criterion, device)
    print(f"terbaik epoch {best_epoch} val_loss={best_loss:.4f} "
          f"(setelah restore val_loss={final_loss:.4f} val_acc={final_acc:.4f})")
    print(args.out)
    return 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNet-B3 BISINDO A-Z."
    )
    parser.add_argument("--data", default="data/finetune",
                        help="folder dataset (train/<letter>, val/<letter>)")
    parser.add_argument("--out", default="models/bisindo_alphabet_finetuned",
                        help="folder keluaran model.pth + labels.json + config.json")
    parser.add_argument("--mode", choices=("head", "full"), default="head",
                        help="head = backbone beku; full = semua dilatih")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto",
                        choices=("auto", "cpu", "cuda"))
    parser.add_argument("--resume", default=None,
                        help="path .pth bobot awal "
                             "(default: recognizer.MODEL_PATH)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not os.path.isdir(args.data):
        raise SystemExit(f"ERROR: folder dataset tidak ada: {args.data}")
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
