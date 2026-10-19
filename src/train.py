"""Fine-tune ImageNet-pretrained MobileNetV2 on the gesture data.

Phase 1 (--freeze-epochs): backbone frozen, only the new classifier trains (--lr).
Phase 2 (remaining epochs): whole network trains with a lower LR (--finetune-lr).
The checkpoint with the best validation accuracy is saved to models/fp32.pth.
The test set is NOT used here.
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import MODELS_DIR, RESULTS_DIR, SEED, ensure_dirs, set_seed
from dataset import GestureDataset
from model import build_model, set_backbone_trainable


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over loader. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def plot_curves(log, path):
    epochs = [e["epoch"] for e in log]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [e["train_loss"] for e in log], label="train")
    ax1.plot(epochs, [e["val_loss"] for e in log], label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(epochs, [e["train_acc"] for e in log], label="train")
    ax2.plot(epochs, [e["val_acc"] for e in log], label="val")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--freeze-epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="LR while backbone is frozen")
    ap.add_argument("--finetune-lr", type=float, default=1e-4, help="LR after unfreezing")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random init (only for offline smoke tests)")
    args = ap.parse_args()

    ensure_dirs()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = GestureDataset("train", augment=True)
    val_ds = GestureDataset("val", augment=False)
    g = torch.Generator().manual_seed(SEED)
    # num_workers=0 keeps things simple and works the same on Windows and Linux.
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    print(f"train={len(train_ds)} val={len(val_ds)} images, device={device}")

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: frozen backbone, train only the classifier.
    set_backbone_trainable(model, False)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    log, best_acc, best_loss = [], -1.0, float("inf")
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            # Phase 2: unfreeze everything and use a smaller learning rate so the
            # pretrained features are adjusted gently instead of being destroyed.
            set_backbone_trainable(model, True)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
        phase = "frozen" if epoch <= args.freeze_epochs else "finetune"

        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        lr = optimizer.param_groups[0]["lr"]

        # Best = highest val accuracy; ties broken by lower val loss.
        improved = va_acc > best_acc or (va_acc == best_acc and va_loss < best_loss)
        if improved:
            best_acc, best_loss = va_acc, va_loss
            torch.save(model.state_dict(), MODELS_DIR / "fp32.pth")

        log.append({"epoch": epoch, "phase": phase, "lr": lr,
                    "train_loss": tr_loss, "train_acc": tr_acc,
                    "val_loss": va_loss, "val_acc": va_acc, "saved": improved})
        print(f"epoch {epoch:>2}/{args.epochs} [{phase:<8}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"{time.time() - t0:.1f}s{'  *saved*' if improved else ''}")

    with open(RESULTS_DIR / "train_log.json", "w") as f:
        json.dump({"args": vars(args), "best_val_acc": best_acc,
                   "best_val_loss": best_loss, "epochs": log}, f, indent=2)
    plot_curves(log, RESULTS_DIR / "train_curves.png")
    print(f"Best val acc {best_acc:.4f}. Saved models/fp32.pth, "
          f"results/train_log.json, results/train_curves.png")


if __name__ == "__main__":
    main()
