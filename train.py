import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import *
from dataset import get_dataloaders
from model import SERModel, count_parameters


# ---------------------------------------------------------------------------
# Training and evaluation steps
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(mel)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — important for LSTM stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)

        logits = model(mel)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="weighted")
    return total_loss / total, correct / total, f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(history["val_f1"], label="Val Weighted F1", color="green")
    axes[2].set_title("Weighted F1 (Val)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(labels, preds, save_path: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=EMOTIONS, yticklabels=EMOTIONS, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- Data ---
    train_loader, val_loader, test_loader, class_weights = get_dataloaders()

    # --- Model ---
    device = torch.device(DEVICE)
    model = SERModel().to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Training on: {device}\n")

    # --- Loss: weighted cross-entropy to handle class imbalance ---
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=LABEL_SMOOTHING)

    # --- Optimiser ---
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- LR scheduler ---
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR,
        patience=LR_PATIENCE
    )

    # --- Training loop ---
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_checkpoint = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS}  |  "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.3f}  |  "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.3f}  f1: {val_f1:.3f}"
        )

        # Checkpoint best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "emotions": EMOTIONS,
            }, best_checkpoint)
            print(f"  ✓ New best — saved checkpoint (val F1: {val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    # --- Save training history ---
    with open(os.path.join(CHECKPOINT_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, os.path.join(CHECKPOINT_DIR, "training_curves.png"))

    # --- Final evaluation on test set ---
    print("\n--- Test Set Evaluation ---")
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    print(f"Test accuracy: {test_acc:.4f}  |  Weighted F1: {test_f1:.4f}\n")
    print(classification_report(test_labels, test_preds, target_names=EMOTIONS))

    plot_confusion_matrix(
        test_labels, test_preds,
        os.path.join(CHECKPOINT_DIR, "confusion_matrix.png")
    )


if __name__ == "__main__":
    train()