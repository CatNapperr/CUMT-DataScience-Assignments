import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_preprocessing import load_and_preprocess, FEATURE_NAMES, LABEL_NAMES
from models import MLP, LSTMModel, LSTMAttention

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100
PATIENCE = 10
OUTPUT_DIR = "outputs"


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
    test_ds  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(-1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(-1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    acc = correct / total
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return total_loss / total, acc, p, r, f1, all_preds, all_labels, all_probs


def train_model(model, train_loader, val_loader, model_name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0
    best_weights = copy.deepcopy(model.state_dict())
    patience_left = PATIENCE
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _, _, _, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            patience_left = PATIENCE
        else:
            patience_left -= 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | patience={patience_left}")

        if patience_left == 0:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_weights)
    torch.save(best_weights, f"{OUTPUT_DIR}/{model_name}.pth")
    print(f"  Best val_acc={best_val_acc:.4f}, saved to {OUTPUT_DIR}/{model_name}.pth")
    return model, history


@torch.no_grad()
def test_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, p_macro, r_macro, f1_macro, preds, labels, probs = \
        evaluate(model, test_loader, criterion)
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1, 2, 3], zero_division=0)
    print(f"  Test Loss={test_loss:.4f}  Acc={test_acc:.4f}  "
          f"Macro-P={p_macro:.4f}  Macro-R={r_macro:.4f}  Macro-F1={f1_macro:.4f}")
    metrics = {
        "test_loss": round(test_loss, 6),
        "accuracy": round(test_acc, 6),
        "macro_precision": round(p_macro, 6),
        "macro_recall": round(r_macro, 6),
        "macro_f1": round(f1_macro, 6),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": round(float(p_per_class[i]), 6),
                "recall": round(float(r_per_class[i]), 6),
                "f1": round(float(f1_per_class[i]), 6),
                "support": int(support[i]),
            }
            for i in range(4)
        },
    }
    return metrics, preds, labels, probs


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), _, _, _ = load_and_preprocess(
        "data/NBA_Season_Stats.csv"
    )
    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    print(f"Device: {DEVICE}")
    print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}\n")

    models = {
        "MLP":            MLP(),
        "LSTM":           LSTMModel(),
        "LSTM_Attention": LSTMAttention(),
    }
    all_results = {}

    for name, model in models.items():
        print(f"=== Training {name} ===")
        model, history = train_model(model, train_loader, val_loader, name)
        print(f"=== Testing {name} ===")
        metrics, preds, labels, probs = test_model(model, test_loader)
        all_results[name] = {"history": history, "metrics": metrics}
        print()

        # --- Persist predictions ---
        np.savez(f"{OUTPUT_DIR}/{name}_predictions.npz",
                 y_true=labels, y_pred=preds, y_prob=probs)
        print(f"  Predictions saved to {OUTPUT_DIR}/{name}_predictions.npz")

        # --- Persist evaluation metrics ---
        with open(f"{OUTPUT_DIR}/{name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  Metrics saved to {OUTPUT_DIR}/{name}_metrics.json")

        # --- Persist training log ---
        log = {
            "model": name,
            "config": {"batch_size": BATCH_SIZE, "lr": LR, "epochs": len(history["train_loss"])},
            "epochs": [
                {
                    "epoch": i + 1,
                    "train_loss": round(float(history["train_loss"][i]), 6),
                    "val_loss": round(float(history["val_loss"][i]), 6),
                    "val_acc": round(float(history["val_acc"][i]), 6),
                }
                for i in range(len(history["train_loss"]))
            ],
            "best_val_acc": round(float(max(history["val_acc"])), 6),
        }
        with open(f"{OUTPUT_DIR}/{name}_training_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
        print(f"  Training log saved to {OUTPUT_DIR}/{name}_training_log.json")

    print("\n" + "=" * 65)
    print(f"{'Model':<18} {'Acc':>8} {'Macro-P':>8} {'Macro-R':>8} {'Macro-F1':>8}")
    print("-" * 65)
    best_name, best_acc = "", 0
    for name, res in all_results.items():
        m = res["metrics"]
        print(f"{name:<18} {m['accuracy']:8.4f} {m['macro_precision']:8.4f} "
              f"{m['macro_recall']:8.4f} {m['macro_f1']:8.4f}")
        if m["accuracy"] > best_acc:
            best_acc, best_name = m["accuracy"], name
    print("-" * 65)
    print(f"Best: {best_name} (Acc={best_acc:.4f})")

    np.savez(f"{OUTPUT_DIR}/training_history.npz",
             MLP=all_results["MLP"]["history"],
             LSTM=all_results["LSTM"]["history"],
             LSTM_Attention=all_results["LSTM_Attention"]["history"])
    print(f"\nHistory saved to {OUTPUT_DIR}/training_history.npz")


if __name__ == "__main__":
    main()
